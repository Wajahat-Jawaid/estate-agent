import os
import asyncio
import json as _json
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
import httpx
from agent import get_response, get_user_memory, user_memories, user_search_histories, price_to_lacs, lacs_to_price
from mortgage_handler import user_mortgage_states

# ── Persist welcomed users across restarts ──
_WELCOMED_FILE = "data/welcomed_users.json"

def _load_welcomed() -> set:
    try:
        with open(_WELCOMED_FILE) as f:
            return set(_json.load(f))
    except (FileNotFoundError, ValueError):
        return set()

def _save_welcomed(s: set):
    os.makedirs("data", exist_ok=True)
    with open(_WELCOMED_FILE, "w") as f:
        _json.dump(list(s), f)

welcomed_users: set = _load_welcomed()

# wamid → {"from_number": str, "listing": dict}  — populated when we send property cards
reply_context_map: dict = {}

def _is_image_request(text: str) -> bool:
    kw = ["image", "images", "photo", "photos", "picture", "pictures", "pic", "pics", "gallery", "photos dikhao", "tasveer"]
    t = text.lower()
    return any(k in t for k in kw)

app = FastAPI()

WHATSAPP_TOKEN = "EAAVf3keuvicBRgEdmSZC58iV53f5pObNsq8M3PwY6cWelCcuVffplt7pYCBzZApwXM5vScoyMAm1LHSr12tNRrJQ8GwrIgQwQqtihMEflvjZCz60ZC0uJQrU4HQFoexbupIGG0uR7ShTRbuPhskCZCk4dqZBGy8JJrcjGAUMqVrRbDnNeVZAEkM0NUIuZASCrZBKmVHKhkpdae1oJZAx9hqe0fw7pcpRkeZBtOfbKkZAKp410IMUK7kHl05U3PpkZAilcMopYFx7IpOgH7olJ7nSh0Eu7QOV8"
PHONE_NUMBER_ID = "1078312188708342"
VERIFY_TOKEN = "estateagent123"
BASE_URL = "https://backlog-defective-retreat.ngrok-free.dev"

META_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
WA_HEADERS = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── WhatsApp message builders ──
def wa_text(to, body):
    return {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": body[:4096]}}

def wa_image(to, url, caption=""):
    img = {"link": url}
    if caption:
        img["caption"] = caption[:1024]
    return {"messaging_product": "whatsapp", "to": to, "type": "image", "image": img}

def wa_buttons(to, body, buttons):
    """Max 3 buttons."""
    return {
        "messaging_product": "whatsapp", "to": to, "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body[:1024]},
            "action": {"buttons": buttons[:3]}
        }
    }

def wa_list(to, body, button_label, rows):
    """List message — up to 10 rows."""
    return {
        "messaging_product": "whatsapp", "to": to, "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": body[:1024]},
            "action": {
                "button": button_label[:20],
                "sections": [{"title": "Options", "rows": rows[:10]}]
            }
        }
    }

def build_property_caption(m: dict, index: int, rental_yield = None) -> str:
    title = m.get("title", "Property").title()
    lines = []

    lines.append(f"*{index}. {title}*")
    lines.append("")

    raw_price = m.get("price", "N/A")
    if raw_price and raw_price != "N/A":
        price_lacs = price_to_lacs(raw_price)
        formatted_price = lacs_to_price(price_lacs) if price_lacs > 0 else raw_price
    else:
        formatted_price = "N/A"
    lines.append(f"*Price:* {formatted_price}")
    lines.append(f"*Location:* {m.get('location', 'N/A')}")

    rooms = ""
    if m.get("bedrooms"):
        rooms = f"{m['bedrooms']} Bed"
        if m.get("bathrooms"):
            rooms += f"  ·  {m['bathrooms']} Bath"
    if rooms:
        lines.append(f"*Beds:* {rooms}")
    if m.get("area_sqyd"):
        lines.append(f"*Size:* {m['area_sqyd']} sq yd")
    if m.get("map_url"):
        lines.append(f"*Map:* {m['map_url']}")

    if rental_yield:
        lo  = f"{rental_yield['monthly_rent_low']:,}"
        hi  = f"{rental_yield['monthly_rent_high']:,}"
        ylo = rental_yield['yield_low']
        yhi = rental_yield['yield_high']
        lines.append(f"💰 Est. rent: PKR {lo}–{hi}/mo (~{ylo}–{yhi}% yield, area est.)")

    agent = m.get("agent", "N/A")
    contact = m.get("contact", "")
    agent_str = f"{agent} ({contact})" if contact else agent
    lines.append("")
    lines.append(f"*Agent:* {agent_str}")

    return "\n".join(lines)

def actions_to_wa_rows(actions: list) -> list:
    """Convert LLM actions to WhatsApp list rows."""
    rows = []
    for a in actions:
        aid = a.get("id", "")
        label = a.get("label", aid)[:24]
        descriptions = {
            "cheaper": "Find more affordable options",
            "larger": "Show bigger properties",
            "contact": "Get agent phone numbers",
            "new_search": "Search for something different",
            "increase_budget": "Try with a higher budget",
            "different_area": "Explore other areas",
            "reset": "Clear and start fresh",
        }
        rows.append({
            "id": aid,
            "title": label,
            "description": descriptions.get(aid, "")
        })
    return rows

def actions_to_wa_buttons(actions: list) -> list:
    """Convert LLM actions to WhatsApp button format (max 3)."""
    buttons = []
    for a in actions[:3]:
        buttons.append({
            "type": "reply",
            "reply": {"id": a.get("id", "action"), "title": a.get("label", "Option")[:20]}
        })
    return buttons

# ── Web endpoints ──
class MessageRequest(BaseModel):
    message: str
    session_id: str = "web"

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(request: MessageRequest):
    result = get_response(request.message, user_id=request.session_id, channel="web")
    return {
        "response": result["response"],
        "listings": result["listings"],
        "filters": result["filters"],
        "follow_up": result.get("follow_up"),
        "actions": result.get("actions", []),
        "meta": result.get("meta", {}),
    }

# ── WhatsApp webhook ──
@app.get("/whatsapp")
async def verify_webhook(request: Request):
    params = request.query_params
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(content=params.get("hub.challenge", ""))
    return PlainTextResponse(content="Forbidden", status_code=403)

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    print(f">> Webhook: {data}")

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" not in value:
            return {"status": "no message"}

        message = value["messages"][0]
        from_number = message["from"]
        msg_type = message.get("type")
        print(f">> from: {from_number} | type: {msg_type}")

        # ── Welcome (first time only, persisted) ──
        if from_number not in welcomed_users:
            welcomed_users.add(from_number)
            _save_welcomed(welcomed_users)
            get_user_memory(from_number)
            welcome = (
                "👋 Welcome to *Theorrem Property Search!*\n\n"
                "I'm your AI-powered property assistant. Tell me what you're looking for — "
                "location, budget, bedrooms, or type — and I'll find the best matches instantly.\n\n"
                "You can search in English or Urdu 🏠"
            )
            async with httpx.AsyncClient() as client:
                r = await client.post(META_URL, headers=WA_HEADERS, json=wa_text(from_number, welcome))
                print(f">> Welcome: {r.status_code}")
            return {"status": "ok"}

        # ── Handle interactive replies (button or list) ──
        if msg_type == "interactive":
            interactive = message.get("interactive", {})
            itype = interactive.get("type")

            if itype == "button_reply":
                action_id = interactive.get("button_reply", {}).get("id", "")
            elif itype == "list_reply":
                action_id = interactive.get("list_reply", {}).get("id", "")
            else:
                action_id = ""

            print(f">> Interactive action: {action_id}")

            # Map action IDs to user queries that the agent understands
            action_query_map = {
                "cheaper": "show cheaper options",
                "larger": "show larger properties",
                "contact": "share agent contact details for all properties shown",
                "new_search": "I want to search for something different",
                "increase_budget": "I can increase my budget, show me more options",
                "different_area": "show me options in different areas",
                "reset": "__RESET__",
                "within_budget": "show me properties within my budget",
                "calculate_emi": "calculate EMI for this property",
            }

            if action_id == "reset" or action_query_map.get(action_id) == "__RESET__":
                user_memories.pop(from_number, None)
                user_search_histories.pop(from_number, None)
                user_mortgage_states.pop(from_number, None)
                async with httpx.AsyncClient() as client:
                    await client.post(META_URL, headers=WA_HEADERS,
                                      json=wa_text(from_number, "All cleared! What property are you looking for? 🏠"))
                return {"status": "ok"}

            # Treat as a regular message using the mapped query
            user_message = action_query_map.get(action_id, action_id)
            print(f">> Action mapped to query: '{user_message}'")

        else:
            # ── Regular text message ──
            if msg_type != "text":
                return {"status": "unsupported type"}

            user_message = message["text"]["body"]
            print(f">> Message: {user_message}")

            # Reset keywords
            if user_message.strip().lower() in ["reset", "start over", "exit", "clear"]:
                user_memories.pop(from_number, None)
                user_search_histories.pop(from_number, None)
                user_mortgage_states.pop(from_number, None)
                async with httpx.AsyncClient() as client:
                    await client.post(META_URL, headers=WA_HEADERS,
                                      json=wa_text(from_number, "All cleared! What property are you looking for? 🏠"))
                return {"status": "ok"}

        # ── Resolve reply context (user replied to a specific property card) ──
        replied_listing = None
        ctx_id = message.get("context", {}).get("id")
        if ctx_id:
            ctx = reply_context_map.get(ctx_id)
            if ctx and ctx.get("from_number") == from_number:
                replied_listing = ctx["listing"]
                print(f">> Reply to property: {replied_listing['metadata'].get('title')}")

        # If user replied to a property card asking for images — handle directly, bypass agent
        if replied_listing and _is_image_request(user_message):
            m = replied_listing["metadata"]
            images_raw = m.get("images", "[]")
            images_list = _json.loads(images_raw) if isinstance(images_raw, str) else images_raw
            prop_title = m.get("title", "Property").title()
            async with httpx.AsyncClient() as client:
                await client.post(META_URL, headers=WA_HEADERS,
                                  json=wa_text(from_number, f"Here are all the photos for *{prop_title}* 📸"))
                for img_url in images_list:
                    await client.post(META_URL, headers=WA_HEADERS,
                                      json=wa_image(from_number, img_url))
            return {"status": "ok"}

        # For any other reply, inject property context so the agent knows which property is meant
        if replied_listing:
            m = replied_listing["metadata"]
            user_message = (
                f"[Replying about: '{m.get('title', '')}' in {m.get('location', '')}] "
                f"{user_message}"
            )
            print(f">> Injected reply context: {user_message[:120]}")

        # ── Get AI response ──
        result = get_response(user_message, user_id=from_number, channel="whatsapp")
        ai_response = result["response"]
        listings = result["listings"]
        follow_up = result.get("follow_up")
        actions = result.get("actions", [])
        no_results = result.get("meta", {}).get("no_results", False)
        has_listings = len(listings) > 0

        images_to_send = result.get("meta", {}).get("images_to_send", [])
        images_title = result.get("meta", {}).get("images_title", "Property")

        async with httpx.AsyncClient() as client:

            if images_to_send:
                r = await client.post(META_URL, headers=WA_HEADERS, json=wa_text(from_number, ai_response))
                print(f">> Images intro: {r.status_code}")
                for img_url in images_to_send:
                    r = await client.post(META_URL, headers=WA_HEADERS,
                                          json=wa_image(from_number, img_url))
                    print(f">> Image: {r.status_code}")
                return {"status": "ok"}

            elif has_listings:
                # Send AI intro message first
                r = await client.post(META_URL, headers=WA_HEADERS, json=wa_text(from_number, ai_response))
                print(f">> Intro: {r.status_code} {r.text[:100]}")

                # Send each property card and track wamid for reply resolution
                for i, listing in enumerate(listings[:5], 1):
                    m = listing["metadata"]
                    caption = build_property_caption(m, i, listing.get("rental_yield"))
                    r = await client.post(META_URL, headers=WA_HEADERS, json=wa_text(from_number, caption))
                    print(f">> Property {i}: {r.status_code}")
                    try:
                        wamid = r.json()["messages"][0]["id"]
                        reply_context_map[wamid] = {"from_number": from_number, "listing": listing}
                    except Exception:
                        pass

                await asyncio.sleep(1)

                # Send follow-up with dynamic actions
                if follow_up and actions:
                    if len(actions) <= 3:
                        # Use buttons for up to 3 actions
                        buttons = actions_to_wa_buttons(actions)
                        r = await client.post(META_URL, headers=WA_HEADERS,
                                              json=wa_buttons(from_number, follow_up, buttons))
                    else:
                        # Use list message for more than 3 actions
                        rows = actions_to_wa_rows(actions)
                        r = await client.post(META_URL, headers=WA_HEADERS,
                                              json=wa_list(from_number, follow_up, "What's next?", rows))
                    print(f">> Follow-up+actions: {r.status_code} {r.text[:100]}")
                elif follow_up:
                    r = await client.post(META_URL, headers=WA_HEADERS, json=wa_text(from_number, follow_up))
                    print(f">> Follow-up text: {r.status_code}")

            elif no_results:
                # No results — show AI response with contextual actions
                if actions:
                    if len(actions) <= 3:
                        buttons = actions_to_wa_buttons(actions)
                        r = await client.post(META_URL, headers=WA_HEADERS,
                                              json=wa_buttons(from_number, ai_response[:1024], buttons))
                    else:
                        rows = actions_to_wa_rows(actions)
                        r = await client.post(META_URL, headers=WA_HEADERS,
                                              json=wa_list(from_number, ai_response[:1024], "Options", rows))
                else:
                    r = await client.post(META_URL, headers=WA_HEADERS, json=wa_text(from_number, ai_response))
                print(f">> No-results: {r.status_code} {r.text[:100]}")

            else:
                # No listings — plain text (small talk, mortgage result, affordability hint, etc.)
                r = await client.post(META_URL, headers=WA_HEADERS, json=wa_text(from_number, ai_response))
                print(f">> Text: {r.status_code}")
                # Send follow-up action buttons if present (e.g. after mortgage result)
                if follow_up and actions:
                    await asyncio.sleep(0.5)
                    if len(actions) <= 3:
                        buttons = actions_to_wa_buttons(actions)
                        r = await client.post(META_URL, headers=WA_HEADERS,
                                              json=wa_buttons(from_number, follow_up, buttons))
                    else:
                        rows = actions_to_wa_rows(actions)
                        r = await client.post(META_URL, headers=WA_HEADERS,
                                              json=wa_list(from_number, follow_up, "Options", rows))
                    print(f">> Actions: {r.status_code}")

        return {"status": "ok"}

    except Exception as e:
        import traceback
        print(f">> Error: {e}")
        print(traceback.format_exc())
        return {"status": "error", "detail": str(e)}

@app.post("/reset")
def reset():
    user_memories.clear()
    user_search_histories.clear()
    user_mortgage_states.clear()
    return {"status": "reset"}

@app.get("/health")
def health():
    return {"status": "ok"}