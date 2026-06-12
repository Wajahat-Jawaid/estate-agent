from __future__ import annotations
import os
import json
import re
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from mortgage_handler import MortgageConversationHandler
from rental_yield import estimate_rent

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

llm = ChatGroq(
    # model="llama-3.3-70b-versatile",
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

llm_fast = ChatGroq(
    # model="llama-3.3-70b-versatile",
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1  # low temp for structured decisions
)

# llm = ChatOpenAI(
#     model="gpt-5.4-mini",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.7
# )

# llm_fast = ChatOpenAI(
#     model="gpt-5.4-mini",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.1  # low temp for structured decisions
# )

# Per-user memory and search history
user_memories = {}
user_search_histories = {}

_mortgage_handler = MortgageConversationHandler()

def get_user_memory(user_id: str) -> ConversationBufferMemory:
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(return_messages=True)
    return user_memories[user_id]

def get_user_search_history(user_id: str) -> list:
    if user_id not in user_search_histories:
        user_search_histories[user_id] = []
    return user_search_histories[user_id]

def price_to_lacs(price_str: str) -> int:
    price_str = price_str.lower().strip()
    crore, lac = 0, 0
    crore_match = re.search(r'(\d+)\s*crore', price_str)
    lac_match = re.search(r'(\d+)\s*lac', price_str)
    if crore_match:
        crore = int(crore_match.group(1))
    if lac_match:
        lac = int(lac_match.group(1))
    return (crore * 100) + lac

def lacs_to_price(n: int) -> str:
    c, r = n // 100, n % 100
    if c and r:
        decimal = f"{c + r / 100:.2f}".rstrip('0').rstrip('.')
        return f"{decimal} crore"
    if c: return f"{c} crore"
    return f"{n} lac"

def extract_filters(query: str) -> dict:
    prompt = f"""Extract property search filters from this query as JSON. Only include filters explicitly mentioned.

Query: "{query}"

Return ONLY a valid JSON object with these possible keys (omit any not mentioned):
- locations (array of strings) — list EVERY location mentioned. If multiple locations mentioned (e.g. "DHA or Clifton"), list all. Omit if no location mentioned.
- types (array of strings) — list EVERY property type mentioned from: house/apartment/upper portion/lower portion/penthouse/farmhouse. Omit if not mentioned.
- bedrooms (integer)
- min_bathrooms (integer)
- max_price (string, in Pakistani format e.g. "2 crore 50 lac")
- features (array of strings)

Return only raw JSON. No explanation. No markdown. No backticks."""
    response = llm_fast.invoke([HumanMessage(content=prompt)])
    try:
        raw = response.content.strip()
        # strip markdown fences if present
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        return json.loads(raw)
    except:
        return {}

_REVERSE_MORTGAGE_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*(?:lakh|lac|crore)\s*(?:mein|main|me)\s*(?:kya|kia)\b",
    r"\bmonthly\s+budget\s*(?:hai|is|of)?\s*\d",
    r"\b\d+(?:\.\d+)?\s*(?:lakh|lac|crore)\s+(?:per|har)\s+(?:month|mahina|maheena)\b",
    r"\bmonthly\s+\d+(?:\.\d+)?\s*(?:lakh|lac|crore)\b",
    r"\b\d+(?:\.\d+)?\s*(?:lakh|lac|crore)\s*(?:monthly|per\s+month|har\s+month)\b",
    r"\bbudget\s+(?:hai|mera|meri|hamara)?\s*\d+\s*(?:lakh|lac|crore)\b",
]

_MORTGAGE_EXPLICIT_KW = {
    "home loan", "installment", "instalment", "mortgage",
    "down payment", "downpayment", "afford kar sakta", "afford ho sakta",
    "kitna loan", "loan milega", "bank loan", "qist", "qarz",
    "monthly kitna", "monthly payment", "interest rate", "bank financing",
    "finance kar sakta", "housing loan", "property loan", "griha loan",
}

_AFFORDABILITY_HINT_KW = {
    "too expensive", "bohot mehnga", "bohat mehnga", "bahut mehnga",
    "budget mein nahi", "budget nahi hai", "afford nahi",
    "ziada mehnga", "zyada mehnga", "budget se bahar",
    "affordable nahi", "mehnga lag raha", "thoda kam price chahiye",
}

# Investment intent — checked BEFORE mortgage regex so "invest X crore" never
# gets swallowed by REVERSE_MORTGAGE patterns.
_INVESTMENT_KW = {
    "invest", "investment", "best yield", "best return", "rental return",
    "rental income", "passive income", "kahan invest", "invest karna",
    "best area to invest", "highest yield", "maximum return", "paisa lagana",
    "paisa invest", "return milega", "yield chahiye",
}

def detect_investment_intent(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _INVESTMENT_KW)

def _extract_investment_budget(query: str) -> int | None:
    """Regex-only budget extraction for investment queries (no LLM call)."""
    q = query.lower()
    crore_m = re.search(r'(\d+(?:\.\d+)?)\s*crore', q)
    lac_m   = re.search(r'(\d+)\s*lac', q)
    if crore_m:
        return int(float(crore_m.group(1)) * 100) + (int(lac_m.group(1)) if lac_m else 0)
    if lac_m:
        return int(lac_m.group(1))
    return None


def detect_mortgage_intent(query: str) -> str | None:
    """Fast regex-based mortgage intent detection — no LLM call."""
    q = query.lower()
    for pattern in _REVERSE_MORTGAGE_PATTERNS:
        if re.search(pattern, q):
            return "REVERSE_MORTGAGE"
    if re.search(r"\bemi\b", q) or re.search(r"\bloan\b", q) or re.search(r"\binstall?ment\b", q) or re.search(r"\bafford\b", q):
        return "MORTGAGE_EXPLICIT"
    if any(kw in q for kw in _MORTGAGE_EXPLICIT_KW):
        return "MORTGAGE_EXPLICIT"
    if any(kw in q for kw in _AFFORDABILITY_HINT_KW):
        return "AFFORDABILITY_HINT"
    return None


def classify_query(query: str, search_history: list) -> dict:
    history_summary = ""
    for i, entry in enumerate(search_history):
        history_summary += f"Search {i}: {entry['topic']}\n"
    if not history_summary:
        history_summary = "(no previous searches)"

    prompt = f"""Classify this real estate chatbot query.

Previous searches:
{history_summary}

Current query: "{query}"

CLASSIFICATION RULES:

SMALLTALK — greetings, general conversation, out of scope (weather, jokes etc)

NEWSEARCH — mentions a new location, new property type, significantly different budget, or is exploring a new category

FOLLOWUP — asking about details of already shown properties, comparing shown properties, selecting by number ("1", "2", "3"), asking about agent/contact of shown properties

CHEAPER — wants cheaper/more affordable/lower price options than what was shown

LARGER — wants bigger/larger/more rooms than what was shown

IMAGES — wants to see photos/pictures/images of a property already shown. May reference a property number (e.g. "images of property 2", "show me photos of the first one")

MORTGAGE_EXPLICIT — user directly asks about EMI, loan, installment, or affordability calculation (kitna loan milega, EMI kya hogi, monthly installment, home loan, afford kar sakta hun)

AFFORDABILITY_HINT — user hints price is a concern but doesn't request a calculation (too expensive, mehnga hai, budget mein nahi, afford nahi kar sakta) — NOT a request for cheaper listings

REVERSE_MORTGAGE — user states a monthly budget and asks what property they can afford (2 lakh mein kya milega, monthly X budget hai, X per month afford kar sakta). The user is asking about AFFORDABILITY from INCOME, not capital deployment.

INVESTMENT — user has a lump-sum capital and wants to know which areas/properties give the best rental yield, with NO fixed location in mind. The user is choosing WHERE to deploy capital for passive income, not asking what they can afford.
  ✓ "invest 1 crore for best yield"
  ✓ "kahan invest karun best return ke liye"
  ✓ "best area to invest 50 lac for rental income"
  ✓ "passive income ke liye property kahan lun"
  ✗ NOT INVESTMENT if a specific location is stated ("invest in DHA?" → NEWSEARCH)
  ✗ NOT INVESTMENT if asking about monthly affordability → REVERSE_MORTGAGE

Return ONLY valid JSON with no explanation:
{{"type": "NEWSEARCH", "index": null, "property_num": null}}
{{"type": "FOLLOWUP", "index": 0, "property_num": null}}
{{"type": "CHEAPER", "index": 0, "property_num": null}}
{{"type": "LARGER", "index": 0, "property_num": null}}
{{"type": "SMALLTALK", "index": null, "property_num": null}}
{{"type": "IMAGES", "index": 0, "property_num": 2}}
{{"type": "MORTGAGE_EXPLICIT", "index": null, "property_num": null}}
{{"type": "AFFORDABILITY_HINT", "index": null, "property_num": null}}
{{"type": "REVERSE_MORTGAGE", "index": null, "property_num": null}}
{{"type": "INVESTMENT", "index": null, "property_num": null}}

Index = which previous search is being referenced. Default to last search index if unclear.
property_num = 1-based property number the user referenced (null if not specified)."""

    response = llm_fast.invoke([HumanMessage(content=prompt)])
    try:
        raw = response.content.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        result = json.loads(raw)
        print(f">> Classification: {result}")
        return result
    except:
        return {"type": "NEWSEARCH", "index": None}

def build_chroma_filter(filters: dict):
    conditions = []

    if filters.get("bedrooms") is not None:
        conditions.append({"bedrooms": {"$eq": filters["bedrooms"]}})

    if filters.get("min_bathrooms") is not None:
        conditions.append({"bathrooms": {"$gte": filters["min_bathrooms"]}})

    types = filters.get("types") or ([filters["type"]] if filters.get("type") else [])
    if types:
        valid_types = [t for t in types if isinstance(t, str)]
        if len(valid_types) == 1:
            conditions.append({"type": {"$eq": valid_types[0]}})
        elif len(valid_types) > 1:
            conditions.append({"type": {"$in": valid_types}})

    if filters.get("max_price") is not None:
        max_lacs = price_to_lacs(str(filters["max_price"]))
        if max_lacs > 0:
            conditions.append({"price_numeric": {"$lte": max_lacs}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}

def search_properties(query: str, filters: dict, k: int = 10) -> list:
    chroma_filter = build_chroma_filter(filters)
    locations = filters.get("locations") or []
    fetch_k = max(k * 4, 40)

    if len(locations) > 1:
        per_loc = []
        for loc in locations:
            loc_lower = loc.lower()
            q = f"{query} {loc}"
            try:
                res = vectorstore.similarity_search_with_score(q, k=fetch_k, filter=chroma_filter) if chroma_filter else vectorstore.similarity_search_with_score(q, k=fetch_k)
            except:
                res = vectorstore.similarity_search_with_score(q, k=fetch_k)

            loc_matches = sorted(
                [(doc, score) for doc, score in res if loc_lower in doc.metadata.get("location", "").lower()],
                key=lambda x: x[1]
            )
            per_loc.append(loc_matches)
            print(f">> {loc}: {len(loc_matches)} candidates")

        seen_ids = set()
        merged = []
        iters = [iter(b) for b in per_loc]
        exhausted = [False] * len(iters)
        while len(merged) < k and not all(exhausted):
            for i, it in enumerate(iters):
                if exhausted[i] or len(merged) >= k:
                    continue
                try:
                    doc, score = next(it)
                    doc_id = doc.metadata.get("id", doc.page_content[:80])
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        merged.append((doc, score))
                except StopIteration:
                    exhausted[i] = True
        return merged[:k]

    try:
        results = vectorstore.similarity_search_with_score(query, k=fetch_k, filter=chroma_filter) if chroma_filter else vectorstore.similarity_search_with_score(query, k=fetch_k)
    except Exception as e:
        print(f"Filter failed, falling back: {e}")
        results = vectorstore.similarity_search_with_score(query, k=fetch_k)

    if not results and chroma_filter:
        results = vectorstore.similarity_search_with_score(query, k=fetch_k)

    if locations:
        loc_lower = locations[0].lower()
        filtered = [(doc, score) for doc, score in results if loc_lower in doc.metadata.get("location", "").lower()]
        if filtered:
            results = filtered

    return results[:k]

def fetch_cheaper(locations: list, max_price_lacs: int, prev_filters: dict, k: int = 10) -> list:
    min_price_lacs = max(int(max_price_lacs * 0.50), 30)
    price_filter = {"$and": [
        {"price_numeric": {"$lte": max_price_lacs}},
        {"price_numeric": {"$gte": min_price_lacs}},
    ]}
    types = prev_filters.get("types", [])
    if types:
        type_cond = {"type": {"$eq": types[0]}} if len(types) == 1 else {"type": {"$in": types}}
        price_filter = {"$and": [price_filter, type_cond]}

    query = " ".join(locations) if locations else "property"
    seen_ids = {}
    for loc in (locations if locations else [""]):
        q = f"{query} {loc}".strip()
        try:
            res = vectorstore.similarity_search_with_score(q, k=200, filter=price_filter)
        except:
            try:
                simple = {"$and": [{"price_numeric": {"$lte": max_price_lacs}}, {"price_numeric": {"$gte": min_price_lacs}}]}
                res = vectorstore.similarity_search_with_score(q, k=200, filter=simple)
            except:
                res = vectorstore.similarity_search_with_score(q, k=200)

        for doc, score in res:
            if not loc or loc.lower() in doc.metadata.get("location", "").lower():
                doc_id = doc.metadata.get("id", doc.page_content[:50])
                if doc_id not in seen_ids:
                    seen_ids[doc_id] = (doc, score)

    sorted_results = sorted(seen_ids.values(), key=lambda x: int(x[0].metadata.get("price_numeric") or 0))
    return sorted_results[:k]

GRID_VISIBLE_COUNT = 10  # must match frontend renderCards(listings.slice(0, N))

def _results_to_listings(results: list) -> tuple:
    """
    Returns (matched_listings, context).
    context only covers the first GRID_VISIBLE_COUNT results so the LLM
    can only reference properties that are immediately visible in the grid.
    """
    context = ""
    matched_listings = []
    raw = list(results)
    if not raw:
        return matched_listings, context
    scores = [s for _, s in raw]
    best = min(scores)
    worst = max(scores)
    span = worst - best
    for i, (doc, score) in enumerate(raw):
        if i < GRID_VISIBLE_COUNT:
            context += f"\n---\n{doc.page_content}\n"
        norm = 95 if span == 0 else 95 - ((score - best) / span) * 23
        meta = doc.metadata
        rent = estimate_rent(
            price_numeric=int(meta.get("price_numeric") or 0),
            location=meta.get("location", ""),
            property_type=meta.get("type", "house"),
        )
        matched_listings.append({"metadata": meta, "score": round(norm), "rental_yield": rent})
    return matched_listings, context

def decide_actions(
    user_query: str,
    classification_type: str,
    matched_listings: list,
    filters: dict,
    search_history: list,
    channel: str,
    history_text: str
) -> dict:
    """
    Second LLM call — decides the follow-up message and contextual actions.
    Returns: {"follow_up": str|null, "actions": [{"id": str, "label": str}]}
    """
    no_results = len(matched_listings) == 0
    prices = [int(l["metadata"].get("price_numeric") or 0) for l in matched_listings if l["metadata"].get("price_numeric")]
    min_price = min(prices) if prices else 0
    max_price_shown = max(prices) if prices else 0
    bedrooms_shown = list(set([l["metadata"].get("bedrooms") for l in matched_listings if l["metadata"].get("bedrooms")]))
    has_price_filter = bool(filters.get("max_price"))
    locations = filters.get("locations", [])
    
    max_actions = 2 if channel == "whatsapp" else 3

    prompt = f"""You are deciding what follow-up options to offer after a real estate search.

Context:
- User query: "{user_query}"
- Query type: {classification_type}
- Properties found: {len(matched_listings)}
- Price range shown: {lacs_to_price(min_price)} to {lacs_to_price(max_price_shown)} {"(no price filter applied)" if not has_price_filter else "(price filter was used)"}
- Bedrooms shown: {bedrooms_shown}
- Locations: {locations}
- Channel: {channel}
- Conversation: {history_text[-200:] if history_text else "none"}

Available action IDs and when to use them:
- "cheaper": user might want cheaper options. Use if results found AND there's room to go cheaper AND no very tight price filter already applied
- "larger": user might want bigger properties. Use if results found AND properties aren't already very large (5+ beds)
- "contact": user wants agent contacts. Use if results found
- "new_search": user wants to search something different. Use if it makes sense contextually
- "increase_budget": user has no results and might consider higher budget. Use only if no_results=true
- "different_area": user has no results and might try different area. Use only if no_results=true

Rules:
- If SMALLTALK or greeting: return empty actions and null follow_up
- If no results: offer increase_budget and/or different_area only
- If results found: offer relevant actions only (max {max_actions})
- The follow_up message should be warm, contextual, and natural — not generic
- Never say "Anything else I can help with?" — be specific to context
- Match the language style of the conversation (Urdu/English)

Return ONLY valid JSON:
{{
  "follow_up": "contextual follow-up message or null",
  "actions": [
    {{"id": "action_id", "label": "emoji + short label"}}
  ]
}}

No explanation. No markdown. No backticks."""

    try:
        response = llm_fast.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        result = json.loads(raw)
        # Safety: cap actions at max
        result["actions"] = result.get("actions", [])[:max_actions]
        result["follow_up"] = result.get("follow_up")
        print(f">> Actions decided: {result}")
        return result
    except Exception as e:
        print(f">> Action decision failed: {e}")
        # Safe fallback
        if no_results:
            return {"follow_up": "Want to try a different area or adjust your budget?", "actions": [{"id": "different_area", "label": "🗺️ Different area"}, {"id": "increase_budget", "label": "💰 Adjust budget"}]}
        return {"follow_up": None, "actions": [{"id": "contact", "label": "📞 Agent contacts"}, {"id": "new_search", "label": "🔍 New search"}]}

def _handle_mortgage_intent(intent: str, user_id: str, query: str, search_history: list) -> dict:
    """Route a detected mortgage intent to the appropriate handler."""
    context_price_rupees = None
    if search_history:
        last_listings = search_history[-1].get("listings", [])
        if last_listings:
            price_numeric = last_listings[0]["metadata"].get("price_numeric")
            if price_numeric:
                context_price_rupees = int(price_numeric) * 100_000  # lacs → rupees

    if intent == "MORTGAGE_EXPLICIT":
        return _mortgage_handler.handle_mortgage_explicit(user_id, query, context_price_rupees=context_price_rupees)

    if intent == "AFFORDABILITY_HINT":
        return _mortgage_handler.handle_affordability_hint(user_id, query)

    if intent == "REVERSE_MORTGAGE":
        result = _mortgage_handler.handle_reverse_mortgage(user_id, query)
        if result["meta"].get("trigger_search"):
            max_lacs = result["meta"]["max_price_lacs"]
            filters = {"max_price": lacs_to_price(max_lacs)}
            search_results = search_properties(query, filters)
            search_results.sort(key=lambda x: int(x[0].metadata.get("price_numeric") or 0))
            matched_listings, context = _results_to_listings(search_results)
            result["listings"] = matched_listings
            topic = f"budget ≤ {lacs_to_price(max_lacs)}"
            search_history.append({
                "topic": topic,
                "listings": matched_listings,
                "context": context,
                "filters": filters,
            })
        return result

    return {"response": "Let me help with that mortgage question!", "listings": [], "filters": {}, "follow_up": None, "actions": [], "meta": {"no_results": False}}


def _handle_investment_intent(user_query: str, user_id: str, search_history: list, channel: str = "web") -> dict:
    """
    Structured investment response: ranks tiers by yield for a lump-sum budget,
    shows 2-3 tiers (highest yield first + a premium appreciation counterpoint),
    each with a representative listing pulled from real stock.
    """
    from collections import defaultdict

    budget_lacs = _extract_investment_budget(user_query)
    if not budget_lacs:
        return {
            "response": "What's your investment budget? E.g. 1 crore, 50 lac, 2 crore 50 lac.",
            "listings": [], "filters": {}, "follow_up": None, "actions": [],
            "meta": {"no_results": False, "action": "investment"},
        }

    budget_str = lacs_to_price(budget_lacs)

    try:
        raw = vectorstore.similarity_search_with_score(
            "rental investment property income yield",
            k=80,
            filter={"price_numeric": {"$lte": budget_lacs}},
        )
    except Exception:
        raw = vectorstore.similarity_search_with_score("property", k=80)
        raw = [(d, s) for d, s in raw if int(d.metadata.get("price_numeric") or 0) <= budget_lacs]

    if not raw:
        return {
            "response": f"No properties found under {budget_str}. Try a higher budget.",
            "listings": [], "filters": {}, "follow_up": None, "actions": [],
            "meta": {"no_results": True, "action": "investment"},
        }

    # ── Annotate every doc with its yield estimate ───────────────────────────
    annotated: list[tuple] = []  # (doc, score, rent_dict)
    for doc, score in raw:
        meta = doc.metadata
        rent = estimate_rent(
            price_numeric=int(meta.get("price_numeric") or 0),
            location=meta.get("location", ""),
            property_type=meta.get("type", "house"),
        )
        annotated.append((doc, score, rent))

    # ── Bucket into yield bands ───────────────────────────────────────────────
    # HIGH  ≥ 6.5 y_low  →  best monthly cash flow (Korangi-tier rates)
    # MID   ≥ 5.5 y_low  →  balanced yield + appreciation (Gulshan-tier rates)
    # PREMIUM < 5.5 y_low →  lowest yield but strongest resale (DHA/Clifton rates)
    _BAND_BRIEF = {
        "HIGH":    "Maximum monthly cash flow — suits income-focused investors. Weaker resale market.",
        "MID":     "Solid monthly cash flow with established demand and balanced appreciation.",
        "PREMIUM": "Lowest monthly yield but the strongest resale value and long-term capital appreciation.",
    }

    def _band(y_low: float) -> str:
        if y_low >= 6.5:
            return "HIGH"
        if y_low >= 5.5:
            return "MID"
        return "PREMIUM"

    band_docs: dict[str, list] = defaultdict(list)
    for doc, score, rent in annotated:
        band_docs[_band(rent["yield_low"])].append((doc, score, rent))

    # Sort bands: HIGH first (best yield), then MID, then PREMIUM.
    # Always include PREMIUM as appreciation counterpoint if it has listings.
    yield_order = [b for b in ("HIGH", "MID", "PREMIUM") if b in band_docs]
    display_bands: list[str] = list(yield_order[:2])
    if "PREMIUM" in band_docs and "PREMIUM" not in display_bands:
        display_bands.append("PREMIUM")

    # ── Representative listing per band (highest price = best quality) ────────
    def _rep(items: list) -> tuple:
        return max(items, key=lambda x: int(x[0].metadata.get("price_numeric") or 0))

    # ── Build structured response text ───────────────────────────────────────
    lines = [
        f"For {budget_str} aimed at rental income, here's where it works hardest "
        f"— these are area-level estimates, not guarantees:\n"
    ]

    for i, band in enumerate(display_bands):
        rep_doc, _, rep_rent = _rep(band_docs[band])
        meta = rep_doc.metadata
        area_name = meta.get("location", "").split(",")[0]

        pos_label = "Best yield" if i == 0 else ("Premium hold" if band == "PREMIUM" else "Balanced")
        y_low, y_high = rep_rent["yield_low"], rep_rent["yield_high"]

        lines.append(f"🔹 {pos_label} — {area_name} (est. {y_low}–{y_high}%)")
        lines.append(
            f"~PKR {rep_rent['monthly_rent_low']:,}–{rep_rent['monthly_rent_high']:,}/mo on a "
            f"{lacs_to_price(int(meta.get('price_numeric', 0)))} property. {_BAND_BRIEF[band]}"
        )
        lines.append(
            f"Example: {meta.get('title', 'Property').title()}, "
            f"{area_name} — {meta.get('price', '')}"
        )
        lines.append("")

    # ── LLM: 2-sentence tradeoff + next-step ─────────────────────────────────
    rep_high = _rep(band_docs[display_bands[0]])
    highest_area = rep_high[0].metadata.get("location", "").split(",")[0]
    prem_band    = next((b for b in display_bands if b == "PREMIUM"), display_bands[-1])
    premium_area = _rep(band_docs[prem_band])[0].metadata.get("location", "").split(",")[0]
    band_areas   = " / ".join(
        _rep(band_docs[b])[0].metadata.get("location", "").split(",")[0]
        for b in display_bands
    )

    tradeoff = llm_fast.invoke([HumanMessage(content=
        f"""Write exactly 2 sentences for a property investor:
1. Start with "Quick read:" — contrast {highest_area} (best monthly yield) vs {premium_area} (lower yield but stronger resale). Be specific and direct.
2. Ask if they want actual listings in any specific area ({band_areas}).
Budget: {budget_str}. Match language of: "{user_query}" (Urdu or English). Conversational, not formal."""
    )])
    lines.append(tradeoff.content.strip())

    # ── Build matched_listings for property cards ─────────────────────────────
    seen_ids: set = set()
    matched_listings = []
    for band in display_bands:
        rep_doc, _, rep_rent = _rep(band_docs[band])
        priority = [(rep_doc, rep_rent)] + [
            (d, r) for d, _, r in band_docs[band] if d is not rep_doc
        ]
        for doc, rent in priority[:3]:
            doc_id = doc.metadata.get("id")
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            matched_listings.append({"metadata": doc.metadata, "score": 85, "rental_yield": rent})

    filters = {"max_price": budget_str}
    search_history.append({
        "topic": f"investment ≤ {budget_str}",
        "listings": matched_listings,
        "context": "",
        "filters": filters,
    })
    print(f">> Investment: budget={budget_str}, bands={display_bands}, listings={len(matched_listings)}")

    return {
        "response": "\n".join(lines),
        "listings": matched_listings,
        "filters": filters,
        "follow_up": None,
        "actions": [{"id": "new_search", "label": "🔍 New search"}],
        "meta": {"no_results": False, "action": "investment"},
    }


def get_response(user_query: str, user_id: str = "web", channel: str = "web") -> dict:
    memory = get_user_memory(user_id)
    search_history = get_user_search_history(user_id)

    # Conversation history
    history = memory.chat_memory.messages
    history_text = ""

    # ── Mortgage slot filling: continue active mortgage conversation ──
    slot_result = _mortgage_handler.handle_slot_filling(user_id, user_query)
    if slot_result:
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(slot_result["response"])
        return slot_result

    # ── Investment intent pre-detection (keyword, no LLM) ──
    # Must run BEFORE mortgage regex so "invest 1 crore" never hits REVERSE_MORTGAGE.
    if detect_investment_intent(user_query):
        result = _handle_investment_intent(user_query, user_id, search_history, channel)
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(result["response"])
        return result

    # ── Mortgage intent pre-detection (regex, no LLM) ──
    mortgage_intent = detect_mortgage_intent(user_query)
    if mortgage_intent:
        result = _handle_mortgage_intent(mortgage_intent, user_id, user_query, search_history)
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(result["response"])
        return result

    # Quick small talk check before full classification
    quick_check = llm_fast.invoke([HumanMessage(content=f"""Is this a greeting or small talk unrelated to property search?
    Message: "{user_query}"
    Reply with only: SMALLTALK or PROPERTY""")])
    if quick_check.content.strip().upper() == "SMALLTALK":
        response = llm.invoke([
            SystemMessage(content="You are a friendly real estate assistant. Respond warmly to greetings, then ask what property they're looking for. Max 2 sentences. Match the user's language."),
            HumanMessage(content=user_query)
        ])
        ai_response = response.content
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(ai_response)
        return {"response": ai_response, "listings": [], "filters": {}, "follow_up": None, "actions": [], "meta": {"no_results": False, "action": None}}

    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    stripped = user_query.strip()
    matched_listings = None
    context = ""
    filters = {}
    action_label = None
    classification_type = "NEWSEARCH"

    # Handle numeric shortcuts (4=cheaper, 5=larger, 6=contact)
    if stripped in ("4", "5", "6") and search_history:
        last = search_history[-1]
        prev_filters = last.get("filters", {})
        prev_listings = last["listings"]
        locations = prev_filters.get("locations", [])

        if stripped == "4":
            action_label = "cheaper"
            classification_type = "CHEAPER"
            shown = prev_listings[:10]
            prices = [int(l["metadata"].get("price_numeric") or 0) for l in shown if l["metadata"].get("price_numeric")]
            if prices:
                max_price_lacs = int(min(prices)) - 1
                filters = {k: v for k, v in prev_filters.items()}
                filters["max_price"] = lacs_to_price(max_price_lacs)
                results = fetch_cheaper(locations, max_price_lacs, prev_filters, k=10)
                matched_listings, context = _results_to_listings(results)
                search_history.append({"topic": f"cheaper · {last['topic']}", "listings": matched_listings, "context": context, "filters": filters})
                user_query = "show cheaper property options"
            else:
                matched_listings = []

        elif stripped == "5":
            action_label = "larger"
            classification_type = "LARGER"
            filters = {k: v for k, v in prev_filters.items()}
            shown_beds = [l["metadata"].get("bedrooms", 0) for l in prev_listings[:10] if l["metadata"].get("bedrooms")]
            base_beds = prev_filters.get("bedrooms") or (max(shown_beds) if shown_beds else 3)
            new_beds = base_beds + 1
            filters["bedrooms"] = new_beds
            loc_str = " ".join(locations) if locations else "properties"
            results = search_properties(f"{new_beds} bed {loc_str}", filters, k=10)
            results.sort(key=lambda x: int(x[0].metadata.get("price_numeric") or 0))
            matched_listings, context = _results_to_listings(results)
            search_history.append({"topic": f"larger · {last['topic']}", "listings": matched_listings, "context": context, "filters": filters})
            user_query = f"show {new_beds} bedroom properties"

        else:  # "6"
            matched_listings = prev_listings
            context = last["context"]
            filters = prev_filters
            user_query = "Share the full agent name, phone number and contact details for each property shown"
            classification_type = "FOLLOWUP"

    if matched_listings is None:
        # Digit follow-up shortcut
        if stripped.isdigit() and int(stripped) <= 9 and search_history:
            classification = {"type": "FOLLOWUP", "index": len(search_history) - 1}
            classification_type = "FOLLOWUP"
            print(f">> Classification: digit shortcut → FOLLOWUP")
        else:
            classification = classify_query(user_query, search_history)
            classification_type = classification["type"]

        # SMALLTALK
        if classification_type == "SMALLTALK":
            response = llm.invoke([
                SystemMessage(content="You are a friendly real estate assistant. Respond warmly and briefly to greetings or small talk, then naturally ask what property they are looking for. Max 2 sentences. Respond in the user's language (Urdu or English)."),
                HumanMessage(content=user_query)
            ])
            ai_response = response.content
            memory.chat_memory.add_user_message(user_query)
            memory.chat_memory.add_ai_message(ai_response)
            return {"response": ai_response, "listings": [], "filters": {}, "follow_up": None, "actions": [], "meta": {"no_results": False, "action": None}}

        # MORTGAGE intents (LLM fallback path — regex pre-check above handles most cases)
        if classification_type in ("MORTGAGE_EXPLICIT", "AFFORDABILITY_HINT", "REVERSE_MORTGAGE"):
            result = _handle_mortgage_intent(classification_type, user_id, user_query, search_history)
            memory.chat_memory.add_user_message(user_query)
            memory.chat_memory.add_ai_message(result["response"])
            return result

        # INVESTMENT (LLM fallback — keyword pre-check above handles most cases)
        if classification_type == "INVESTMENT":
            result = _handle_investment_intent(user_query, user_id, search_history, channel)
            memory.chat_memory.add_user_message(user_query)
            memory.chat_memory.add_ai_message(result["response"])
            return result

        # LARGER
        if classification_type == "LARGER" and search_history:
            action_label = "larger"
            idx = min(classification.get("index") or len(search_history) - 1, len(search_history) - 1)
            last = search_history[idx]
            prev_filters = last.get("filters", {})
            prev_listings = last["listings"]
            locations = prev_filters.get("locations", [])
            shown_beds = [l["metadata"].get("bedrooms", 0) for l in prev_listings[:10] if l["metadata"].get("bedrooms")]
            base_beds = prev_filters.get("bedrooms") or (max(shown_beds) if shown_beds else 3)
            new_beds = base_beds + 1
            filters = {k: v for k, v in prev_filters.items()}
            filters["bedrooms"] = new_beds
            loc_str = " ".join(locations) if locations else "properties"
            results = search_properties(f"{new_beds} bed {loc_str}", filters, k=10)
            results.sort(key=lambda x: int(x[0].metadata.get("price_numeric") or 0))
            matched_listings, context = _results_to_listings(results)
            search_history.append({"topic": f"larger · {last['topic']}", "listings": matched_listings, "context": context, "filters": filters})
            user_query = f"show {new_beds} bedroom properties"

        # CHEAPER
        elif classification_type == "CHEAPER" and search_history:
            action_label = "cheaper"
            idx = min(classification.get("index") or len(search_history) - 1, len(search_history) - 1)
            last = search_history[idx]
            prev_filters = last.get("filters", {})
            prev_listings = last["listings"]
            locations = prev_filters.get("locations", [])
            shown = prev_listings[:10]
            prices = [int(l["metadata"].get("price_numeric") or 0) for l in shown if l["metadata"].get("price_numeric")]
            if prices:
                max_price_lacs = int(min(prices)) - 1
                filters = {k: v for k, v in prev_filters.items()}
                filters["max_price"] = lacs_to_price(max_price_lacs)
                results = fetch_cheaper(locations, max_price_lacs, prev_filters, k=10)
                matched_listings, context = _results_to_listings(results)
                search_history.append({"topic": f"cheaper · {last['topic']}", "listings": matched_listings, "context": context, "filters": filters})
                user_query = "show cheaper property options"
            else:
                matched_listings = []

        # FOLLOWUP
        elif classification_type == "FOLLOWUP" and classification.get("index") is not None:
            idx = classification["index"]
            if idx >= len(search_history):
                idx = len(search_history) - 1
            matched_listings = search_history[idx]["listings"]
            context = search_history[idx]["context"]
            filters = {}
            if stripped.isdigit():
                selected = int(stripped) - 1
                if 0 <= selected < len(matched_listings):
                    matched_listings = [matched_listings[selected]]
                    context = matched_listings[0]["metadata"].get("description", context)

        # IMAGES
        elif classification_type == "IMAGES" and search_history:
            action_label = "images"
            idx = min((classification.get("index") or len(search_history) - 1), len(search_history) - 1)
            # Fall back to most recent search that actually had results
            while idx > 0 and not search_history[idx]["listings"]:
                idx -= 1
            prev_listings = search_history[idx]["listings"]

            # Resolve which property the user wants
            prop_num = classification.get("property_num")
            if not prop_num:
                num_match = re.search(r'\b([1-9])\b', user_query)
                prop_num = int(num_match.group(1)) if num_match else 1
            prop_idx = max(0, min(int(prop_num) - 1, len(prev_listings) - 1))
            target = prev_listings[prop_idx] if prev_listings else None

            if target:
                images_raw = target["metadata"].get("images", "[]")
                images_list = json.loads(images_raw) if isinstance(images_raw, str) else images_raw
                prop_title = target["metadata"].get("title", "Property").title()
                ai_response = f"Here are all the photos for *{prop_title}* 📸"
                memory.chat_memory.add_user_message(user_query)
                memory.chat_memory.add_ai_message(ai_response)
                return {
                    "response": ai_response,
                    "listings": [target],
                    "filters": {},
                    "follow_up": None,
                    "actions": [],
                    "meta": {
                        "no_results": False,
                        "action": "images",
                        "images_to_send": images_list,
                        "images_title": prop_title,
                    }
                }
            else:
                matched_listings = []

        # NEWSEARCH
        if matched_listings is None:
            filters = extract_filters(user_query)
            results = search_properties(user_query, filters)
            results.sort(key=lambda x: int(x[0].metadata.get("price_numeric") or 0))
            matched_listings, context = _results_to_listings(results)

            topic_parts = []
            if filters.get("locations"):
                topic_parts.append(" / ".join(filters["locations"]))
            if filters.get("types"):
                topic_parts.append("/".join(filters["types"]))
            if filters.get("bedrooms"):
                topic_parts.append(f"{filters['bedrooms']} bed")
            if filters.get("max_price"):
                topic_parts.append(f"under {filters['max_price']}")
            topic = ", ".join(topic_parts) if topic_parts else user_query[:60]
            search_history.append({"topic": topic, "listings": matched_listings, "context": context, "filters": filters})
            print(f">> New search saved as: '{topic}'")

    no_results = len(matched_listings) == 0

    # ── CALL 1: Generate conversational response ──
    if channel == "whatsapp" and not no_results:
        system_prompt = """You are a WhatsApp real estate assistant. Property cards will be sent right after your message — do NOT list or describe any properties.
Write ONE warm, natural sentence introducing the results. Examples: "Great news, found some solid options! 👇" or "Here's what we have for you 🏠"
Match the user's language (Urdu or English). One sentence only."""
    elif no_results:
        system_prompt = """You are a helpful real estate assistant. No properties were found.
Rules:
- ONE sentence saying nothing matched — friendly, not apologetic
- ONE sentence suggesting a specific alternative (different area, higher budget, or different type)
- Never say "I'm sorry", "I apologize", or anything formal
- Speak like a helpful friend
- Match the user's language (Urdu or English)"""
    else:
        system_prompt = f"""You are a helpful real estate assistant. Keep responses SHORT and NATURAL.
Rules:
- Maximum 2 sentences
- Never say "I'm sorry", "I apologize", "database", or anything formal
- Speak like a knowledgeable friend, not customer service
- NEVER repeat back criteria the user already stated in their query — they know what they searched for
- Instead, ADD information they couldn't already know: standout amenities, size (sq yd), what makes it worth considering, value vs other options, notable features (sea view, gated community, pool, etc.)
- If the query already specifies location + type + bedrooms, skip restating those — lead with price and then a differentiating detail
- Reference the best 1-2 options using their location and price (e.g. "the 2 crore house in DHA Phase 5")
- NEVER reference any property by ID number — not from current results and not from conversation history
- Never invent details not in the 'Available properties' section below
- Match the user's language (Urdu or English)

User's query (treat this as already-known context — do NOT repeat it back): "{user_query}" """

    user_prompt = f"""Conversation history:
{history_text}

User query: {user_query}

{"No matching properties found." if no_results else f"Available properties:{context}"}

{"Tell the user nothing matched and suggest what to try." if no_results else "Answer naturally based on the available properties above."}"""

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    ai_response = response.content

    memory.chat_memory.add_user_message(user_query)
    memory.chat_memory.add_ai_message(ai_response)

    # ── CALL 2: Decide follow-up actions ──
    action_decision = decide_actions(
        user_query=user_query,
        classification_type=classification_type,
        matched_listings=matched_listings,
        filters=filters,
        search_history=search_history,
        channel=channel,
        history_text=history_text
    )

    return {
        "response": ai_response,
        "listings": matched_listings,
        "filters": filters,
        "follow_up": action_decision.get("follow_up"),
        "actions": action_decision.get("actions", []),
        "meta": {
            "no_results": no_results,
            "action": action_label,
        }
    }