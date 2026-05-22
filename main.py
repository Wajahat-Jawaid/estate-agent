import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent import get_response
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from fastapi.responses import PlainTextResponse
from fastapi import Form

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend files from /static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

class MessageRequest(BaseModel):
    message: str

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(request: MessageRequest):
    result = get_response(request.message)
    return {
        "response": result["response"],
        "listings": result["listings"],
        "filters": result["filters"]
    }

@app.post("/whatsapp", response_class=PlainTextResponse)
async def whatsapp_webhook(
    Body: str = Form(...),
    From: str = Form(...)
):
    from agent import get_response
    
    user_message = Body.strip()
    print(f"WhatsApp from {From}: {user_message}")
    
    result = get_response(user_message)
    ai_response = result["response"]
    
    # Format listings for WhatsApp
    if result["listings"]:
        ai_response += "\n\n🏠 *Matched Properties:*"
        for i, listing in enumerate(result["listings"][:3], 1):
            m = listing["metadata"]
            ai_response += f"\n\n*{i}. {m['title']}*"
            ai_response += f"\n💰 {m['price']}"
            ai_response += f"\n📍 {m['location']}"
            ai_response += f"\n👤 Agent: {m['agent']} — {m['contact']}"

    resp = MessagingResponse()
    resp.message(ai_response)
    return str(resp)

@app.post("/reset")
def reset():
    from agent import memory, search_history
    memory.chat_memory.clear()
    search_history.clear()
    return {"status": "reset"}

@app.get("/health")
def health():
    return {"status": "ok"}