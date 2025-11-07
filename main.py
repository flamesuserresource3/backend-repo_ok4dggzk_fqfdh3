import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal, Dict, Any
from database import db, create_document, get_documents
from schemas import Task, Note, Idea, Summary, Message, Setting

app = FastAPI(title="Rabbit AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatPayload(BaseModel):
    message: str

class CommandPayload(BaseModel):
    type: Literal['task','note','idea','summary']
    payload: Dict[str, Any]

class SettingsPayload(BaseModel):
    active: str
    keys: Dict[str, str]
    endpoint: Optional[str] = None
    customPrompt: Optional[str] = None

@app.get("/")
def root():
    return {"message": "Rabbit AI backend is running"}

@app.get("/summary-data")
def summary_data():
    try:
        tasks = get_documents('task')
        notes = get_documents('note')
        ideas = get_documents('idea')
    except Exception:
        # offline/local only
        tasks, notes, ideas = [], [], []
    return {"tasks": tasks, "notes": notes, "ideas": ideas}

@app.post("/command")
def command(cmd: CommandPayload):
    ts = cmd.payload.get('ts')
    if cmd.type == 'task':
        create_document('task', Task(**cmd.payload))
        return {"message": "Task added. I will remind you gently.", "tts": "Task added. I will remind you gently."}
    if cmd.type == 'note':
        create_document('note', Note(**cmd.payload))
        return {"message": "Noted. I saved it to your notebook.", "tts": "Noted. I saved it to your notebook."}
    if cmd.type == 'idea':
        create_document('idea', Idea(**cmd.payload))
        return {"message": "Nice idea! I put it on the board.", "tts": "Nice idea! I put it on the board."}
    if cmd.type == 'summary':
        create_document('summary', Summary(**cmd.payload))
        return {"message": "Summary saved.", "tts": "Summary saved."}
    raise HTTPException(400, "Unknown command")


def call_openai(prompt: str, key: str):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are Rabbit AI, a calm, witty, emotionally aware assistant. You can respond in English and Indonesian."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    r = requests.post(url, headers=headers, json=body, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def call_gemini(prompt: str, key: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, json=body, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return "I couldn't get a reply from Gemini right now."


def call_anthropic(prompt: str, key: str):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
        "system": "You are Rabbit AI, a calm, witty, emotionally aware assistant. Bilingual EN/ID.",
    }
    r = requests.post(url, headers=headers, json=body, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    try:
        return "".join([b.get("text", "") for b in data["content"]]).strip()
    except Exception:
        return "I couldn't get a reply from Anthropic right now."


def call_mistral(prompt: str, key: str):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": "You are Rabbit AI, a calm, witty, emotionally aware assistant. Bilingual EN/ID."},
            {"role": "user", "content": prompt},
        ],
    }
    r = requests.post(url, headers=headers, json=body, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def call_custom(prompt: str, endpoint: str):
    r = requests.post(endpoint, json={"prompt": prompt}, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    try:
        return r.json().get("reply") or r.text
    except Exception:
        return r.text


def get_settings_from_db() -> Optional[Setting]:
    try:
        s = get_documents('setting', {}, 1)
        if s:
            doc = s[0]
            return Setting(active=doc.get('active','openai'), keys=doc.get('keys',{}), endpoint=doc.get('endpoint'), customPrompt=doc.get('customPrompt'))
    except Exception:
        pass
    return None

@app.post("/chat")
def chat(payload: ChatPayload):
    prompt = payload.message

    settings = get_settings_from_db()
    custom_prompt = settings.customPrompt if settings and settings.customPrompt else "You are Rabbit AI, a calm, witty, emotionally aware assistant. Bilingual EN/ID."

    # choose provider
    provider = settings.active if settings else 'openai'
    key = (settings.keys.get(provider) if settings else None) if provider != 'custom' else None

    # Fallback to environment keys
    if not key:
        env_map = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'gemini': os.getenv('GEMINI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'mistral': os.getenv('MISTRAL_API_KEY'),
        }
        key = env_map.get(provider)

    try:
        if provider == 'openai' and key:
            reply = call_openai(f"{custom_prompt}\n\nUser: {prompt}", key)
        elif provider == 'gemini' and key:
            reply = call_gemini(f"{custom_prompt}\n\nUser: {prompt}", key)
        elif provider == 'anthropic' and key:
            reply = call_anthropic(f"{custom_prompt}\n\nUser: {prompt}", key)
        elif provider == 'mistral' and key:
            reply = call_mistral(f"{custom_prompt}\n\nUser: {prompt}", key)
        elif provider == 'custom' and settings and settings.endpoint:
            reply = call_custom(f"{custom_prompt}\n\nUser: {prompt}", settings.endpoint)
        else:
            reply = "I need an API key to talk to the selected model. Open settings to add one."
    except HTTPException as e:
        raise e
    except Exception as e:
        reply = "I'm having trouble connecting to the AI service. I'll keep your message and try again soon."

    # Log message
    try:
        create_document('message', Message(role='user', text=prompt))
        create_document('message', Message(role='assistant', text=reply))
    except Exception:
        pass

    return {"reply": reply}


@app.post("/settings")
def set_settings(payload: SettingsPayload):
    try:
        create_document('setting', payload.dict())
    except Exception:
        pass
    return {"status": "ok"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
