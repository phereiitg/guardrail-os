"""
GuardRail-OS gateway - Step 2: input guardrail wired in.

Flow: request -> classify prompt -> BLOCK (403 + trace headers) or
PASS (forward to Gemini, attach safety-score header).
"""

import os
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

from app.guardrails.input_layer import InputClassifier

# ---- Config ----
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found. Put GEMINI_API_KEY=... in your .env file.")
genai.configure(api_key=API_KEY)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")

app = FastAPI(title="GuardRail-OS", version="0.2.0")

# Load the classifier once at startup (heavy: model load happens here, not per request)
classifier = InputClassifier()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


def to_gemini(messages: List[ChatMessage]):
    system_instruction = None
    contents = []
    for m in messages:
        if m.role == "system":
            system_instruction = (
                m.content if system_instruction is None
                else system_instruction + "\n" + m.content
            )
        elif m.role == "assistant":
            contents.append({"role": "model", "parts": [m.content]})
        else:
            contents.append({"role": "user", "parts": [m.content]})
    return system_instruction, contents


def latest_user_text(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


@app.get("/health")
def health():
    return {"status": "ok", "default_model": DEFAULT_MODEL, "provider": classifier.provider}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, response: Response):
    # ---- Input guardrail ----
    user_text = latest_user_text(req.messages)
    verdict = classifier.classify(user_text)
    input_score = max(verdict["scores"].values()) if verdict["scores"] else 0.0

    if verdict["decision"] == "BLOCK":
        raise HTTPException(
            status_code=403,
            detail={
                "error": "blocked_by_guardrail",
                "reason_categories": verdict["fired_labels"],
                "scores": verdict["scores"],
            },
            headers={
                "X-GuardRail-Decision": "BLOCK",
                "X-GuardRail-Reason": ",".join(verdict["fired_labels"]),
                "X-GuardRail-Input-Score": f"{input_score:.4f}",
            },
        )

    # ---- PASS: forward to Gemini ----
    model_name = req.model or DEFAULT_MODEL
    system_instruction, contents = to_gemini(req.messages)
    gen_config = {}
    if req.temperature is not None:
        gen_config["temperature"] = req.temperature
    if req.max_tokens is not None:
        gen_config["max_output_tokens"] = req.max_tokens

    try:
        model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
        resp = model.generate_content(contents, generation_config=gen_config or None)
        text = resp.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    um = getattr(resp, "usage_metadata", None)
    prompt_tokens = getattr(um, "prompt_token_count", 0) if um else 0
    completion_tokens = getattr(um, "candidates_token_count", 0) if um else 0

    response.headers["X-GuardRail-Decision"] = "PASS"
    response.headers["X-GuardRail-Input-Score"] = f"{input_score:.4f}"
    response.headers["X-GuardRail-Model-Used"] = model_name

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }