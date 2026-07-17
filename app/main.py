"""
GuardRail-OS gateway - Step 1: pure pass-through proxy.

Exposes an OpenAI-compatible POST /v1/chat/completions endpoint and forwards
requests to Gemini (Google AI Studio SDK). No guardrails yet - this step only
proves the plumbing: OpenAI-format request in, OpenAI-format response out.
"""

import os
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# ---- Config ----
load_dotenv()  # reads .env from the project root
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found. Put GEMINI_API_KEY=... in your .env file.")
genai.configure(api_key=API_KEY)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")

app = FastAPI(title="GuardRail-OS", version="0.1.0")


# ---- OpenAI-format request schema ----
class ChatMessage(BaseModel):
    role: str          # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ---- OpenAI <-> Gemini translation ----
def to_gemini(messages: List[ChatMessage]):
    """OpenAI messages -> (system_instruction, gemini contents)."""
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
        else:  # "user" or anything unknown -> treat as user
            contents.append({"role": "user", "parts": [m.content]})
    return system_instruction, contents


@app.get("/health")
def health():
    return {"status": "ok", "default_model": DEFAULT_MODEL}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
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

    # Map Gemini usage -> OpenAI usage shape
    um = getattr(resp, "usage_metadata", None)
    prompt_tokens = getattr(um, "prompt_token_count", 0) if um else 0
    completion_tokens = getattr(um, "candidates_token_count", 0) if um else 0

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
