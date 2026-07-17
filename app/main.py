"""
GuardRail-OS gateway - Step 3a: classifier + GraphRAG policy engine.

Flow: request -> classify -> policy resolve -> act
  BLOCK    : 403 + full policy chain in headers, never reaches the LLM
  SANITIZE : redact PII from the prompt, forward the cleaned prompt to Gemini
  PASS     : forward unchanged
"""

import os
import re
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

from app.guardrails.input_layer import InputClassifier
from app.guardrails.policy_rag import PolicyEngine

# ---- Config ----
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found. Put GEMINI_API_KEY=... in your .env file.")
genai.configure(api_key=API_KEY)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")

app = FastAPI(title="GuardRail-OS", version="0.3.0")

classifier = InputClassifier()
policy_engine = PolicyEngine()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ---- minimal, real PII redaction for the SANITIZE path (spaCy NER comes later) ----
_REDACT_PATTERNS = [
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD_OR_ID]"),   # long digit runs (cards, Aadhaar)
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),   # emails
    (re.compile(r"\bcvv[:\s]*\d{3,4}\b", re.I), "cvv [REDACTED]"),       # cvv
]


def sanitize_text(text: str) -> str:
    for pattern, repl in _REDACT_PATTERNS:
        text = pattern.sub(repl, text)
    return text


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


@app.get("/policy-graph")
def policy_graph():
    """Inspection endpoint - shows the graph structure (nice for a demo)."""
    return policy_engine.summary()


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, response: Response):
    user_text = latest_user_text(req.messages)

    # ---- 1. classify ----
    verdict = classifier.classify(user_text)
    fired = verdict["fired_labels"]
    input_score = max(verdict["scores"].values()) if verdict["scores"] else 0.0

    # ---- 2. resolve against the policy graph ----
    decision = policy_engine.resolve(fired, user_text)
    chain_str = " > ".join(decision["policy_chain"])
    regs_str = ",".join(decision["governing_regulations"])

    # ---- 3a. BLOCK ----
    if decision["decision"] == "BLOCK":
        raise HTTPException(
            status_code=403,
            detail={
                "error": "blocked_by_guardrail",
                "fired_labels": fired,
                "winning_policy": decision["winning_policy"],
                "policy_chain": decision["policy_chain"],
                "governing_regulations": decision["governing_regulations"],
                "conflicts": decision["conflicts"],
            },
            headers={
                "X-GuardRail-Decision": "BLOCK",
                "X-GuardRail-Policy-Applied": decision["winning_policy"] or "",
                "X-GuardRail-Policy-Chain": chain_str,
                "X-GuardRail-Regulations": regs_str,
                "X-GuardRail-Input-Score": f"{input_score:.4f}",
            },
        )

    # ---- 3b. SANITIZE: redact prompt, then forward ----
    outgoing = req.messages
    if decision["decision"] == "SANITIZE":
        outgoing = [
            ChatMessage(role=m.role, content=sanitize_text(m.content)) if m.role == "user" else m
            for m in req.messages
        ]

    # ---- 3c. forward to Gemini (PASS or sanitized) ----
    model_name = req.model or DEFAULT_MODEL
    system_instruction, contents = to_gemini(outgoing)
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

    response.headers["X-GuardRail-Decision"] = decision["decision"]  # PASS or SANITIZE
    response.headers["X-GuardRail-Input-Score"] = f"{input_score:.4f}"
    response.headers["X-GuardRail-Model-Used"] = model_name
    if decision["winning_policy"]:
        response.headers["X-GuardRail-Policy-Applied"] = decision["winning_policy"]
        response.headers["X-GuardRail-Policy-Chain"] = chain_str

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }