"""
GuardRail-OS gateway - Step 4: input + policy + OUTPUT guardrails.

Flow: request -> classify -> policy resolve -> BLOCK/SANITIZE/PASS
      -> call Gemini -> OUTPUT check (toxicity/schema/consistency)
      -> retry up to 2x on failure -> safe fallback if still failing -> deliver
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
from app.guardrails.output_layer import OutputGuardrail

# ---- Config ----
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found. Put GEMINI_API_KEY=... in your .env file.")
genai.configure(api_key=API_KEY)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
MAX_RETRIES = 2

app = FastAPI(title="GuardRail-OS", version="0.4.0")

classifier = InputClassifier()
policy_engine = PolicyEngine()
output_guard = OutputGuardrail()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    required_json_fields: Optional[List[str]] = None  # triggers schema validation


_REDACT_PATTERNS = [
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD_OR_ID]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\bcvv[:\s]*\d{3,4}\b", re.I), "cvv [REDACTED]"),
]


def sanitize_text(text: str) -> str:
    for pattern, repl in _REDACT_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def to_gemini(messages: List[ChatMessage], extra_system: Optional[str] = None):
    system_instruction = None
    contents = []
    for m in messages:
        if m.role == "system":
            system_instruction = m.content if system_instruction is None else system_instruction + "\n" + m.content
        elif m.role == "assistant":
            contents.append({"role": "model", "parts": [m.content]})
        else:
            contents.append({"role": "user", "parts": [m.content]})
    if extra_system:
        system_instruction = (system_instruction + "\n" + extra_system) if system_instruction else extra_system
    return system_instruction, contents


def latest_user_text(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def call_gemini(model_name, messages, gen_config, extra_system=None) -> str:
    system_instruction, contents = to_gemini(messages, extra_system)
    model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
    resp = model.generate_content(contents, generation_config=gen_config or None)
    return resp.text, resp


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "classifier_provider": classifier.provider,
        "toxicity_device": output_guard.detox_device or "unavailable",
    }


@app.get("/policy-graph")
def policy_graph():
    return policy_engine.summary()


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, response: Response):
    user_text = latest_user_text(req.messages)

    # ---- INPUT: classify + resolve ----
    verdict = classifier.classify(user_text)
    fired = verdict["fired_labels"]
    input_score = max(verdict["scores"].values()) if verdict["scores"] else 0.0
    decision = policy_engine.resolve(fired, user_text)
    chain_str = " > ".join(decision["policy_chain"])

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
                "retrieval": decision["retrieval"],
            },
            headers={
                "X-GuardRail-Decision": "BLOCK",
                "X-GuardRail-Policy-Applied": decision["winning_policy"] or "",
                "X-GuardRail-Policy-Chain": chain_str,
                "X-GuardRail-Input-Score": f"{input_score:.4f}",
            },
        )

    outgoing = req.messages
    if decision["decision"] == "SANITIZE":
        outgoing = [
            ChatMessage(role=m.role, content=sanitize_text(m.content)) if m.role == "user" else m
            for m in req.messages
        ]

    model_name = req.model or DEFAULT_MODEL
    gen_config = {}
    if req.temperature is not None:
        gen_config["temperature"] = req.temperature
    if req.max_tokens is not None:
        gen_config["max_output_tokens"] = req.max_tokens

    # ---- LLM CALL + OUTPUT guardrail with retry ----
    attempt = 0
    extra_system = None
    out_eval = None
    text = ""
    last_resp = None
    while True:
        try:
            text, last_resp = call_gemini(model_name, outgoing, gen_config, extra_system)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

        out_eval = output_guard.evaluate(user_text, text, req.required_json_fields)
        if out_eval["passed"] or attempt >= MAX_RETRIES:
            break
        attempt += 1
        extra_system = (
            "Your previous response was flagged for: "
            + ", ".join(out_eval["reasons"])
            + ". Provide a safe, non-toxic, on-topic response"
            + (f" as valid JSON containing fields {req.required_json_fields}" if req.required_json_fields else "")
            + "."
        )

    if not out_eval["passed"]:
        text = "I'm sorry, but I can't provide a safe response to that request."
        output_check = "FALLBACK"
    else:
        output_check = "PASS" if attempt == 0 else "RETRY_RECOVERED"

    um = getattr(last_resp, "usage_metadata", None)
    prompt_tokens = getattr(um, "prompt_token_count", 0) if um else 0
    completion_tokens = getattr(um, "candidates_token_count", 0) if um else 0

    response.headers["X-GuardRail-Decision"] = decision["decision"]
    response.headers["X-GuardRail-Input-Score"] = f"{input_score:.4f}"
    response.headers["X-GuardRail-Output-Score"] = f"{out_eval['output_score']:.4f}"
    response.headers["X-GuardRail-Output-Check"] = output_check
    response.headers["X-GuardRail-Retries"] = str(attempt)
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