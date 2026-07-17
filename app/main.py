"""
GuardRail-OS gateway - Step 5: LangGraph-orchestrated pipeline.

main.py is now thin: it builds the guardrail components + the LangGraph pipeline
once, then each request invokes the graph and this endpoint translates the final
graph state into an HTTP response. All branching (block/sanitize/pass, retry
cycle) lives in app/graph.py.
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
from app.graph import build_pipeline

# ---- Config ----
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key found. Put GEMINI_API_KEY=... in your .env file.")
genai.configure(api_key=API_KEY)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
MAX_RETRIES = 2

app = FastAPI(title="GuardRail-OS", version="0.5.0")

# ---- components (loaded once) ----
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
    required_json_fields: Optional[List[str]] = None


# ---- LLM translation + call (operates on plain dict messages) ----
def to_gemini(messages, extra_system=None):
    system_instruction = None
    contents = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            system_instruction = content if system_instruction is None else system_instruction + "\n" + content
        elif role == "assistant":
            contents.append({"role": "model", "parts": [content]})
        else:
            contents.append({"role": "user", "parts": [content]})
    if extra_system:
        system_instruction = (system_instruction + "\n" + extra_system) if system_instruction else extra_system
    return system_instruction, contents


def call_gemini(model_name, messages, gen_config, extra_system=None):
    system_instruction, contents = to_gemini(messages, extra_system)
    model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
    resp = model.generate_content(contents, generation_config=gen_config or None)
    return resp.text, resp


_REDACT_PATTERNS = [
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD_OR_ID]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\bcvv[:\s]*\d{3,4}\b", re.I), "cvv [REDACTED]"),
]


def sanitize_text(text: str) -> str:
    for pattern, repl in _REDACT_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ---- build the LangGraph pipeline once ----
pipeline = build_pipeline(classifier, policy_engine, output_guard, call_gemini, sanitize_text, MAX_RETRIES)


def latest_user_text(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


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
    gen_config = {}
    if req.temperature is not None:
        gen_config["temperature"] = req.temperature
    if req.max_tokens is not None:
        gen_config["max_output_tokens"] = req.max_tokens

    initial = {
        "messages": [m.model_dump() for m in req.messages],
        "user_text": latest_user_text(req.messages),
        "required_json_fields": req.required_json_fields,
        "model_name": req.model or DEFAULT_MODEL,
        "gen_config": gen_config,
        "attempt": 0,
    }

    final = pipeline.invoke(initial)

    decision = final["decision"]
    chain_str = " > ".join(decision["policy_chain"])
    input_score = final["input_score"]

    # ---- BLOCK ----
    if final.get("blocked"):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "blocked_by_guardrail",
                "fired_labels": final["fired_labels"],
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

    # ---- PASS / SANITIZE (delivered) ----
    out_eval = final["out_eval"]
    response.headers["X-GuardRail-Decision"] = decision["decision"]
    response.headers["X-GuardRail-Input-Score"] = f"{input_score:.4f}"
    response.headers["X-GuardRail-Output-Score"] = f"{out_eval['output_score']:.4f}"
    response.headers["X-GuardRail-Output-Check"] = final["output_check"]
    response.headers["X-GuardRail-Retries"] = str(final["attempt"])
    response.headers["X-GuardRail-Model-Used"] = final["model_name"]
    if decision["winning_policy"]:
        response.headers["X-GuardRail-Policy-Applied"] = decision["winning_policy"]
        response.headers["X-GuardRail-Policy-Chain"] = chain_str

    usage = final.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": final["model_name"],
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": final["text"]}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
        },
    }