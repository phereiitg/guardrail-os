"""
LangGraph orchestration for the guardrail pipeline.

The guardrail *components* (classifier, policy engine, output guard, LLM call)
are unchanged - this module only wires how they connect, as an explicit state
graph. The two genuinely graph-shaped parts are:
  - the BLOCK / continue branch after policy resolution (conditional edge)
  - the output retry loop (a cycle: output_check -> prepare_retry -> llm_call)

Nodes read/write a shared GraphState. The FastAPI endpoint invokes the compiled
graph and translates the final state into an HTTP response (it stays HTTP-free).
"""

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict, total=False):
    # inputs
    messages: list          # list of {"role","content"} dicts
    user_text: str
    required_json_fields: Optional[list]
    model_name: str
    gen_config: dict
    # produced by nodes
    fired_labels: list
    input_score: float
    decision: dict
    blocked: bool
    outgoing: list
    extra_system: Optional[str]
    attempt: int
    text: str
    usage: dict
    out_eval: dict
    output_check: str


def build_pipeline(classifier, policy_engine, output_guard, call_gemini, sanitize_text, max_retries=2):

    def input_check(state: GraphState) -> dict:
        v = classifier.classify(state["user_text"])
        scores = v["scores"]
        return {
            "fired_labels": v["fired_labels"],
            "input_score": max(scores.values()) if scores else 0.0,
        }

    def policy_resolve(state: GraphState) -> dict:
        d = policy_engine.resolve(state["fired_labels"], state["user_text"])
        return {"decision": d, "blocked": d["decision"] == "BLOCK"}

    def route_after_policy(state: GraphState) -> str:
        return "blocked" if state["blocked"] else "continue"

    def sanitize(state: GraphState) -> dict:
        if state["decision"]["decision"] == "SANITIZE":
            outgoing = [
                {"role": m["role"], "content": sanitize_text(m["content"])} if m["role"] == "user" else m
                for m in state["messages"]
            ]
        else:
            outgoing = state["messages"]
        return {"outgoing": outgoing, "attempt": 0, "extra_system": None}

    def llm_call(state: GraphState) -> dict:
        text, resp = call_gemini(
            state["model_name"], state["outgoing"], state["gen_config"], state.get("extra_system")
        )
        um = getattr(resp, "usage_metadata", None)
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", 0) if um else 0,
            "completion_tokens": getattr(um, "candidates_token_count", 0) if um else 0,
        }
        return {"text": text, "usage": usage}

    def output_check(state: GraphState) -> dict:
        ev = output_guard.evaluate(state["user_text"], state["text"], state.get("required_json_fields"))
        return {"out_eval": ev}

    def route_after_output(state: GraphState) -> str:
        if state["out_eval"]["passed"]:
            return "deliver"
        if state["attempt"] >= max_retries:
            return "deliver"
        return "retry"

    def prepare_retry(state: GraphState) -> dict:
        ev = state["out_eval"]
        fields = state.get("required_json_fields")
        extra = (
            "Your previous response was flagged for: " + ", ".join(ev["reasons"])
            + ". Provide a safe, non-toxic, on-topic response"
            + (f" as valid JSON containing fields {fields}" if fields else "")
            + "."
        )
        return {"attempt": state["attempt"] + 1, "extra_system": extra}

    def finalize(state: GraphState) -> dict:
        ev = state["out_eval"]
        if not ev["passed"]:
            return {
                "text": "I'm sorry, but I can't provide a safe response to that request.",
                "output_check": "FALLBACK",
            }
        return {"output_check": "PASS" if state["attempt"] == 0 else "RETRY_RECOVERED"}

    g = StateGraph(GraphState)
    g.add_node("input_check", input_check)
    g.add_node("policy_resolve", policy_resolve)
    g.add_node("sanitize", sanitize)
    g.add_node("llm_call", llm_call)
    g.add_node("output_check", output_check)
    g.add_node("prepare_retry", prepare_retry)
    g.add_node("finalize", finalize)

    g.add_edge(START, "input_check")
    g.add_edge("input_check", "policy_resolve")
    g.add_conditional_edges("policy_resolve", route_after_policy, {"blocked": END, "continue": "sanitize"})
    g.add_edge("sanitize", "llm_call")
    g.add_edge("llm_call", "output_check")
    g.add_conditional_edges("output_check", route_after_output, {"retry": "prepare_retry", "deliver": "finalize"})
    g.add_edge("prepare_retry", "llm_call")
    g.add_edge("finalize", END)

    return g.compile()
