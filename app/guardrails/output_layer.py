"""
Output guardrail layer (Step 4).

Runs on the buffered LLM response BEFORE it reaches the client:
  - TOXICITY   : Detoxify (torch, GPU if available). Flags harmful content.
  - SCHEMA     : Pydantic. If the caller requested JSON with required fields,
                 validate the response parses and contains them.
  - HALLUCINATION (proxy): cosine consistency between prompt and response using
                 the ONNX MiniLM embedder (same one ChromaDB uses - no torch).
                 Low similarity => possible topic drift. This is a lightweight
                 heuristic, not ground-truth hallucination detection.

Detoxify is optional: if torch / detoxify is not installed, that single check is
skipped gracefully and the layer still runs the other two.
"""

import json
from typing import Any, Optional

import numpy as np
from pydantic import create_model, ValidationError
from chromadb.utils import embedding_functions

# Toxicity model is optional (needs torch). Degrade gracefully if absent.
try:
    from detoxify import Detoxify
    _DETOX_AVAILABLE = True
except Exception:
    _DETOX_AVAILABLE = False


class OutputGuardrail:
    def __init__(
        self,
        toxicity_threshold: float = 0.5,
        consistency_threshold: float = 0.15,
        device: str = "auto",
    ):
        self.toxicity_threshold = toxicity_threshold
        self.consistency_threshold = consistency_threshold
        self.embedder = embedding_functions.DefaultEmbeddingFunction()

        self.detox = None
        self.detox_device = None
        if _DETOX_AVAILABLE:
            try:
                import torch
                dev = "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
                self.detox = Detoxify("original", device=dev)
                self.detox_device = dev
            except Exception:
                self.detox = None  # keep running without toxicity if load fails

    # ---- toxicity ----
    def check_toxicity(self, text: str) -> dict:
        if not self.detox:
            return {"available": False, "flagged": False, "scores": {}}
        scores = {k: float(v) for k, v in self.detox.predict(text).items()}
        top = max(scores, key=scores.get)
        return {
            "available": True,
            "flagged": scores[top] >= self.toxicity_threshold,
            "top_category": top,
            "top_score": round(scores[top], 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
        }

    # ---- hallucination proxy (embedding consistency) ----
    def consistency_score(self, prompt: str, response: str) -> dict:
        embs = self.embedder([prompt, response])
        a, b = np.array(embs[0]), np.array(embs[1])
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return {"consistency": round(cos, 4), "flagged": cos < self.consistency_threshold}

    # ---- schema (Pydantic) ----
    def check_schema(self, response: str, required_fields: Optional[list[str]]) -> dict:
        if not required_fields:
            return {"checked": False, "valid": True}
        try:
            parsed = json.loads(response)
        except Exception:
            return {"checked": True, "valid": False, "reason": "response is not valid JSON"}
        Model = create_model("ResponseSchema", **{f: (Any, ...) for f in required_fields})
        try:
            Model(**parsed)
            return {"checked": True, "valid": True}
        except ValidationError as e:
            return {"checked": True, "valid": False, "reason": str(e.errors()[:2])}

    # ---- combined ----
    def evaluate(self, prompt: str, response: str, required_fields: Optional[list[str]] = None) -> dict:
        tox = self.check_toxicity(response)
        hall = self.consistency_score(prompt, response)
        sch = self.check_schema(response, required_fields)

        reasons = []
        if tox["flagged"]:
            reasons.append(f"toxicity:{tox.get('top_category')}")
        if hall["flagged"]:
            reasons.append("low_consistency")
        if sch["checked"] and not sch["valid"]:
            reasons.append("schema_invalid")

        # blended output score in [0,1]: high = safe + on-topic
        base = max(hall["consistency"], 0.0)
        if tox["available"]:
            base = min(base, 1.0 - tox["top_score"])
        output_score = round(base, 4)

        return {
            "passed": len(reasons) == 0,
            "reasons": reasons,
            "output_score": output_score,
            "toxicity": tox,
            "hallucination": hall,
            "schema": sch,
        }
