"""
Input guardrail layer (Step 2).

Loads the exported ONNX classifier once at startup and scores each incoming
prompt. Returns a decision (BLOCK / PASS), the per-label scores, and the list
of labels that fired above their threshold. PII handling and full GraphRAG
policy resolution come in later steps; this layer is the fast first gate.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

_CONFIG_PATH = Path("config/guardrail_config.json")
_ONNX_PATH = Path("model/guardrail_classifier.onnx")
_TOKENIZER_DIR = Path("model/lora_adapter")


class InputClassifier:
    def __init__(self):
        cfg = json.loads(_CONFIG_PATH.read_text())
        self.labels = cfg["label_names"]
        self.thresholds = cfg["label_thresholds"]
        self.max_len = cfg.get("max_length", 512)

        self.tokenizer = AutoTokenizer.from_pretrained(str(_TOKENIZER_DIR), use_fast=True)

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # silence the constant-fold warnings
        self.session = ort.InferenceSession(str(_ONNX_PATH), sess_options=opts, providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_names = [i.name for i in self.session.get_inputs()]

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _feed(self, text: str):
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len,
            return_tensors="np", return_token_type_ids=True,
        )
        feed = {}
        for name in self.input_names:
            if name in enc:
                feed[name] = enc[name].astype(np.int64)
            elif name == "token_type_ids":
                feed[name] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        return feed

    def classify(self, text: str) -> dict:
        logits = self.session.run(None, self._feed(text))[0][0]
        probs = self._sigmoid(logits)
        scores, fired = {}, []
        for label, p in zip(self.labels, probs):
            p = float(p)
            scores[label] = round(p, 4)
            if p >= self.thresholds.get(label, 0.5):
                fired.append(label)
        return {
            "decision": "BLOCK" if fired else "PASS",
            "fired_labels": fired,
            "scores": scores,
        }
