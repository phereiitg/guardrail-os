"""
GraphRAG policy engine (Step 3b) - DUAL-PATH retrieval + graph expansion + resolver.

Retrieval for a fired-labels set + prompt text:
  PATH 1 - ENTITY (hard lookup): each fired classifier label -> candidate policies.
           Keyword-gated policies only fire if a required keyword is present.
           This is the guaranteed, compliance-critical path.
  PATH 2 - SEMANTIC (ChromaDB): embed the prompt, retrieve the nearest policies by
           MEANING, keep only those within a distance threshold. Catches policies
           the classifier has no label for (e.g. GDPR right-to-erasure).

The two anchor sets are unioned, expanded up the NetworkX graph for the governing
regulations, then resolved by priority with conflict detection.

Embedding note: uses ChromaDB's ONNX MiniLM default embedder (no PyTorch), keeping
the serving path ONNX-only and consistent with the classifier.
"""

from pathlib import Path

import yaml
import networkx as nx
import chromadb
from chromadb.utils import embedding_functions

_POLICY_DIR = Path("config/policies")


class PolicyEngine:
    def __init__(
        self,
        policy_dir: Path = _POLICY_DIR,
        semantic: bool = True,
        top_k: int = 3,
        distance_threshold: float = 0.70,  # cosine distance; lower = more similar
    ):
        self.policies: dict[str, dict] = {}
        self.graph = nx.DiGraph()
        self.label_index: dict[str, list] = {}
        self.semantic_enabled = semantic
        self.top_k = top_k
        self.distance_threshold = distance_threshold

        self._load(policy_dir)
        if self.semantic_enabled:
            self._build_vector_store()

    # ---- load YAML -> policies + graph + label index ----
    def _load(self, policy_dir: Path):
        for path in sorted(Path(policy_dir).glob("*.yaml")):
            p = yaml.safe_load(path.read_text())
            pid = p["id"]
            self.policies[pid] = p

            leaf = ("policy", pid)
            domain = ("domain", p["domain"])
            regulation = ("regulation", p["regulation"])
            self.graph.add_node(leaf, kind="policy")
            self.graph.add_node(domain, kind="domain")
            self.graph.add_node(regulation, kind="regulation")
            self.graph.add_edge(leaf, domain)
            self.graph.add_edge(domain, regulation)

            for label in p.get("triggers", {}).get("classifier_labels", []):
                self.label_index.setdefault(label, []).append(pid)

    # ---- ChromaDB vector store (rebuilt fresh each startup) ----
    def _build_vector_store(self):
        self.chroma = chromadb.Client()
        ef = embedding_functions.DefaultEmbeddingFunction()  # ONNX MiniLM, no torch
        try:
            self.chroma.delete_collection("policies")
        except Exception:
            pass
        self.collection = self.chroma.create_collection(
            name="policies",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        ids, docs = [], []
        for pid, p in self.policies.items():
            kws = ", ".join(p.get("triggers", {}).get("keywords", []))
            docs.append(
                f"{pid}. {p['description'].strip()} "
                f"Keywords: {kws}. Domain: {p['domain']}. Regulation: {p['regulation']}."
            )
            ids.append(pid)
        self.collection.add(ids=ids, documents=docs)

    # ---- PATH 1: entity retrieval ----
    def _entity_hits(self, fired_labels: list[str], text: str) -> list[str]:
        text_low = text.lower()
        hits = []
        for label in fired_labels:
            for pid in self.label_index.get(label, []):
                p = self.policies[pid]
                kws = p.get("triggers", {}).get("keywords")
                if kws and not any(k.lower() in text_low for k in kws):
                    continue
                if pid not in hits:
                    hits.append(pid)
        return hits

    # ---- PATH 2: semantic retrieval ----
    def _semantic_hits(self, text: str) -> list[str]:
        if not self.semantic_enabled:
            return []
        res = self.collection.query(
            query_texts=[text],
            n_results=min(self.top_k, len(self.policies)),
        )
        hits = []
        for pid, dist in zip(res["ids"][0], res["distances"][0]):
            if dist <= self.distance_threshold:
                hits.append(pid)
        return hits

    # ---- graph expansion ----
    def _governing_regulations(self, policy_ids: list[str]) -> list[str]:
        regs = set()
        for pid in policy_ids:
            for node in nx.descendants(self.graph, ("policy", pid)):
                if node[0] == "regulation":
                    regs.add(node[1])
        return sorted(regs)

    def _chain(self, policy_id: str) -> list[str]:
        p = self.policies[policy_id]
        return [policy_id, p["domain"], p["regulation"]]

    # ---- resolver ----
    def resolve(self, fired_labels: list[str], text: str) -> dict:
        entity = self._entity_hits(fired_labels, text)
        semantic = self._semantic_hits(text)

        # union, remembering which path(s) found each policy
        sources: dict[str, list[str]] = {}
        for pid in entity:
            sources.setdefault(pid, []).append("entity")
        for pid in semantic:
            sources.setdefault(pid, []).append("semantic")

        applicable = [self.policies[pid] for pid in sources]
        if not applicable:
            return {
                "decision": "PASS", "winning_policy": None, "priority": None,
                "policy_chain": [], "applied_policies": [],
                "governing_regulations": [], "conflicts": [], "retrieval": {},
            }

        applicable.sort(key=lambda p: p["priority"], reverse=True)
        winner = applicable[0]
        applied_ids = [p["id"] for p in applicable]

        conflicts = []
        if len({p["action"] for p in applicable}) > 1:
            conflicts.append({
                "resolved_to": winner["action"],
                "winning_policy": winner["id"],
                "competing": [
                    {"id": p["id"], "action": p["action"], "priority": p["priority"]}
                    for p in applicable
                ],
            })

        return {
            "decision": winner["action"].upper(),
            "winning_policy": winner["id"],
            "priority": winner["priority"],
            "policy_chain": self._chain(winner["id"]),
            "applied_policies": applied_ids,
            "governing_regulations": self._governing_regulations(applied_ids),
            "conflicts": conflicts,
            "retrieval": sources,  # pid -> ["entity"] / ["semantic"] / ["entity","semantic"]
        }

    # ---- inspection helper ----
    def summary(self) -> dict:
        kinds = {"policy": 0, "domain": 0, "regulation": 0}
        for _, data in self.graph.nodes(data=True):
            kinds[data["kind"]] += 1
        multi_parent = {}
        for node, data in self.graph.nodes(data=True):
            if data["kind"] == "domain":
                parents = [n[1] for n in self.graph.successors(node)]
                if len(parents) > 1:
                    multi_parent[node[1]] = parents
        return {
            "counts": kinds,
            "multi_parent_domains": multi_parent,
            "total_policies": len(self.policies),
            "semantic_search": self.semantic_enabled,
            "semantic_distance_threshold": self.distance_threshold,
        }