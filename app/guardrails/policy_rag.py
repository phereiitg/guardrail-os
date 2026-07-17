"""
GraphRAG policy engine (Step 3a) - entity path + graph expansion + resolver.

Pipeline for a fired-labels set:
  1. ENTITY RETRIEVAL: map each fired classifier label -> candidate policies
     (a hard lookup, guaranteed - this is the compliance-critical path).
     Keyword-gated policies only fire if a required keyword is present.
  2. GRAPH EXPANSION: walk upward in the NetworkX graph from each applicable
     leaf to collect the full set of governing domains + regulations (BFS
     reachability). This is what puts "PCI-DSS" in the audit trace, not just
     the leaf rule.
  3. RESOLVER: sort applicable policies by priority, detect action conflicts,
     highest priority wins.

Design note (be ready to defend this): the precise per-decision audit CHAIN
uses the winning policy's own declared leaf->domain->regulation, so we never
over-attribute (a general-PII redaction isn't claimed under PCI-DSS just
because it shares a domain). The graph's payoff is (a) storing a real DAG
where one domain has multiple regulation parents, and (b) answering structural
queries like "which frameworks govern everything that fired". For ~6 policies a
flat list would also work; the graph earns its place as the policy set grows
and gains multi-parent / cross-domain relationships.
"""

from pathlib import Path

import yaml
import networkx as nx

_POLICY_DIR = Path("config/policies")


class PolicyEngine:
    def __init__(self, policy_dir: Path = _POLICY_DIR):
        self.policies: dict[str, dict] = {}     # id -> policy dict
        self.graph = nx.DiGraph()               # edges point UP: leaf -> domain -> regulation
        self.label_index: dict[str, list] = {}  # classifier_label -> [policy_id]
        self._load(policy_dir)

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

    # ---- 1. entity-path retrieval (hard lookup, keyword-gated) ----
    def _applicable(self, fired_labels: list[str], text: str) -> list[dict]:
        text_low = text.lower()
        seen, out = set(), []
        for label in fired_labels:
            for pid in self.label_index.get(label, []):
                if pid in seen:
                    continue
                p = self.policies[pid]
                kws = p.get("triggers", {}).get("keywords")
                if kws and not any(k.lower() in text_low for k in kws):
                    continue  # keyword-gated policy, required keyword absent -> skip
                seen.add(pid)
                out.append(p)
        return out

    # ---- 2. graph expansion: frameworks governing a set of leaves ----
    def _governing_regulations(self, policy_ids: list[str]) -> list[str]:
        regs = set()
        for pid in policy_ids:
            for node in nx.descendants(self.graph, ("policy", pid)):
                if node[0] == "regulation":
                    regs.add(node[1])
        return sorted(regs)

    def _chain(self, policy_id: str) -> list[str]:
        p = self.policies[policy_id]
        return [policy_id, p["domain"], p["regulation"]]  # precise declared path

    # ---- 3. resolver ----
    def resolve(self, fired_labels: list[str], text: str) -> dict:
        applicable = self._applicable(fired_labels, text)
        if not applicable:
            return {
                "decision": "PASS", "winning_policy": None, "priority": None,
                "policy_chain": [], "applied_policies": [],
                "governing_regulations": [], "conflicts": [],
            }

        applicable.sort(key=lambda p: p["priority"], reverse=True)
        winner = applicable[0]
        applied_ids = [p["id"] for p in applicable]

        conflicts = []
        distinct_actions = {p["action"] for p in applicable}
        if len(distinct_actions) > 1:
            conflicts.append({
                "resolved_to": winner["action"],
                "winning_policy": winner["id"],
                "competing": [
                    {"id": p["id"], "action": p["action"], "priority": p["priority"]}
                    for p in applicable
                ],
            })

        return {
            "decision": winner["action"].upper(),   # BLOCK | SANITIZE | PASS
            "winning_policy": winner["id"],
            "priority": winner["priority"],
            "policy_chain": self._chain(winner["id"]),
            "applied_policies": applied_ids,
            "governing_regulations": self._governing_regulations(applied_ids),
            "conflicts": conflicts,
        }

    # ---- inspection helper (nice for the demo) ----
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
        }
