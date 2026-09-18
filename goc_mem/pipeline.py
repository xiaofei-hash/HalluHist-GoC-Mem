"""Backend-neutral GoC-Mem inference pipeline."""
from dataclasses import asdict
from typing import Protocol
import json

from .core import build_graph, normalize_label, parse_claims, parse_json, prune, reconstruct
from .prompts import extraction_prompt, verification_prompt, answering_prompt


class Backend(Protocol):
    def generate_text(self, prompt: str, image_path: str | None = None) -> str: ...


def format_history(history):
    rows = []
    for turn in history:
        rows.append("Turn {turn}\nUser: {question}\nAssistant: {assistant}".format(**turn))
    return "\n\n".join(rows)


def _verification_rows(raw):
    parsed = parse_json(raw)
    if isinstance(parsed, dict):
        parsed = parsed.get("verifications")
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON verification list")
    return parsed


def run_sample(sample, backend, tau_c=0.6, cascade=True):
    """Run extraction, all-claim verification, pruning, reconstruction, answering."""
    question = sample["final_question"]
    image_path = sample["image_path"]
    raw_claims = backend.generate_text(extraction_prompt(format_history(sample["history"]), question))
    historical_claims = parse_claims(raw_claims)
    nodes, edges = build_graph(historical_claims)

    claim_payload = json.dumps([asdict(c) for c in nodes], ensure_ascii=False)
    raw_verifications = backend.generate_text(verification_prompt(claim_payload), image_path=image_path)
    rows = _verification_rows(raw_verifications)
    expected = {c.id for c in nodes}
    if len(rows) != len(expected) or {row.get("id") for row in rows} != expected:
        raise ValueError("Verifier must return exactly one result for every claim and anchor")
    labels = {}
    verification = []
    for row in rows:
        label = normalize_label(row.get("label"), row.get("confidence"), tau_c)
        labels[row["id"]] = label
        verification.append({**row, "normalized_label": label})

    seeds = {claim_id for claim_id, label in labels.items() if label == "CONTRADICTED"}
    removed = prune(edges, seeds, cascade=cascade)
    memory = reconstruct(nodes, labels, removed, question)
    answer_prompt = answering_prompt(question, memory)
    answer = backend.generate_text(answer_prompt, image_path=image_path)
    return {
        "sample_id": sample["sample_id"],
        "answer": answer.strip(),
        "claims": [asdict(c) for c in nodes],
        "edges": [list(edge) for edge in edges],
        "verification": verification,
        "pruning_seeds": sorted(seeds),
        "pruned_nodes": sorted(removed),
        "memory": memory,
        "answer_prompt": answer_prompt,
    }
