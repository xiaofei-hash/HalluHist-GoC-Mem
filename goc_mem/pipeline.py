"""Backend-neutral GoC-Mem inference pipeline."""
from dataclasses import asdict
from typing import Protocol
import json

from .core import build_graph, normalize_label, parse_claims, parse_json, prune, reconstruct
from .prompts import extraction_prompt, verification_prompt, answering_prompt


class Backend(Protocol):
    def generate_text(
        self,
        prompt: str,
        image_path: str | None = None,
        max_new_tokens: int = 128,
    ) -> str: ...


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
    errors = []
    raw_claims = backend.generate_text(
        extraction_prompt(format_history(sample["history"]), question),
        max_new_tokens=1024,
    )
    try:
        historical_claims = parse_claims(raw_claims)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        historical_claims = []
        errors.append({"stage": "extraction", "error": repr(exc)})
    nodes, edges = build_graph(historical_claims)

    expected = {c.id for c in nodes}
    raw_verifications = "[]"
    fallback_verifications = []
    rows = []
    if nodes:
        claim_payload = json.dumps([asdict(c) for c in nodes], ensure_ascii=False)
        raw_verifications = backend.generate_text(
            verification_prompt(claim_payload),
            image_path=image_path,
            max_new_tokens=128,
        )
        try:
            rows = _verification_rows(raw_verifications)
            if len(rows) != len(expected) or {row.get("id") for row in rows} != expected:
                raise ValueError("Verifier must return exactly one result for every claim and anchor")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"stage": "batch_verification", "error": repr(exc)})
            rows = []
            for claim in nodes:
                payload = json.dumps([asdict(claim)], ensure_ascii=False)
                raw_single = backend.generate_text(
                    verification_prompt(payload),
                    image_path=image_path,
                    max_new_tokens=128,
                )
                fallback_verifications.append({"id": claim.id, "raw": raw_single})
                try:
                    parsed = _verification_rows(raw_single)
                    if len(parsed) != 1 or parsed[0].get("id") != claim.id:
                        raise ValueError("Single-claim verifier returned the wrong result")
                    rows.append(parsed[0])
                except (ValueError, TypeError, json.JSONDecodeError) as single_exc:
                    errors.append({
                        "stage": "single_verification",
                        "claim_id": claim.id,
                        "error": repr(single_exc),
                    })
                    rows.append({
                        "id": claim.id,
                        "label": "UNCERTAIN",
                        "confidence": 0.0,
                        "evidence": "Verifier output could not be parsed.",
                    })
    labels = {}
    verification = []
    for row in rows:
        label = normalize_label(row.get("label"), row.get("confidence"), tau_c)
        labels[row["id"]] = label
        verification.append({**row, "normalized_label": label})

    seeds = {claim_id for claim_id, label in labels.items() if label == "CONTRADICTED"}
    removed = prune(edges, seeds, cascade=cascade)
    memory = reconstruct(nodes, labels, removed, question,
                         confidences={row['id']: row.get('confidence') for row in verification})
    answer_prompt = answering_prompt(question, memory)
    answer = backend.generate_text(answer_prompt, image_path=image_path, max_new_tokens=32)
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
        "raw_claim_extraction": raw_claims,
        "raw_batch_verification": raw_verifications,
        "raw_fallback_verifications": fallback_verifications,
        "errors": errors,
    }
