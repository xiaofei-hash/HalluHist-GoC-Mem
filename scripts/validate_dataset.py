import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    return [json.loads(line) for line in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines() if line]


inputs = rows("evaluation_inputs.jsonl")
labels = rows("evaluation_labels.jsonl")
assert len(inputs) == len(labels) == 1000
assert len({row["sample_id"] for row in inputs}) == 1000
assert {row["sample_id"] for row in inputs} == {row["sample_id"] for row in labels}
grouped = defaultdict(list)
for row in labels:
    grouped[row["pair_id"]].append(row)
assert len(grouped) == 500
assert all({item["subset"] for item in pair} == {"contaminated", "grounded"} for pair in grouped.values())
assert Counter(row["target_design"]["dialogue_length"] for row in labels) == {3: 200, 4: 200, 5: 200, 6: 200, 7: 200}
assert Counter(row["target_design"]["final_question_type"] for row in labels) == {
    "object_existence": 200, "attribute": 200, "counting": 200, "relation": 200, "action": 200}
assert Counter(row["target_design"]["history_configuration"] for row in labels) == {
    "erroneous_premise": 200, "mixed_true_false": 200, "correction_conflict": 200,
    "uncertain_evidence": 200, "topic_shift": 200}
for name in ("evaluation_inputs.jsonl", "evaluation_labels.jsonl"):
    data = (ROOT / "data" / name).read_bytes()
    print(name, len(data), hashlib.sha256(data).hexdigest())
print("validated: 500 pairs / 1,000 inputs")
