import argparse
import json

from goc_mem.core import score_pairs


def rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/evaluation_labels.jsonl")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    print(json.dumps(score_pairs(rows(args.labels), rows(args.predictions)), indent=2))
