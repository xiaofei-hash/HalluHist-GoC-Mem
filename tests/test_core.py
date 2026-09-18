import unittest
import json

from goc_mem.core import Claim, build_graph, normalize_label, prune, reconstruct, score_pairs
from goc_mem.pipeline import run_sample


class CoreTests(unittest.TestCase):
    def test_label_normalization(self):
        self.assertEqual(normalize_label("SUPPORTED", 0.59, 0.6), "UNCERTAIN")
        self.assertEqual(normalize_label("CONTRADICTED", 0.6, 0.6), "CONTRADICTED")
        self.assertEqual(normalize_label("UNCERTAIN", 0.99, 0.6), "UNCERTAIN")

    def test_existence_cascade(self):
        claims = [Claim("c1", 1, "A red stroller is present.", "stroller", "exists", "yes", "existence"),
                  Claim("c2", 2, "The stroller carries a child.", "stroller", "carries", "child", "action")]
        nodes, edges = build_graph(claims)
        self.assertIn(("c1", "c2"), edges)
        self.assertEqual(prune(edges, {"c1"}), {"c1", "c2"})
        self.assertEqual(prune(edges, {"c1"}, cascade=False), {"c1"})

    def test_memory_states(self):
        nodes = [Claim("a", 1, "A dog is visible.", "dog", "exists", "yes", "existence"),
                 Claim("b", 2, "The dog is brown.", "dog", "color", "brown", "attribute")]
        memory = reconstruct(nodes, {"a": "CONTRADICTED", "b": "UNCERTAIN"}, {"a"}, "Is a dog visible?")
        self.assertEqual(memory["corrections"], ["A dog is visible."])
        self.assertEqual(memory["uncertain"], ["The dog is brown."])

    def test_pair_score(self):
        labels = [{"sample_id": "c", "pair_id": "p", "subset": "contaminated", "reference_answer": "No."},
                  {"sample_id": "g", "pair_id": "p", "subset": "grounded", "reference_answer": "Yes."}]
        predictions = [{"sample_id": "c", "answer": "No"}, {"sample_id": "g", "answer": "Yes"}]
        self.assertEqual(score_pairs(labels, predictions)["PairAcc"], 100)

    def test_pipeline_verifies_every_node(self):
        class FakeBackend:
            def __init__(self):
                self.calls = []

            def generate_text(self, prompt, image_path=None):
                self.calls.append((prompt, image_path))
                if len(self.calls) == 1:
                    return json.dumps([{"id": "c1", "turn": 1,
                        "text": "A red stroller is visible.", "subject": "stroller",
                        "relation": "has_color", "object": "red", "type": "attribute"}])
                if len(self.calls) == 2:
                    start = prompt.index('[')
                    end = prompt.index('\n\nTask:', start)
                    claims = json.loads(prompt[start:end])
                    return json.dumps([{"id": c["id"], "label": "CONTRADICTED",
                                        "confidence": 0.9, "evidence": "not visible"}
                                       for c in claims])
                return "No"

        backend = FakeBackend()
        sample = {"sample_id": "s", "image_path": "images/1.jpg",
                  "history": [{"turn": 1, "question": "What is visible?",
                               "assistant": "A red stroller is visible."}],
                  "final_question": "Is there a stroller?"}
        result = run_sample(sample, backend)
        self.assertEqual(len(result["verification"]), 2)  # claim + existence anchor
        self.assertEqual(result["answer"], "No")


if __name__ == "__main__":
    unittest.main()
