import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from similarity import SimilarityEngine

class TestSimilarity(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.engine = SimilarityEngine(self.mock_llm)

    def test_compute_similarity(self):
        self.mock_llm.get_embeddings.return_value = [[1, 0, 0], [0, 1, 0]]
        score = self.engine.compute_similarity("a", "b")
        self.assertEqual(score, 0.0)
        
        self.mock_llm.get_embeddings.return_value = [[1, 0, 0], [1, 0, 0]]
        score = self.engine.compute_similarity("a", "a")
        self.assertAlmostEqual(score, 1.0)

    def test_compute_similarity_scores_empty_candidates(self):
        self.assertEqual(self.engine.compute_similarity_scores([], ["ref"]), [])

    def test_compute_similarity_scores_empty_references(self):
        scores = self.engine.compute_similarity_scores(["cand"], [])
        self.assertEqual(scores, [("cand", 0.0)])

    def test_pick_least_similar_single(self):
        self.assertEqual(self.engine.pick_least_similar(["only"], ["used"]), "only")

    def test_pick_least_similar_empty_raises(self):
        with self.assertRaises(ValueError):
            self.engine.pick_least_similar([], ["used"])

    def test_pick_least_similar_no_used(self):
        candidates = ["a", "b", "c"]
        choice = self.engine.pick_least_similar(candidates, [])
        self.assertIn(choice, candidates)

    def test_find_most_similar_single(self):
        self.assertEqual(self.engine.find_most_similar("query", ["only"]), "only")

    def test_find_most_similar(self):
        self.mock_llm.get_embeddings.return_value = [[1, 0], [1, 0], [0, 1]]
        result = self.engine.find_most_similar("query", ["match", "unmatch"])
        self.assertEqual(result, "match")

    def test_find_least_similar_objects_no_match(self):
        objects = [{"topic": "A", "marks": 1, "question_content": "q"}]
        result = self.engine.find_least_similar_objects(objects, "comp", 1, topic_value="B")
        self.assertEqual(result, [])

    def test_find_least_similar_objects(self):
        objects = [
            {"topic": "A", "marks": 1, "question_content": "similar"},
            {"topic": "A", "marks": 1, "question_content": "different"},
        ]
        def mock_embeddings(texts, model=None):
            embs = []
            for t in texts:
                if t in ("comp", "similar"):
                    embs.append([1.0, 0.0])
                else:
                    embs.append([0.0, 1.0])
            return embs
        
        self.mock_llm.get_embeddings.side_effect = mock_embeddings
        result = self.engine.find_least_similar_objects(objects, "comp", 1, topic_value="A")
        self.assertEqual(result, ["different"])

if __name__ == '__main__':
    unittest.main()
