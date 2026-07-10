import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from similarity import SimilarityEngine
from exam_generator import LocalSimilarityEngine, is_calculation_content, is_practical_content

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

class TestLocalSimilarity(unittest.TestCase):
    def test_pick_diverse_subset_empty(self):
        engine = LocalSimilarityEngine(embedding_model=None, llm_client=None)
        self.assertEqual(engine.pick_diverse_subset([], 3), [])

    def test_pick_diverse_subset_less_than_k(self):
        engine = LocalSimilarityEngine(embedding_model=None, llm_client=None)
        candidates = ["a", "b"]
        self.assertEqual(engine.pick_diverse_subset(candidates, 3), candidates)

    def test_pick_diverse_subset_diversity(self):
        mock_embedding_model = MagicMock()
        
        def mock_embed(texts):
            embs = []
            for t in texts:
                if t == "a":
                    embs.append([1.0, 0.0])
                elif t == "b":
                    embs.append([0.9, 0.1])
                elif t == "c":
                    embs.append([0.0, 1.0])
                else:
                    embs.append([0.5, 0.5])
            return embs
            
        mock_embedding_model.embed_documents.side_effect = mock_embed
        engine = LocalSimilarityEngine(embedding_model=mock_embedding_model, llm_client=None)
        
        candidates = ["a", "b", "c"]
        result = engine.pick_diverse_subset(candidates, 2)
        self.assertEqual(result, ["a", "c"])

class TestCognitiveFiltering(unittest.TestCase):
    def test_is_calculation_content(self):
        self.assertTrue(is_calculation_content("Calculate the velocity"))
        self.assertTrue(is_calculation_content("Using the equation, work out the mass"))
        self.assertFalse(is_calculation_content("Describe the structure of a cell"))

    def test_is_practical_content(self):
        self.assertTrue(is_practical_content("A student investigated the experiment using a graph"))
        self.assertTrue(is_practical_content("Complete Table 1 showing the results"))
        self.assertFalse(is_practical_content("Define gravity"))

    def test_find_least_similar_objects_cognitive_filtering(self):
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_documents.return_value = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        engine = LocalSimilarityEngine(embedding_model=mock_embedding_model, llm_client=None)

        objects = [
            {"topic": "T1", "marks": 2, "question_content": "Calculate the speed"},
            {"topic": "T1", "marks": 2, "question_content": "Describe the cell biology"},
        ]
        
        # When comparing to a calculation concept, it should filter out the description question
        # and only return the calculation question.
        result = engine.find_least_similar_objects(
            objects=objects,
            comparison="Calculate mass",
            n=1,
            topic_value="T1",
            marks_value=2
        )
        self.assertEqual(result, ["Calculate the speed"])

if __name__ == '__main__':
    unittest.main()
