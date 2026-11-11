import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_subject_config, SubjectConfig
from exam_generation import QuestionGenerator
from unittest.mock import MagicMock


class TestConfig(unittest.TestCase):
    def test_load_valid_config(self):
        config = load_subject_config("ReligiousStudies", "AQA")
        self.assertIsInstance(config, SubjectConfig)
        self.assertEqual(config.mark_pattern, "[\\[\\(]\\s*(\\d{1,2})\\s*(marks?)?\\s*[\\]\\)]?")

    def test_load_fallback_subject(self):
        config = load_subject_config("FakeSubject", "Fake")
        self.assertIsInstance(config, SubjectConfig)
        self.assertEqual(config.subject, "FakeSubject")
        self.assertEqual(config.examiner, "Fake")

    def test_config_types(self):
        config = load_subject_config("Biology", "Edexcel")
        self.assertIsInstance(config.mark_pattern, str)
        self.assertIsInstance(config.max_structure_retries, int)
        self.assertIsInstance(config.similarity_threshold, float)

    def test_configurable_fields(self):
        config = load_subject_config("Biology", "Edexcel")
        self.assertEqual(config.max_structure_retries, 50)
        self.assertEqual(config.max_parallel_workers, 10)
        self.assertEqual(config.max_parent_exemplars, 3)
        self.assertEqual(config.fallback_embedding_dim, 384)
        self.assertEqual(config.temperature_structure, 0.0)
        self.assertEqual(config.temperature_generation, 0.7)

    def test_subject_config_fields(self):
        config = load_subject_config("Physics", "Edexcel")
        self.assertIsInstance(config, SubjectConfig)
        self.assertEqual(config.subject, "Physics")
        self.assertEqual(config.examiner, "Edexcel")
        self.assertEqual(config.sub_question_pattern, r"\(\s*[a-h]\s*\)")


if __name__ == '__main__':
    unittest.main()
