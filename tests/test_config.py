import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_subject_config, SubjectConfig
from exam_generator import QuestionGenerator
from unittest.mock import MagicMock


class TestConfig(unittest.TestCase):
    def test_load_valid_config(self):
        config = load_subject_config("ReligiousStudies", "AQA")
        self.assertIsInstance(config, SubjectConfig)
        self.assertEqual(config.mark_pattern, "[\\[\\(]\\s*(\\d{1,2})\\s*(marks?)?\\s*[\\]\\)]?")

    def test_load_missing_subject(self):
        with self.assertRaises(ValueError):
            load_subject_config("FakeSubject", "Fake")

    def test_config_types(self):
        config = load_subject_config("Biology", "Edexcel")
        self.assertIsInstance(config.spec_chunk_size, int)
        self.assertIsInstance(config.question_no_importance, bool)
        self.assertIsInstance(config.mark_pattern, str)

    def test_optional_example_descriptions(self):
        config = load_subject_config("ReligiousStudies", "AQA")
        self.assertEqual(config.example_descriptions, 0)

    def test_biology_has_example_descriptions(self):
        config = load_subject_config("Biology", "Edexcel")
        self.assertEqual(config.example_descriptions, 5)

    def test_new_configurable_fields(self):
        config = load_subject_config("Biology", "Edexcel")
        self.assertEqual(config.max_structure_retries, 50)
        self.assertEqual(config.max_parallel_workers, 10)
        self.assertEqual(config.max_parent_exemplars, 3)
        self.assertEqual(config.fallback_embedding_dim, 384)
        self.assertEqual(config.temperature_structure, 0.0)
        self.assertEqual(config.temperature_generation, 0.7)

    def test_empty_subject_config(self):
        config = load_subject_config("Physics", "Edexcel")
        self.assertIsInstance(config, SubjectConfig)
        self.assertEqual(config.subject, "Physics")
        self.assertEqual(config.examiner, "Edexcel")
        # Verify it inherited Edexcel defaults (e.g., specific letter_pattern, marks, chunk size)
        self.assertEqual(config.letter_pattern, r"\(\s*[a-h]\s*\)")
        self.assertEqual(config.spec_chunk_size, 2000)

    def test_get_mark_calibration_examples(self):
        config = load_subject_config("Biology", "Edexcel")
        # Empty questions list should fallback to default guidelines
        q_gen = QuestionGenerator(
            config=config,
            llm_client=MagicMock(),
            similarity_engine=MagicMock(),
            questions=[],
            prompts={},
            queries={},
            spec_qa_chain=MagicMock()
        )
        guidelines = q_gen._get_mark_calibration_examples()
        self.assertIn("GCSE Mark Calibration Guidelines", guidelines)
        self.assertIn("- 1 Mark Guideline", guidelines)
        self.assertIn("- 12 Marks Guideline", guidelines)

        # Database with real questions should use their content
        db_questions = [
            {"marks": 1, "text": "Define force."},
            {"marks": 12, "text": "Evaluate the ethics of abortion."}
        ]
        q_gen_with_db = QuestionGenerator(
            config=config,
            llm_client=MagicMock(),
            similarity_engine=MagicMock(),
            questions=db_questions,
            prompts={},
            queries={},
            spec_qa_chain=MagicMock()
        )
        guidelines_db = q_gen_with_db._get_mark_calibration_examples()
        self.assertIn("Define force.", guidelines_db)
        self.assertIn("Evaluate the ethics of abortion.", guidelines_db)

if __name__ == '__main__':
    unittest.main()
