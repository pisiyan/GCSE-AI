import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_subject_config, SubjectConfig


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

    def test_empty_subject_config(self):
        config = load_subject_config("Physics", "Edexcel")
        self.assertIsInstance(config, SubjectConfig)
        self.assertEqual(config.subject, "Physics")
        self.assertEqual(config.examiner, "Edexcel")
        # Verify it inherited Edexcel defaults (e.g., specific letter_pattern, marks, chunk size)
        self.assertEqual(config.letter_pattern, r"\(\s*[a-h]\s*\)")
        self.assertEqual(config.spec_chunk_size, 2000)

if __name__ == '__main__':
    unittest.main()
