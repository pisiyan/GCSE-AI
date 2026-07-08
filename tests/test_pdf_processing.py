import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from load_and_store import PdfFile

class TestPdfProcessing(unittest.TestCase):
    def setUp(self):
        self.pdf = object.__new__(PdfFile)

    def test_flatten_nested(self):
        self.assertEqual(self.pdf.flatten([1, [2, 3], 4]), [1, 2, 3, 4])

    def test_flatten_already_flat(self):
        self.assertEqual(self.pdf.flatten([1, 2, 3]), [1, 2, 3])

    def test_flatten_empty(self):
        self.assertEqual(self.pdf.flatten([]), [])

    def test_extract_mark_parentheses(self):
        self.assertEqual(self.pdf.extract_mark("(4)", r"\((\d+)\)"), 4)

    def test_extract_mark_brackets(self):
        self.assertEqual(self.pdf.extract_mark("[3 marks]", r"\[(\d+)\s*(?:marks?)?\]"), 3)

    def test_extract_mark_no_match(self):
        self.assertIsNone(self.pdf.extract_mark("no marks here", r"\((\d+)\)"))

    def test_extract_mark_multiple_matches(self):
        self.assertEqual(self.pdf.extract_mark("(2) some text (4)", r"\((\d+)\)"), 4)

    def test_is_parent_question_valid_match(self):
        parent_question = {
            "parent_question_description": "desc",
            "parent_question_structure": [1, [2, 3]]
        }
        questions = [
            {"marks": 1, "parent_question_description": "desc"},
            {"marks": 2, "parent_question_description": "desc"},
            {"marks": 3, "parent_question_description": "desc"}
        ]
        self.assertTrue(self.pdf.is_parent_question_valid(parent_question, questions))

    def test_is_parent_question_valid_mismatch(self):
        parent_question = {
            "parent_question_description": "desc",
            "parent_question_structure": [1, [2, 3]]
        }
        questions = [
            {"marks": 1, "parent_question_description": "desc"},
            {"marks": 2, "parent_question_description": "desc"}
        ]
        self.assertFalse(self.pdf.is_parent_question_valid(parent_question, questions))

if __name__ == '__main__':
    unittest.main()
