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

    def test_process_questions_hierarchical(self):
        from config import SubjectConfig
        config = SubjectConfig(
            mark_pattern=r"\((\d+)\)",
            sub_question_pattern=r"\(\s*[a-h]\s*\)",
            sub_sub_question_pattern=r"\((?:i{1,3}|iv|v|vi{1,3}|ix|x)\)",
            question_pattern=r"Question \d+"
        )
        self.pdf.config = config
        self.pdf.sub_question_pattern = config.sub_question_pattern
        self.pdf.sub_sub_question_pattern = config.sub_sub_question_pattern
        self.pdf.marks_pattern = config.mark_pattern

        raw_questions = [
            "Intro context (a) Sub-part A (2) (b) Sub-part B (i) Sub-sub-part 1 (1) (ii) Sub-sub-part 2 (3)",
            "Basic question without subparts (5)"
        ]
        results = self.pdf.process_questions(raw_questions, "Biology", "E1")
        self.assertEqual(len(results), 2)
        
        # Test basic question
        self.assertEqual(results[1]["type"], "basic_question")
        self.assertEqual(results[1]["marks"], 5)
        
        # Test parent question
        parent = results[0]
        self.assertEqual(parent["type"], "parent_question")
        self.assertEqual(parent["parent_description"], "Intro context")
        self.assertEqual(parent["parent_question_structure"], [2, [1, 3]])
        self.assertEqual(len(parent["sub_questions"]), 2)
        self.assertEqual(parent["sub_questions"][0]["label"], "a)")
        self.assertEqual(parent["sub_questions"][0]["marks"], 2)
        self.assertEqual(parent["sub_questions"][1]["label"], "b)")
        self.assertEqual(len(parent["sub_questions"][1]["sub_parts"]), 2)
        self.assertEqual(parent["sub_questions"][1]["sub_parts"][0]["label"], "i)")
        self.assertEqual(parent["sub_questions"][1]["sub_parts"][0]["marks"], 1)

if __name__ == '__main__':
    unittest.main()
