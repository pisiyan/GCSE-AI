import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exam_segmenter import ExamSegmenter, extract_mark_val
from config import SubjectConfig


class TestExamSegmenter(unittest.TestCase):
    def setUp(self):
        self.mock_config = SubjectConfig(
            subject="Biology",
            examiner="Edexcel",
            mark_pattern=r"\((\d+)\)",
            sub_question_pattern=r"\(\s*[a-h]\s*\)",
            sub_sub_question_pattern=r"\((?:i{1,3}|iv|v|vi{1,3}|ix|x)\)",
            question_pattern=r"(?i)question\s*\d+",
            ms_pattern=r"(?i)question\s*number\s*\d+",
        )
        self.segmenter = ExamSegmenter(self.mock_config)

    def test_extract_mark_val(self):
        m1 = extract_mark_val("Explain enzyme denaturation. (3)", r"\((\d+)\)")
        self.assertEqual(m1, 3)
        m2 = extract_mark_val("Calculate mass change. [2 marks]", r"\((\d+)\)")
        self.assertEqual(m2, 2)

    def test_parse_exam_paper_single(self):
        text = "Question 1\nState two functions of cell membrane. (2)"
        items = self.segmenter.parse_exam_paper(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["label"], "Q1")
        self.assertEqual(items[0]["marks"], 2)

    def test_parse_exam_paper_subquestions(self):
        text = """Question 1
A student investigates osmosis.
(a) Define osmosis. (2)
(b) Explain why mass increased. (3)"""
        items = self.segmenter.parse_exam_paper(text)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["label"], "Q1(a)")
        self.assertEqual(items[0]["marks"], 2)
        self.assertEqual(items[1]["label"], "Q1(b)")
        self.assertEqual(items[1]["marks"], 3)

    def test_parse_mark_scheme(self):
        ms_text = """Question Number 1(a)
Answer: Net movement of water molecules (1) across a partially permeable membrane (1).

Question Number 1(b)
Answer: Water enters cell (1) by osmosis down concentration gradient (1) increasing turgor (1)."""
        ms_map = self.segmenter.parse_mark_scheme(ms_text)
        self.assertIn("1(a)", ms_map)
        self.assertIn("1(b)", ms_map)
        self.assertIn("Net movement", ms_map["1(a)"])

    def test_segment_student_answers(self):
        ans_text = """Question 1(a): Osmosis is the movement of water.
1(b): Water moved into the potato by osmosis causing mass to increase."""
        questions = [
            {"label": "Q1(a)"},
            {"label": "Q1(b)"}
        ]
        ans_map = self.segmenter.segment_student_answers(ans_text, questions)
        self.assertIn("Q1(a)", ans_map)
        self.assertIn("movement of water", ans_map["Q1(a)"])
        self.assertIn("Q1(b)", ans_map)

    def test_segment_full_paper(self):
        q_text = """Question 1
(a) Define diffusion. (1)
(b) Give one example. (1)"""
        ms_text = """Question 1(a) Movement of particles from high to low concentration.
Question 1(b) Oxygen into blood."""
        ans_text = """1a: Particles spread out.
1b: Oxygen in lungs."""

        items = self.segmenter.segment_full_paper(q_text, ans_text, ms_text)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["label"], "Q1(a)")
        self.assertIn("Particles spread out", items[0]["answer"])
        self.assertIn("Movement of particles", items[0]["mark_scheme"])


if __name__ == "__main__":
    unittest.main()
