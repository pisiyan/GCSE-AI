"""Unit tests for canvas typesetting, booklet assembly, and whole exam marking."""

import base64
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
import pymupdf

from marking.canvas_typesetter import typeset_student_answer_page, typeset_student_answer_to_b64
from marking.booklet import assemble_marked_booklet_pdf, render_summary_cover_image
from marking.marker import ExamMarker


class TestCanvasTypesetting(unittest.TestCase):
    def test_typeset_student_answer_page(self):
        """Typesetting generates a clean, well-dimensioned PIL Image."""
        img = typeset_student_answer_page(
            question_label="Question 1(a)",
            question_prompt="Describe the differences between aerobic and anaerobic respiration.",
            student_answer_text="Aerobic respiration uses oxygen to release energy. Anaerobic does not use oxygen and produces lactic acid.",
            marks=4,
            body_width=760,
            min_height=650,
        )

        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size[0], 760)
        self.assertGreaterEqual(img.size[1], 650)

    def test_typeset_student_answer_to_b64(self):
        """Base64 conversion returns valid decodable PNG string."""
        b64_str = typeset_student_answer_to_b64(
            question_label="Q2",
            question_prompt="Define velocity.",
            student_answer_text="Speed in a given direction.",
            marks=1,
        )

        self.assertIsInstance(b64_str, str)
        raw_bytes = base64.b64decode(b64_str)
        self.assertTrue(raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_render_summary_cover_image(self):
        """Cover image renders with score, performance grade, and question table."""
        q_results = [
            {"label": "Q1(a)", "marks": 4, "awarded_marks": 4, "question_type": "point_based"},
            {"label": "Q1(b)", "marks": 2, "awarded_marks": 1, "question_type": "point_based"},
            {"label": "Q2", "marks": 6, "awarded_marks": 4, "question_type": "leveled"},
        ]
        summary_text = (
            "### 🌟 Content Strengths\n- Aerobic respiration mastered.\n\n"
            "### ⚠️ Content Weaknesses & Conceptual Gaps\n- Missed activation energy.\n\n"
            "### 🔍 Exam Technique & Command Word Issues\n- Explain command word requires mechanism.\n"
        )
        img = render_summary_cover_image(
            subject="Biology",
            examiner="Edexcel",
            topic="Bioenergetics",
            total_available=12,
            total_awarded=9,
            question_results=q_results,
            executive_summary_text=summary_text,
            width=1000,
            min_height=1200,
        )

        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size[0], 1000)
        self.assertGreaterEqual(img.size[1], 1200)

    def test_assemble_marked_booklet_pdf(self):
        """Booklet assembler stitches summary page and question pages into PDF."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create two dummy question images
            p1 = os.path.join(tmp_dir, "q1_marked.png")
            p2 = os.path.join(tmp_dir, "q2_marked.png")
            Image.new("RGB", (700, 500), (255, 255, 255)).save(p1)
            Image.new("RGB", (700, 500), (250, 250, 250)).save(p2)

            q_results = [
                {"label": "Q1", "marks": 4, "awarded_marks": 3},
                {"label": "Q2", "marks": 3, "awarded_marks": 2},
            ]
            out_pdf = os.path.join(tmp_dir, "marked_exam_booklet.pdf")

            assemble_marked_booklet_pdf(
                subject="Physics",
                examiner="AQA",
                topic="Forces",
                marked_image_paths=[p1, p2],
                question_results=q_results,
                executive_summary_text="Overall solid performance.",
                output_pdf_path=out_pdf,
            )

            self.assertTrue(os.path.exists(out_pdf))
            # Open with pymupdf and verify page count = 1 cover + 2 question pages = 3 pages
            doc = pymupdf.open(out_pdf)
            self.assertEqual(len(doc), 3)
            doc.close()

    def test_mark_exam_whole_flow_fallback_compatible(self):
        """mark_exam works with mock LLM for batch questions."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Good answer. [3/4]"
        mock_config = MagicMock()
        mock_config.mark_pattern = r"\((\d+)\)"

        marker = ExamMarker(
            config=mock_config,
            llm_client=mock_llm,
            similarity_engine=MagicMock(),
            questions=[],
            mark_schemes=[],
            subject="Chemistry",
            examiner="Edexcel",
        )

        questions = [
            {"label": "Q1", "question": "Explain ionic bonding", "student_answer": "Transfer of electrons", "marks": 4},
            {"label": "Q2", "question": "Define covalent bonding", "student_answer": "Sharing of electrons", "marks": 2},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("os.makedirs"):
                results = marker.mark_exam(questions, topic="Bonding", compile_booklet=False)
                self.assertEqual(len(results), 2)
                self.assertIsNotNone(results[0].get("awarded_marks"))


if __name__ == "__main__":
    unittest.main()
