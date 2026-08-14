import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exam_marker import ExamMarker

class TestMarking(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_sim = MagicMock()
        self.questions = [
            {"topic": "T1", "question": "q1"},
            {"topic": "T2", "question": "q2"},
            {"topic": "T1", "question": "q3"},
        ]
        
        self.prompts = {
            "extract_marks": "{question}",
            "get_command_word": "{question}",
            "format_mark_scheme": "{mark_scheme}",
            "mark_answer": "{answer}",
            "create_ms_structure": "{subject}",
            "create_new_markscheme": "{question}",
            "exam_type_of_question": "{question}",
            "model_answer": "{question}",
            "read_user_exam_page": "read page",
        }
        
        self.marker = ExamMarker(
            config=self.mock_config,
            llm_client=self.mock_llm,
            similarity_engine=self.mock_sim,
            questions=self.questions,
            mark_schemes=[],
            prompts=self.prompts,
            queries={},
            spec_qa_chain=MagicMock(),
            ms_qa_chain=MagicMock(),
            subject="TestSubject",
            examiner="TestExaminer"
        )

    def test_exam_types_cached(self):
        types1 = self.marker.exam_types
        types2 = self.marker.exam_types
        self.assertIs(types1, types2)

    def test_exam_types_unique(self):
        types = self.marker.exam_types
        self.assertEqual(len(types), 2)
        self.assertIn("T1", types)
        self.assertIn("T2", types)

    def test_get_marks_valid(self):
        self.mock_llm.invoke.return_value = "4"
        marks = self.marker.get_marks_from_question("q")
        self.assertEqual(marks, 4)

    def test_get_marks_invalid(self):
        self.mock_llm.invoke.return_value = "not a number"
        with self.assertRaises(ValueError):
            self.marker.get_marks_from_question("q")

    def test_parallel_mark_exam(self):
        self.mock_llm.invoke.side_effect = lambda prompt: (
            "Command" if "command word" in str(prompt).lower()
            else "Formatted MS" if "formatting assistant" in str(prompt).lower()
            else "Good answer. [3/4]"
        )
        questions_to_mark = [
            {"question": "Q1 text", "marks": 4, "answer": "A1", "mark_scheme": "MS1"},
            {"question": "Q2 text", "marks": 5, "answer": "A2", "mark_scheme": "MS2"},
            {"question": "Q3 text", "marks": 2, "answer": "A3", "mark_scheme": "MS3"},
        ]
        results = self.marker.mark_exam(questions_to_mark, "T1")
        self.assertEqual(len(results), 3)
        self.assertIn("Q1 text", results[0]["question"])
        self.assertIn("Q2 text", results[1]["question"])
        self.assertIn("Q3 text", results[2]["question"])
        self.assertEqual(results[0]["awarded_marks"], 3)

    def test_direct_mark_answer(self):
        self.mock_llm.invoke.return_value = "Feedback [2/2]"
        res = self.marker.mark_answer(
            answer="Student answer text",
            mark_scheme="MS",
            question="Question 1 (2)",
            marks=2
        )
        self.assertEqual(res, "Feedback [2/2]")

    def test_format_marking_as_markdown(self):
        from generate_content import format_marking_as_markdown
        results = [
            {
                "question": "Explain enzyme action",
                "student_answer": "Enzymes bind to substrate",
                "mark_scheme": "Key-lock model",
                "result": "Good detail [2/3]",
                "marks": 3,
                "awarded_marks": 2
            }
        ]
        md = format_marking_as_markdown("Biology", "Edexcel", "Higher", results)
        self.assertIn("GCSE Biology (Edexcel) - Higher Marking Report", md)
        self.assertIn("Executive Score Summary", md)
        self.assertIn("Explain enzyme action", md)

    def test_pdf_support_in_marking(self):
        from unittest.mock import patch
        with patch("pypdf.PdfReader") as mock_pdf_reader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "PDF extracted question content"
            mock_pdf_reader.return_value.pages = [mock_page]
            text = self.marker.pdf_to_text("dummy.pdf")
            self.assertEqual(text, "PDF extracted question content")


if __name__ == '__main__':
    unittest.main()
