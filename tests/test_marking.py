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
        
        self.marker = ExamMarker(
            config=self.mock_config,
            llm_client=self.mock_llm,
            similarity_engine=self.mock_sim,
            questions=self.questions,
            mark_schemes=[],
            prompts={"extract_marks": "{question}"},
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

if __name__ == '__main__':
    unittest.main()
