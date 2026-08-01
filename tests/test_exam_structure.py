import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exam_generator import ExamStructureBuilder, _filter_by_exam_type

class TestExamStructure(unittest.TestCase):
    def test_filter_by_exam_type(self):
        questions = [
            {"text": "Q1", "exam_type": "Higher"},
            {"text": "Q2", "exam_type": "Foundation"},
            {"text": "Q3", "exam_type": "Higher"},
            {"text": "Q4", "exam_type": ""},
        ]
        higher_qs = _filter_by_exam_type(questions, "Higher")
        self.assertEqual(len(higher_qs), 2)
        self.assertEqual([q["text"] for q in higher_qs], ["Q1", "Q3"])

        foundation_qs = _filter_by_exam_type(questions, "Foundation")
        self.assertEqual(len(foundation_qs), 1)
        self.assertEqual(foundation_qs[0]["text"], "Q2")

        # Unknown exam type falls back to unspecified ones
        unknown_qs = _filter_by_exam_type(questions, "UnknownTier")
        self.assertEqual(len(unknown_qs), 1)
        self.assertEqual(unknown_qs[0]["text"], "Q4")
    def test_flatten_marks_int(self):
        self.assertEqual(ExamStructureBuilder.flatten_marks(5), 5)

    def test_flatten_marks_flat_list(self):
        self.assertEqual(ExamStructureBuilder.flatten_marks([2, 3, 1]), 6)

    def test_flatten_marks_nested(self):
        self.assertEqual(ExamStructureBuilder.flatten_marks([2, [3, 1], 4]), 10)

    def test_distribute_even(self):
        self.assertEqual(
            ExamStructureBuilder.distribute_to_topics([1, 2, 3, 4], ["A", "B"]),
            {"A": [1, 2], "B": [3, 4]}
        )

    def test_distribute_uneven(self):
        self.assertEqual(
            ExamStructureBuilder.distribute_to_topics([1, 2, 3, 4, 5], ["A", "B"]),
            {"A": [1, 2, 3], "B": [4, 5]}
        )

    def test_distribute_single_topic(self):
        self.assertEqual(
            ExamStructureBuilder.distribute_to_topics([1, 2, 3], ["A"]),
            {"A": [1, 2, 3]}
        )

    def test_distribute_empty_topics_raises(self):
        with self.assertRaises(ValueError):
            ExamStructureBuilder.distribute_to_topics([1, 2, 3], [])

    def test_get_past_exam_structures(self):
        questions = [
            {"topic": "T1", "type": "basic_question", "exam": "E1", "marks": 2},
            {"topic": "T1", "type": "parent_question", "exam": "E1", "parent_question_structure": [1, 2]},
            {"topic": "T1", "type": "basic_question", "exam": "E2", "marks": 4},
            {"topic": "T2", "type": "basic_question", "exam": "E1", "marks": 5},
        ]
        builder = ExamStructureBuilder(config=MagicMock(), questions=questions)
        structures = builder.get_past_exam_structures("T1")
        self.assertEqual(len(structures), 2)
        self.assertEqual(structures[0], [[2, "basic"], [3, "parent"]])
        self.assertEqual(structures[1], [[4, "basic"]])

    def test_build_structure_terminates(self):
        questions = [
            {"topic": "T1", "type": "basic_question", "exam": "E1", "marks": 2},
            {"topic": "T1", "type": "parent_question", "exam": "E1", "parent_question_structure": [1, 2]},
        ]
        mock_config = MagicMock()
        mock_config.question_no_importance = False
        builder = ExamStructureBuilder(config=mock_config, questions=questions)
        
        structure = builder.build_structure(5, "T1")
        
        # Calculate total marks in the returned structure
        total = 0
        for item in structure:
            total += ExamStructureBuilder.flatten_marks(item)
            
        self.assertEqual(total, 5)

if __name__ == '__main__':
    unittest.main()
