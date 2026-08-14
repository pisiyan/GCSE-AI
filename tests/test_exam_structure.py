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

    def test_question_position_importance(self):
        # Exam E1 has: pos0 = 2 marks (basic), pos1 = 3 marks (parent)
        # Exam E2 has: pos0 = 4 marks (basic), pos1 = 1 mark (basic)
        questions = [
            {"topic": "T1", "type": "basic_question", "exam": "E1", "marks": 2},
            {"topic": "T1", "type": "parent_question", "exam": "E1", "parent_question_structure": [1, 2]},
            {"topic": "T1", "type": "basic_question", "exam": "E2", "marks": 4},
            {"topic": "T1", "type": "basic_question", "exam": "E2", "marks": 1},
        ]

        # Enabled: position 0 must be 2 or 4; position 1 must be 3 or 1
        mock_config_enabled = MagicMock()
        mock_config_enabled.question_no_importance = True
        builder_enabled = ExamStructureBuilder(config=mock_config_enabled, questions=questions)
        pos_opts = builder_enabled.get_position_options("T1")
        self.assertIn(0, pos_opts)
        self.assertIn((2, "basic"), pos_opts[0])
        self.assertIn((4, "basic"), pos_opts[0])
        self.assertIn(1, pos_opts)
        self.assertIn((3, "parent"), pos_opts[1])
        self.assertIn((1, "basic"), pos_opts[1])

        # Test structure generation with position importance enforced
        struct_enabled = builder_enabled.build_structure(5, "T1")
        self.assertEqual(sum(item[0] for item in struct_enabled), 5)
        # First question must match position 0 valid options (2 or 4)
        self.assertIn(struct_enabled[0][0], [2, 4])

    def test_pattern_tiling_scaling(self):
        # Create a past exam structure with standard 5-question block: [1, 2, 4, 5, 12]
        questions = []
        pattern = [1, 2, 4, 5, 12]
        for idx, m in enumerate(pattern * 2):
            questions.append({
                "topic": "Islam",
                "type": "basic_question",
                "exam": "E1",
                "marks": m
            })
        
        mock_config = MagicMock()
        mock_config.question_no_importance = False
        builder = ExamStructureBuilder(config=mock_config, questions=questions)

        # Request 48 marks and 10 questions -> should tile [1, 2, 4, 5, 12] twice
        struct = builder.build_structure(48, "Islam", num_questions=10)
        self.assertEqual(len(struct), 10)
        self.assertEqual(sum(x[0] for x in struct), 48)
        expected_marks = [1, 2, 4, 5, 12, 1, 2, 4, 5, 12]
        self.assertEqual([x[0] for x in struct], expected_marks)

    def test_tier3_patternless_fallback(self):
        # Questions with no repeating blocks (e.g. 3 and 7 marks)
        questions = [
            {"topic": "Physics", "type": "basic_question", "exam": "E1", "marks": 3},
            {"topic": "Physics", "type": "basic_question", "exam": "E1", "marks": 7},
        ]
        mock_config = MagicMock()
        mock_config.question_no_importance = False
        builder = ExamStructureBuilder(config=mock_config, questions=questions)

        # Request 20 marks and 4 questions -> should fall back safely to 2x7 + 2x3 = 20
        struct = builder.build_structure(20, "Physics", num_questions=4)
        self.assertEqual(len(struct), 4)
        self.assertEqual(sum(x[0] for x in struct), 20)

if __name__ == '__main__':
    unittest.main()
