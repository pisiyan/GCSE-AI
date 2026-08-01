import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generate_content import format_exam_as_markdown


class TestMarkdownFormatter(unittest.TestCase):

    def test_format_exam_as_markdown_basic_only(self):
        exam_data = {
            "structure": [[3], [2]],
            "questions": {
                "Topic A": [
                    {
                        "number": "1)",
                        "text": "Explain photosynthesis.",
                        "marks": 3,
                        "subtopic": "Photosynthesis Basics"
                    },
                    {
                        "number": "2)",
                        "text": "State the formula of glucose.",
                        "marks": 2,
                        "subtopic": "Chemical Formulas"
                    }
                ]
            }
        }
        
        md_output = format_exam_as_markdown("Biology", "Edexcel", "Higher", exam_data)
        
        self.assertIn("# GCSE Biology (Edexcel) - Higher Exam", md_output)
        self.assertIn("**Total Marks:** 5", md_output)
        self.assertIn("## Topic: Topic A", md_output)
        self.assertIn("### Question 1) (Subtopic: Photosynthesis Basics) *(3 marks)*", md_output)
        self.assertIn("Explain photosynthesis.", md_output)
        self.assertIn("### Question 2) (Subtopic: Chemical Formulas) *(2 marks)*", md_output)
        self.assertIn("State the formula of glucose.", md_output)

    def test_format_exam_as_markdown_parent_question(self):
        exam_data = {
            "structure": [[[1, 2]]],
            "questions": {
                "Topic B": [
                    {
                        "number": "1)",
                        "parent_description": "Figure 1 shows a cell diagram.",
                        "subtopic": "Cell Structure",
                        "sub_questions": [
                            {
                                "label": "a)",
                                "text": "Label structure X.",
                                "marks": 1
                            },
                            {
                                "label": "b)",
                                "context": "This is context for part b.",
                                "sub_parts": [
                                    {
                                        "label": "i)",
                                        "text": "Explain the function of structure X.",
                                        "marks": 2
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        
        md_output = format_exam_as_markdown("Biology", "Edexcel", "Higher", exam_data)
        
        self.assertIn("# GCSE Biology (Edexcel) - Higher Exam", md_output)
        self.assertIn("**Total Marks:** 3", md_output)
        self.assertIn("## Topic: Topic B", md_output)
        self.assertIn("### Question 1) (Subtopic: Cell Structure)", md_output)
        self.assertIn("Figure 1 shows a cell diagram.", md_output)
        self.assertIn("* **a)** Label structure X. *(1 mark)*", md_output)
        self.assertIn("**b)** This is context for part b.", md_output)
        self.assertIn("  * **i)** Explain the function of structure X. *(2 marks)*", md_output)

    def test_format_exam_as_markdown_spec_trees(self):
        exam_data = {
            "structure": [[3]],
            "spec_trees": {
                "Topic A": {
                    "spec_code": "Topic 1 (1.1-1.17)",
                    "subtopics": ["Photosynthesis"]
                }
            },
            "questions": {
                "Topic A": [
                    {
                        "number": "1)",
                        "text": "Explain photosynthesis.",
                        "marks": 3,
                        "subtopic": "Photosynthesis"
                    }
                ]
            }
        }

        md_output = format_exam_as_markdown("Biology", "Edexcel", "Higher", exam_data)

        self.assertIn("## Topic & Specification Breakdown", md_output)
        self.assertIn("### Topic: Topic A (Specification Code: Topic 1 (1.1-1.17))", md_output)
        self.assertIn("- **Subtopic:** Photosynthesis", md_output)


if __name__ == '__main__':
    unittest.main()
