import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exam_quality_analyzer import ExamQualityAnalyzer


class TestQualityAnalyzer(unittest.TestCase):

    def setUp(self):
        self.mock_assistant = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_sim = MagicMock()

        self.mock_assistant.llm_client = self.mock_llm
        self.mock_assistant.similarity = self.mock_sim
        self.mock_sim.llm_client = self.mock_llm
        self.mock_assistant.prompts = {
            "get_command_word": "Mock command word prompt for {question}",
            "evaluate_question_quality": "Mock quality prompt {question_text} {marks} {subtopic} {parent_context}"
        }
        self.mock_assistant.queries = {}
        self.mock_assistant.spec_qa_chain = MagicMock()

        # Mock past paper questions database
        self.mock_assistant.questions = [
            {"topic": "T1", "marks": 1, "question_content": "State one function of xylem."},
            {"topic": "T1", "marks": 2, "question_content": "Describe how xylem is adapted to its function."},
            {"topic": "T2", "marks": 4, "question_content": "Explain why plants lose water from their leaves."},
            {"topic": "T2", "marks": 6, "question_content": "Evaluate the effect of wind speed on transpiration rate."}
        ]

        self.analyzer = ExamQualityAnalyzer(self.mock_assistant)

    def test_extract_questions_flat_standalone(self):
        generated_exam = {
            "questions": {
                "Topic 1": [
                    {
                        "number": "1)",
                        "text": "What is the powerhouse of the cell?",
                        "marks": 1,
                        "subtopic": "Cell Structure"
                    }
                ]
            }
        }
        flat = self.analyzer.extract_questions_flat(generated_exam)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["id"], "1")
        self.assertEqual(flat[0]["text"], "What is the powerhouse of the cell?")
        self.assertEqual(flat[0]["marks"], 1)
        self.assertEqual(flat[0]["subtopic"], "Cell Structure")
        self.assertEqual(flat[0]["topic"], "Topic 1")
        self.assertEqual(flat[0]["parent_context"], "")

    def test_extract_questions_flat_nested(self):
        generated_exam = {
            "questions": {
                "Topic 1": [
                    {
                        "number": "2)",
                        "parent_description": "Figure 1 shows a cell.",
                        "subtopic": "Cell division",
                        "sub_questions": [
                            {
                                "label": "a)",
                                "text": "Name part A.",
                                "marks": 1
                            },
                            {
                                "label": "b)",
                                "context": "Mitosis occurs in somatic cells.",
                                "sub_parts": [
                                    {
                                        "label": "i)",
                                        "text": "State the number of daughter cells.",
                                        "marks": 2
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        flat = self.analyzer.extract_questions_flat(generated_exam)
        self.assertEqual(len(flat), 2)
        
        # Check standard child question
        self.assertEqual(flat[0]["id"], "2_a")
        self.assertEqual(flat[0]["text"], "Name part A.")
        self.assertEqual(flat[0]["marks"], 1)
        self.assertEqual(flat[0]["parent_context"], "Figure 1 shows a cell.")
        
        # Check grandchild question
        self.assertEqual(flat[1]["id"], "2_b_i")
        self.assertEqual(flat[1]["text"], "State the number of daughter cells.")
        self.assertEqual(flat[1]["marks"], 2)
        self.assertEqual(flat[1]["parent_context"], "Figure 1 shows a cell.\nMitosis occurs in somatic cells.")

    def test_analyze_mark_distribution_identical(self):
        # Setting up generated exam questions to have the exact same mark distribution proportions as setup
        flat_questions = [
            {"marks": 1},
            {"marks": 2},
            {"marks": 4},
            {"marks": 6}
        ]
        score = self.analyzer.analyze_mark_distribution(flat_questions)
        # Cosine similarity should be 100.0 (identical proportions)
        self.assertEqual(score, 100.0)

    def test_analyze_mark_distribution_skewed(self):
        # A completely different distribution (e.g. all 1-mark questions)
        flat_questions = [
            {"marks": 1},
            {"marks": 1},
            {"marks": 1}
        ]
        score = self.analyzer.analyze_mark_distribution(flat_questions)
        # Cosine similarity should be much lower (not 100.0)
        self.assertTrue(score < 100.0)
        self.assertTrue(score > 0.0)

    def test_analyze_command_words(self):
        flat_questions = [
            {"id": "1", "text": "State the name of a plant cell.", "marks": 1}
        ]
        
        # Mock the LLM to return "State"
        self.mock_llm.invoke.return_value = "State"
        
        score, details = self.analyzer.analyze_command_words(flat_questions)
        
        # Should be a perfect match because in past papers, "State" is associated with 1 mark
        self.assertEqual(score, 100.0)
        self.assertEqual(details[0]["command_word"], "State")
        self.assertEqual(details[0]["score"], 100.0)
        self.assertIn("Typical mark allocation", details[0]["reason"])

    def test_analyze_specification_relevance(self):
        flat_questions = [
            {"id": "1", "text": "Describe xylem.", "marks": 2, "parent_context": "", "subtopic": "Xylem", "topic": "T1"}
        ]
        
        # Mock retriever returning document chunks
        mock_doc = MagicMock()
        mock_doc.page_content = "The specification states that xylem transports water and mineral ions."
        self.mock_assistant.spec_retriever.invoke.return_value = [mock_doc]
        
        # Mock LLM response evaluating relevance
        self.mock_llm.invoke_json.return_value = {
            "relevance_score": 95,
            "reasoning": "The question tests xylem description which is explicitly mentioned in the spec."
        }
        
        score, details = self.analyzer.analyze_specification_relevance(flat_questions)
        self.assertEqual(score, 95.0)
        self.assertEqual(details[0]["score"], 95.0)
        self.assertEqual(details[0]["reasoning"], "The question tests xylem description which is explicitly mentioned in the spec.")

    def test_evaluate_qualitative_aspects(self):
        flat_questions = [
            {"id": "1", "text": "Describe xylem.", "marks": 2, "subtopic": "Xylem", "parent_context": ""}
        ]
        
        # Mock LLM evaluation output
        self.mock_llm.invoke_json.return_value = {
            "clarity_score": 9,
            "clarity_feedback": "Very clear",
            "accuracy_score": 10,
            "accuracy_feedback": "Perfect",
            "difficulty_score": 9,
            "difficulty_feedback": "Appropriate",
            "mark_alignment_score": 8,
            "mark_alignment_feedback": "Good",
            "overall_qualitative_score": 9.0,
            "general_feedback": "Excellent question."
        }
        
        score, details = self.analyzer.evaluate_qualitative_aspects(flat_questions)
        self.assertEqual(score, 90.0) # overall_qualitative_score 9.0 * 10
        self.assertEqual(details[0]["overall_score"], 90.0)
        self.assertEqual(details[0]["feedback"], "Excellent question.")
        self.assertEqual(details[0]["scores"]["clarity"], 90.0)

    def test_analyze_internal_coherence(self):
        generated_exam = {
            "questions": {
                "Topic 1": [
                    {
                        "number": "1)",
                        "parent_description": "Context about plants.",
                        "subtopic": "Subtopic 1",
                        "sub_questions": [
                            {"label": "a)", "text": "Question about leaf cells.", "marks": 1},
                            {"label": "b)", "text": "Question about xylem transport.", "marks": 2}
                        ]
                    }
                ]
            }
        }
        # Mock get_embeddings to return distinct vectors (so no redundancy warning)
        # We need three vectors: parent, sub_a, sub_b
        self.mock_llm.get_embeddings.return_value = [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5]
        ]
        
        score, details = self.analyzer.analyze_internal_coherence(generated_exam)
        self.assertTrue(score > 80.0)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["parent_question"], "Context about plants.")


if __name__ == '__main__':
    unittest.main()
