import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feedback_quality_analyzer import FeedbackQualityAnalyzer


class TestFeedbackQualityAnalyzer(unittest.TestCase):

    def setUp(self):
        self.mock_assistant = MagicMock()
        self.mock_llm = MagicMock()

        self.mock_assistant.llm_client = self.mock_llm
        self.mock_assistant.prompts = {
            "evaluate_feedback_quality": "Mock feedback quality prompt {question_text} {marks} {mark_scheme} {answer} {feedback}"
        }

        self.analyzer = FeedbackQualityAnalyzer(self.mock_assistant)

    def test_analyze_feedback_success(self):
        # Mock successful JSON response from LLM (scores on a 0-10 scale)
        self.mock_llm.invoke_json.return_value = {
            "marking_accuracy_score": 9,
            "marking_accuracy_feedback": "Accurate marking",
            "relevance_score": 10,
            "relevance_feedback": "Perfect relevance",
            "usefulness_score": 8,
            "usefulness_feedback": "Very useful feedback",
            "clarity_score": 9,
            "clarity_feedback": "Clear feedback",
            "overall_score": 9.1,
            "general_summary": "Highly effective feedback and accurate grading."
        }

        report = self.analyzer.analyze_feedback(
            question="Explain photosynthesis.",
            marks=4,
            mark_scheme="Light + CO2 + Water -> Glucose + Oxygen",
            answer="Plants use light, water, and CO2 to produce glucose.",
            feedback="Good answer. Awarded [3/4] marks."
        )

        # Expected overall score calculations:
        # weights = Accuracy: 35%, Relevance: 30%, Usefulness: 25%, Clarity: 10%
        # (9*10)*0.35 + (10*10)*0.30 + (8*10)*0.25 + (9*10)*0.10
        # = 31.5 + 30.0 + 20.0 + 9.0 = 90.5
        self.assertEqual(report["overall_score"], 90.5)
        self.assertEqual(report["metrics"]["marking_accuracy_score"], 90.0)
        self.assertEqual(report["metrics"]["relevance_score"], 100.0)
        self.assertEqual(report["metrics"]["usefulness_score"], 80.0)
        self.assertEqual(report["metrics"]["clarity_score"], 90.0)
        self.assertEqual(report["feedbacks"]["marking_accuracy"], "Accurate marking")
        self.assertEqual(report["general_summary"], "Highly effective feedback and accurate grading.")

    def test_analyze_feedback_fallback_on_exception(self):
        # Mock LLM raising an exception (e.g. timeout or JSON parse issue)
        self.mock_llm.invoke_json.side_effect = Exception("LLM connection error")

        report = self.analyzer.analyze_feedback(
            question="Explain photosynthesis.",
            marks=4,
            mark_scheme="Light + CO2 + Water -> Glucose + Oxygen",
            answer="Plants use light, water, and CO2 to produce glucose.",
            feedback="Good answer. Awarded [3/4] marks."
        )

        # Should fall back gracefully to 80.0 for all metrics
        # Overall score = 80.0*0.35 + 80.0*0.30 + 80.0*0.25 + 80.0*0.10 = 80.0
        self.assertEqual(report["overall_score"], 80.0)
        self.assertEqual(report["metrics"]["marking_accuracy_score"], 80.0)
        self.assertEqual(report["feedbacks"]["marking_accuracy"], "Error parsing accuracy feedback.")
        self.assertEqual(report["general_summary"], "Failed to parse qualitative evaluation.")

    def test_generate_markdown_report(self):
        report_data = {
            "overall_score": 90.5,
            "metrics": {
                "marking_accuracy_score": 90.0,
                "relevance_score": 100.0,
                "usefulness_score": 80.0,
                "clarity_score": 90.0,
            },
            "feedbacks": {
                "marking_accuracy": "Accurate marking",
                "relevance": "Perfect relevance",
                "usefulness": "Very useful feedback",
                "clarity": "Clear feedback",
            },
            "general_summary": "Highly effective feedback and accurate grading.",
            "weights": {
                "marking_accuracy": 0.35,
                "relevance_to_mark_scheme": 0.30,
                "actionability_usefulness": 0.25,
                "clarity_tone": 0.10,
            }
        }

        md = self.analyzer.generate_markdown_report(
            question="Explain photosynthesis.",
            marks=4,
            mark_scheme="Light + CO2 + Water -> Glucose + Oxygen",
            answer="Plants use light, water, and CO2 to produce glucose.",
            feedback="Good answer. Awarded [3/4] marks.",
            report=report_data
        )

        self.assertIn("# GCSE Answer Feedback Evaluation Report", md)
        self.assertIn("**Question:**\n> Explain photosynthesis.", md)
        self.assertIn("Light + CO2 + Water -> Glucose + Oxygen", md)
        self.assertIn("Plants use light, water, and CO2 to produce glucose.", md)
        self.assertIn("Good answer. Awarded [3/4] marks.", md)
        self.assertIn("### **Overall Quality Score: 90.5/100**", md)
        self.assertIn("Accurate marking", md)
        self.assertIn("Perfect relevance", md)
        self.assertIn("Very useful feedback", md)
        self.assertIn("Clear feedback", md)
        self.assertIn("Highly effective feedback and accurate grading.", md)

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_save_report(self, mock_file_open):
        report_data = {
            "overall_score": 90.5,
            "metrics": {
                "marking_accuracy_score": 90.0,
                "relevance_score": 100.0,
                "usefulness_score": 80.0,
                "clarity_score": 90.0,
            },
            "feedbacks": {
                "marking_accuracy": "Accurate marking",
                "relevance": "Perfect relevance",
                "usefulness": "Very useful feedback",
                "clarity": "Clear feedback",
            },
            "general_summary": "Highly effective feedback and accurate grading.",
            "weights": {
                "marking_accuracy": 0.35,
                "relevance_to_mark_scheme": 0.30,
                "actionability_usefulness": 0.25,
                "clarity_tone": 0.10,
            }
        }

        self.analyzer.save_report(
            filepath="dummy_path.md",
            question="Explain photosynthesis.",
            marks=4,
            mark_scheme="Light + CO2 + Water -> Glucose + Oxygen",
            answer="Plants use light, water, and CO2 to produce glucose.",
            feedback="Good answer. Awarded [3/4] marks.",
            report=report_data
        )

        mock_file_open.assert_called_once_with("dummy_path.md", "w", encoding="utf-8")
        # Ensure it wrote the generated markdown content
        mock_file_open().write.assert_called()


if __name__ == '__main__':
    unittest.main()
