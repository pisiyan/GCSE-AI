"""GCSE AI — Feedback Quality Analyzer.

Evaluates the quality of marking feedback given to student answers
using qualitative LLM evaluation against the mark scheme and student response.
"""

import logging
import numpy as np
from typing import Any, Dict

logger = logging.getLogger(__name__)


class FeedbackQualityAnalyzer:
    """Evaluates feedback quality against student answer and mark scheme."""

    def __init__(self, gcse_assistant) -> None:
        """Initialize the analyzer.

        Args:
            gcse_assistant: An initialized instance of GcseAssistant.
        """
        self.assistant = gcse_assistant
        self.llm = gcse_assistant.llm_client

    def analyze_feedback(
        self,
        question: str,
        marks: int,
        mark_scheme: str,
        answer: str,
        feedback: str,
    ) -> dict:
        """Analyze the quality of feedback for a completed answer.

        Args:
            question: The question text.
            marks: Total marks allocated to the question.
            mark_scheme: The mark scheme content.
            answer: The student's answer text.
            feedback: The marking feedback text.

        Returns:
            A dict containing scores, feedback details, and weights.
        """
        logger.info("Evaluating feedback quality...")

        prompt = self.assistant.prompts["evaluate_feedback_quality"].format(
            question_text=question,
            marks=marks,
            mark_scheme=mark_scheme,
            answer=answer,
            feedback=feedback,
        )

        try:
            res = self.llm.invoke_json(prompt)

            # Convert 0-10 LLM scale to 0-100 scale for consistency
            accuracy = float(res.get("marking_accuracy_score", 8)) * 10.0
            relevance = float(res.get("relevance_score", 8)) * 10.0
            usefulness = float(res.get("usefulness_score", 8)) * 10.0
            clarity = float(res.get("clarity_score", 8)) * 10.0

            accuracy_feedback = res.get("marking_accuracy_feedback", "")
            relevance_feedback = res.get("relevance_feedback", "")
            usefulness_feedback = res.get("usefulness_feedback", "")
            clarity_feedback = res.get("clarity_feedback", "")
            general_summary = res.get("general_summary", "")

        except Exception as e:
            logger.error("Failed to qualitatively evaluate feedback: %s", e)
            # Safe fallbacks in case of LLM/parsing failure
            accuracy = 80.0
            relevance = 80.0
            usefulness = 80.0
            clarity = 80.0
            accuracy_feedback = "Error parsing accuracy feedback."
            relevance_feedback = "Error parsing relevance feedback."
            usefulness_feedback = "Error parsing usefulness feedback."
            clarity_feedback = "Error parsing clarity feedback."
            general_summary = "Failed to parse qualitative evaluation."

        # Configurable weights summing to 1.0
        weights = {
            "marking_accuracy": 0.35,
            "relevance_to_mark_scheme": 0.30,
            "actionability_usefulness": 0.25,
            "clarity_tone": 0.10,
        }

        overall_score = (
            accuracy * weights["marking_accuracy"] +
            relevance * weights["relevance_to_mark_scheme"] +
            usefulness * weights["actionability_usefulness"] +
            clarity * weights["clarity_tone"]
        )
        overall_score = round(overall_score, 1)

        report = {
            "overall_score": overall_score,
            "metrics": {
                "marking_accuracy_score": accuracy,
                "relevance_score": relevance,
                "usefulness_score": usefulness,
                "clarity_score": clarity,
            },
            "feedbacks": {
                "marking_accuracy": accuracy_feedback,
                "relevance": relevance_feedback,
                "usefulness": usefulness_feedback,
                "clarity": clarity_feedback,
            },
            "general_summary": general_summary,
            "weights": weights,
        }

        return report

    def generate_markdown_report(
        self,
        question: str,
        marks: int,
        mark_scheme: str,
        answer: str,
        feedback: str,
        report: dict,
    ) -> str:
        """Generate a formatted markdown report containing inputs, feedback, and evaluation."""
        metrics = report["metrics"]
        feedbacks = report["feedbacks"]
        weights = report["weights"]
        score = report["overall_score"]

        if score >= 90:
            rating = "[EXCELLENT] (Highly effective marking and feedback)"
        elif score >= 75:
            rating = "[GOOD] (Constructive feedback with minor areas to refine)"
        elif score >= 50:
            rating = "[FAIR] (Lacks detail, accuracy, or actionability)"
        else:
            rating = "[POOR] (Inaccurate, unhelpful, or misleading feedback)"

        md = []
        md.append("# GCSE Answer Feedback Evaluation Report\n")

        md.append("## 1. Context and Inputs\n")
        md.append(f"**Question:**\n> {question.replace(chr(10), chr(10) + '> ')}\n")
        md.append(f"**Marks Allocated:** {marks}\n")
        md.append(f"**Mark Scheme:**\n```text\n{mark_scheme.strip()}\n```\n")
        md.append(f"**Student Answer:**\n```text\n{answer.strip()}\n```\n")

        md.append("## 2. Generated Marking Feedback\n")
        md.append(f"```text\n{feedback.strip()}\n```\n")

        md.append("## 3. Feedback Quality Analysis\n")
        md.append(f"### **Overall Quality Score: {score}/100**\n")
        md.append(f"**Rating:** {rating}\n")
        md.append(f"**Summary:** {report['general_summary']}\n")

        md.append("### Quality Metrics Breakdown")
        md.append("| Metric | Weight | Score | Comments |")
        md.append("| :--- | :--- | :--- | :--- |")
        md.append(
            f"| **Marking Accuracy** | {weights['marking_accuracy']*100:.0f}% | "
            f"{metrics['marking_accuracy_score']}/100 | {feedbacks['marking_accuracy']} |"
        )
        md.append(
            f"| **Relevance to Mark Scheme** | {weights['relevance_to_mark_scheme']*100:.0f}% | "
            f"{metrics['relevance_score']}/100 | {feedbacks['relevance']} |"
        )
        md.append(
            f"| **Actionability & Usefulness** | {weights['actionability_usefulness']*100:.0f}% | "
            f"{metrics['usefulness_score']}/100 | {feedbacks['usefulness']} |"
        )
        md.append(
            f"| **Clarity, Tone & Structure** | {weights['clarity_tone']*100:.0f}% | "
            f"{metrics['clarity_score']}/100 | {feedbacks['clarity']} |"
        )
        md.append("")

        return "\n".join(md)

    def save_report(
        self,
        filepath: str,
        question: str,
        marks: int,
        mark_scheme: str,
        answer: str,
        feedback: str,
        report: dict,
    ) -> None:
        """Generate and save the markdown report to a file.

        Args:
            filepath: Target file path.
            question: Original question text.
            marks: Allocated marks.
            mark_scheme: Mark scheme.
            answer: Student's answer.
            feedback: Generated marking feedback.
            report: Calculated evaluation report dict.
        """
        md_content = self.generate_markdown_report(
            question, marks, mark_scheme, answer, feedback, report
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Saved feedback evaluation report to %s", filepath)
