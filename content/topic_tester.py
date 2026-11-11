"""Topic Content Interactive Testing Module for GCSE AI."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from config import SubjectConfig
from llm.client import LLMClient
from llm.prompts import load_prompt

logger = logging.getLogger(__name__)


class TopicTester:
    """Manages interactive topic content testing workflows."""

    def __init__(
        self,
        config: SubjectConfig,
        llm_client: LLMClient,
        specification_text: str = "",
        spec_qa_chain: Any = None,
    ) -> None:
        self.config = config
        self.llm = llm_client
        self.specification_text = specification_text

    def compile_target_subtopics(
        self, desired_area: str, subject: str, examiner: str
    ) -> Dict[str, Any]:
        """Compile a focused list of target topics/subtopics based on user's desired area."""
        logger.info("Compiling target subtopics for desired area: '%s'...", desired_area)

        summary_data = {}
        summary_path = f"data/{subject}/{examiner}/{subject}-{examiner}-specification_summary.json"
        if not os.path.exists(summary_path):
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            summary_path = os.path.join(root_dir, "data", subject, examiner, f"{subject}-{examiner}-specification_summary.json")

        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
            except Exception as e:
                logger.warning("Could not read specification summary JSON at %s: %s", summary_path, e)

        summary_json_str = json.dumps(summary_data, indent=2) if summary_data else "No specification summary JSON available."

        prompt = load_prompt(
            "compile_target_subtopics",
            desired_area=desired_area,
            subject=subject,
            examiner=examiner,
            summary_json_str=summary_json_str,
        )

        try:
            compiled_scope = self.llm.invoke_json(prompt)
            logger.info("Compiled scope successfully: %s", compiled_scope.get("scope_description", ""))
            return compiled_scope
        except Exception as e:
            logger.error("Failed to compile target subtopics via LLM: %s. Using fallback.", e)
            return {
                "desired_area": desired_area,
                "is_full_topic": False,
                "scope_description": f"Custom study scope for {desired_area}",
                "target_items": [
                    {
                        "topic": "",
                        "subtopics": [desired_area]
                    }
                ]
            }

    def generate_questions_and_answers(
        self, compiled_scope: Dict[str, Any], subject: str, examiner: str
    ) -> List[Dict[str, Any]]:
        """Generate content-testing questions and exemplar perfect answers."""
        logger.info("Generating study questions & perfect answers...")

        target_items = compiled_scope.get("target_items", [])
        target_lines = []
        for item in target_items:
            top_title = item.get("topic", "").strip()
            if top_title:
                target_lines.append(f"Topic: {top_title}")
            for sub in item.get("subtopics", []):
                target_lines.append(f"  - Subtopic: {sub}")

        target_material = "\n".join(target_lines) if target_lines else compiled_scope.get("desired_area", "")
        spec_snippet = self.specification_text[:12000] if self.specification_text else "Official specification content."

        prompt = load_prompt(
            "generate_study_questions",
            subject=subject,
            examiner=examiner,
            target_material=target_material,
            spec_text_snippet=spec_snippet,
        )

        try:
            response_json = self.llm.invoke_json(prompt)
            qa_list = response_json.get("questions", [])
            logger.info("Generated %d questions and answers successfully.", len(qa_list))
            return qa_list
        except Exception as e:
            logger.error("Failed to generate questions & answers: %s. Using fallback.", e)
            return [
                {
                    "id": 1,
                    "subtopic": compiled_scope.get("desired_area", "General"),
                    "question": f"Explain the key concepts of {compiled_scope.get('desired_area', 'this topic')}.",
                    "perfect_answer": f"A full exemplar explanation covering core principles of {compiled_scope.get('desired_area', 'this topic')}."
                }
            ]

    def evaluate_student_answer(
        self, question: str, perfect_answer: str, student_answer: str, subtopic: str = ""
    ) -> Dict[str, Any]:
        """Evaluate a student's answer against the exemplar perfect answer."""
        logger.info("Evaluating student response for subtopic '%s'...", subtopic)

        prompt = load_prompt(
            "evaluate_topic_answer",
            question=question,
            perfect_answer=perfect_answer,
            student_answer=student_answer or "No answer provided.",
        )

        try:
            return self.llm.invoke_json(prompt)
        except Exception as e:
            logger.error("Failed to evaluate answer via LLM: %s. Using fallback.", e)
            return {
                "score": 5,
                "max_score": 10,
                "mastery_status": "Partially Understood",
                "what_went_well": "Attempted the question.",
                "missing_elements": "Key details omitted.",
                "examiner_feedback": "Review the core concepts in the specification."
            }

    def generate_performance_report(
        self,
        subject: str,
        examiner: str,
        desired_area: str,
        test_results: List[Dict[str, Any]],
        output_dir: str = "test_outputs",
    ) -> str:
        """Build and save a comprehensive markdown performance report."""
        os.makedirs(output_dir, exist_ok=True)

        digest_lines = []
        total_score = 0
        total_max = 0

        for idx, item in enumerate(test_results, start=1):
            q_text = item.get("question", "")
            ans = item.get("student_answer", "")
            eval_data = item.get("evaluation", {})
            score = eval_data.get("score", 0)
            max_s = eval_data.get("max_score", 10)
            status = eval_data.get("mastery_status", "N/A")

            total_score += score
            total_max += max_s
            digest_lines.append(
                f"Q{idx} ({item.get('subtopic', '')}): Score {score}/{max_s} ({status})\n"
                f"  - Q: {q_text}\n"
                f"  - Student: {ans}\n"
                f"  - Feedback: {eval_data.get('examiner_feedback', '')}"
            )

        results_digest = "\n\n".join(digest_lines)
        prompt = load_prompt(
            "topic_test_report",
            subject=subject,
            examiner=examiner,
            desired_area=desired_area,
            results_digest=results_digest,
        )

        try:
            ai_analysis = self.llm.invoke(prompt)
        except Exception as e:
            logger.warning("Failed to generate AI analysis for topic test report: %s", e)
            ai_analysis = "Performance summary generated."

        pct = (total_score / total_max * 100) if total_max > 0 else 0
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        md = [
            f"# 📊 GCSE {subject} ({examiner}) — Topic Test Report",
            f"**Area:** `{desired_area}` | **Date:** `{now_str}`",
            "",
            "## 🏆 Executive Summary",
            f"| Total Score | Percentage | Mastery Band |",
            f"| :---: | :---: | :---: |",
            f"| **{total_score}/{total_max}** | **{pct:.1f}%** | **{'Mastery 🌟' if pct >= 75 else ('Developing 👍' if pct >= 50 else 'Revision Needed ⚠️')}** |",
            "",
            ai_analysis,
            "",
            "## 📝 Detailed Question Breakdown",
            "",
        ]

        for idx, item in enumerate(test_results, start=1):
            eval_data = item.get("evaluation", {})
            md.extend([
                f"### Question {idx}: {item.get('subtopic', '')} ({eval_data.get('score', 0)}/{eval_data.get('max_score', 10)} Marks)",
                f"**Question:** {item.get('question', '')}",
                "",
                f"> **Student Answer:** {item.get('student_answer', '*None*')}",
                "",
                f"**Model Answer:** {item.get('perfect_answer', '')}",
                "",
                f"* **What Went Well:** {eval_data.get('what_went_well', '')}",
                f"* **Missing Elements:** {eval_data.get('missing_elements', '')}",
                f"* **Examiner Advice:** {eval_data.get('examiner_feedback', '')}",
                "",
                "---",
                "",
            ])

        report_md = "\n".join(md)
        canonical_path = os.path.join(output_dir, "topic_test_report.md")
        with open(canonical_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        logger.info("Saved topic test report to: %s", canonical_path)
        return canonical_path
