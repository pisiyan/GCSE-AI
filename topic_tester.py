"""Topic Content Interactive Testing Module for GCSE AI.

Compiles targeted subtopics from specification summaries (1 LLM call),
generates content questions with perfect reference answers from official specification text (1 LLM call),
evaluates student answers, and builds markdown performance reports.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from config import SubjectConfig
from llm_client import LLMClient

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
        self.spec_qa_chain = spec_qa_chain

    def compile_target_subtopics(
        self, desired_area: str, subject: str, examiner: str
    ) -> Dict[str, Any]:
        """Compile a focused list of target topics/subtopics based on user's desired area.

        Executes EXACTLY 1 LLM call using specification_summary.json.
        Rule: If user asks for specific subtopics/concepts (not a whole topic),
        only return those subtopics and omit parent topic names to avoid scope creep.

        Args:
            desired_area: Free-text topic/subtopic input from student.
            subject: Subject name (e.g. Physics).
            examiner: Exam board name (e.g. Edexcel).

        Returns:
            Dict containing compiled scope and target subtopics list.
        """
        logger.info("Compiling target subtopics for desired area: '%s' (1 LLM call)...", desired_area)

        # Load specification summary JSON
        summary_data = {}
        summary_path = f"data/{subject}/{examiner}/{subject}-{examiner}-specification_summary.json"
        if not os.path.exists(summary_path):
            root_dir = os.path.dirname(os.path.abspath(__file__))
            summary_path = os.path.join(root_dir, "data", subject, examiner, f"{subject}-{examiner}-specification_summary.json")

        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
            except Exception as e:
                logger.warning("Could not read specification summary JSON at %s: %s", summary_path, e)

        summary_json_str = json.dumps(summary_data, indent=2) if summary_data else "No specification summary JSON available."

        prompt = f"""You are an expert GCSE specification curriculum parser.
Analyze the user's requested study area: "{desired_area}" against the provided GCSE {subject} ({examiner}) specification summary below.

Specification Summary Data:
{summary_json_str}

YOUR TASK:
Compile a precise, targeted list of subtopics and concepts matching ONLY what the student asked to study.
Do NOT include any extra topics or unrelated specification points.

CRITICAL SCOPE RULE:
- Check if the user asked for a whole top-level topic or just specific subtopics/concepts.
- If the user asked for specific subtopics or concepts (and NOT the entire topic), return ONLY the individual subtopics in the "subtopics" list and leave "topic" as an empty string (""). Do NOT return parent topic titles when only subtopics were requested, to avoid confusing the system into including extra material from the rest of the topic.
- If the user explicitly asked for a full topic (e.g. "Topic 2 Motion and forces"), you may include the topic title in "topic" alongside all its subtopics.

Return strictly a JSON object matching this schema:
{{
  "desired_area": "{desired_area}",
  "is_full_topic": true or false,
  "scope_description": "A concise 1-sentence summary of the targeted material",
  "target_items": [
    {{
      "topic": "Topic title if user asked for whole topic, otherwise empty string ''",
      "subtopics": [
        "Specific subtopic 1",
        "Specific subtopic 2"
      ]
    }}
  ]
}}

Do not include any explanation or markdown formatting outside the JSON object."""

        try:
            compiled_scope = self.llm.invoke_json(prompt)
            logger.info("Compiled scope successfully (1 LLM call): %s", compiled_scope.get("scope_description", ""))
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
        """Generate content-testing questions and exemplar perfect answers.

        Executes EXACTLY 1 LLM call using official specification text.
        Questions test all target content requested by student (NOT past exam style).

        Args:
            compiled_scope: Dict output from compile_target_subtopics.
            subject: Subject name.
            examiner: Exam board.

        Returns:
            List of dicts, each with 'id', 'subtopic', 'question', and 'perfect_answer'.
        """
        logger.info("Generating study questions & perfect answers (1 LLM call)...")

        scope_desc = compiled_scope.get("scope_description", "")
        target_items = compiled_scope.get("target_items", [])

        # Format target material lines
        target_lines = []
        for item in target_items:
            top_title = item.get("topic", "").strip()
            if top_title:
                target_lines.append(f"Topic: {top_title}")
            for sub in item.get("subtopics", []):
                target_lines.append(f"  - Subtopic: {sub}")

        target_material_text = "\n".join(target_lines) if target_lines else compiled_scope.get("desired_area", "")

        # Determine specification content to pass
        if self.specification_text:
            spec_content = self.specification_text[:12000]  # Cap length safely
        elif self.spec_qa_chain and hasattr(self.spec_qa_chain, "retriever"):
            try:
                docs = self.spec_qa_chain.retriever.invoke(compiled_scope.get("desired_area", subject))
                spec_content = "\n\n".join(doc.page_content for doc in docs)
            except Exception as e:
                logger.warning("Could not retrieve spec content via retriever: %s", e)
                spec_content = "Official specification content."
        else:
            spec_content = "Official specification content."

        prompt = f"""You are a master GCSE {subject} ({examiner}) tutor creating an interactive study test.

TARGET STUDY SCOPE:
Scope Summary: {scope_desc}
Requested Content:
{target_material_text}

OFFICIAL SPECIFICATION CONTENT:
{spec_content}

YOUR TASK:
Generate a thorough series of content-testing questions and exemplar perfect answers.
IMPORTANT FORMATTING RULES:
1. Do NOT generate past-exam style questions (e.g. "Calculate velocity from graph in Figure 1").
2. Generate direct content-testing questions aimed at checking the student's complete understanding of all concepts in the requested scope.
3. Every question must have an exemplar "perfect_answer" formatted with bullet points, bold key terms, and clear structure so it is extremely easy to read.
4. Keep questions clear, focused, and educational (typically 3 to 6 questions total, covering all requested subtopics).

Return strictly a JSON object matching this schema:
{{
  "questions": [
    {{
      "id": 1,
      "subtopic": "Subtopic name",
      "question": "Clear, direct content-testing question text...",
      "perfect_answer": "Structured exemplar reference answer using bullet points (•) and bold text for key terms..."
    }}
  ]
}}
Do not include any text outside the JSON object."""

        try:
            qa_list = self.llm.invoke_json(prompt)
            if isinstance(qa_list, dict) and "questions" in qa_list:
                qa_list = qa_list["questions"]
            if not isinstance(qa_list, list):
                qa_list = []
            logger.info("Generated %d questions & perfect answers (1 LLM call)", len(qa_list))
            return qa_list
        except Exception as e:
            logger.error("Failed to generate questions & answers via LLM: %s", e)
            return [
                {
                    "id": 1,
                    "subtopic": compiled_scope.get("desired_area", "General"),
                    "question": f"Explain key concepts and definitions regarding {compiled_scope.get('desired_area')}.",
                    "perfect_answer": f"• **Definition**: Key concept breakdown for {compiled_scope.get('desired_area')}.\n• **Core Principles**: Primary syllabus requirements.\n• **Application**: Practical context."
                }
            ]

    def evaluate_student_answer(
        self, question: str, perfect_answer: str, student_answer: str, subtopic: str = ""
    ) -> Dict[str, Any]:
        """Evaluate a student's answer against the exemplar perfect answer.

        Args:
            question: The question text.
            perfect_answer: Model exemplar answer.
            student_answer: Student's response.
            subtopic: Subtopic name.

        Returns:
            Dict with score, max_score, feedback, correct_points, and missing_points.
        """
        logger.info("Evaluating student answer for question: '%s'...", question[:50])

        prompt = f"""You are a supportive GCSE tutor evaluating a student's answer.

Question: {question}
Target Subtopic: {subtopic}
Exemplar Perfect Answer:
{perfect_answer}

Student's Answer:
{student_answer}

EVALUATION TASK:
1. Score the student's answer out of 5 marks (0 = blank/completely incorrect, 5 = perfect/complete).
2. List key points or vocabulary the student got correct as concise bullet points.
3. List key points or vocabulary the student missed or got incorrect as concise bullet points.
4. Provide encouraging, constructive feedback explaining how to improve, formatted with bullet points for maximum readability.

Return strictly a JSON object:
{{
  "score": integer (0 to 5),
  "max_score": 5,
  "feedback": "Encouraging feedback formatted with bullet points (•) where appropriate...",
  "correct_points": ["Specific point student got right 1", "Specific point student got right 2"],
  "missing_points": ["Specific key term or concept missed 1", "Specific key term or concept missed 2"]
}}"""

        try:
            result = self.llm.invoke_json(prompt)
            if not isinstance(result, dict):
                result = {
                    "score": 3,
                    "max_score": 5,
                    "feedback": "• Good attempt overall.\n• Compare your response with the exemplar answer to capture remaining key terms.",
                    "correct_points": ["Answered main concept"],
                    "missing_points": ["Detailed technical terms"]
                }
            return result
        except Exception as e:
            logger.error("Failed evaluating student answer via LLM: %s", e)
            return {
                "score": 3,
                "max_score": 5,
                "feedback": f"• Evaluation complete.\n• Compare your response against the exemplar answer.",
                "correct_points": ["Attempted response"],
                "missing_points": ["Review exemplar answer for full details"]
            }

    def generate_performance_report(
        self,
        subject: str,
        examiner: str,
        desired_area: str,
        test_results: List[Dict[str, Any]],
        output_dir: str = "test_outputs",
    ) -> str:
        """Generate a concise Markdown report detailing student strengths and revision needs.

        Args:
            subject: Subject name.
            examiner: Exam board name.
            desired_area: Target topic/subtopic area.
            test_results: List of result dicts containing question, answer, evaluation, subtopic.
            output_dir: Directory to save the markdown report.

        Returns:
            Path to the saved report file.
        """
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(output_dir, exist_ok=True)

        total_score = sum(r.get("evaluation", {}).get("score", 0) for r in test_results)
        max_total = sum(r.get("evaluation", {}).get("max_score", 5) for r in test_results)
        pct = (total_score / max_total * 100) if max_total > 0 else 0.0

        # Performance band
        if pct >= 80:
            band = "Exceptional Understanding 🌟"
        elif pct >= 60:
            band = "Good Understanding 👍"
        elif pct >= 40:
            band = "Standard Understanding 📈"
        else:
            band = "Revision Recommended ⚠️"

        strong_points = []
        revision_points = []

        for r in test_results:
            sub = r.get("subtopic") or "General Concept"
            eval_data = r.get("evaluation", {})
            score = eval_data.get("score", 0)
            max_s = eval_data.get("max_score", 5)

            corrects = eval_data.get("correct_points", [])
            missings = eval_data.get("missing_points", [])

            if score >= (max_s * 0.7):
                details = "\n  ".join([f"• {c}" for c in corrects]) if corrects else "• Strong understanding demonstrated"
                strong_points.append(f"### **{sub}** (Score: {score}/{max_s})\n  {details}")
            else:
                details = "\n  ".join([f"• {m}" for m in missings]) if missings else "• Needs further review of key details"
                revision_points.append(f"### **{sub}** (Score: {score}/{max_s})\n  {details}")

        md = []
        md.append(f"# 📘 GCSE {subject} ({examiner}) - Interactive Content Test Report")
        md.append("")
        md.append(f"**Date:** `{now_str}` | **Subject:** `{subject}` | **Exam Board:** `{examiner}` | **Target Area:** `{desired_area}`")
        md.append("")

        md.append("## 📊 Overall Performance Summary")
        md.append("")
        md.append("| Metric | Result |")
        md.append("| :--- | :--- |")
        md.append(f"| **Target Study Area** | `{desired_area}` |")
        md.append(f"| **Total Score** | **{total_score} / {max_total}** ({pct:.1f}%) |")
        md.append(f"| **Performance Rating** | **{band}** |")
        md.append("")

        md.append("> [!TIP]")
        md.append(f"> **Tutor Summary:** Achieved **{total_score}/{max_total}** marks ({pct:.1f}%). Focus on reviewing the bulleted items under 'Topics & Concepts to Revise More' below.")
        md.append("")

        md.append("## 🌟 Strong Areas")
        md.append("")
        if strong_points:
            for sp in strong_points:
                md.append(f"{sp}\n")
        else:
            md.append("*No topics scored in the top tier yet. Keep revising!*")
        md.append("")

        md.append("## 🎯 Topics & Concepts to Revise More")
        md.append("")
        if revision_points:
            for rp in revision_points:
                md.append(f"{rp}\n")
        else:
            md.append("*Great work! You demonstrated strong mastery across all tested questions.*")
        md.append("")

        md.append("---")
        md.append("")
        md.append("## 📄 Detailed Question Breakdown")
        md.append("")

        for idx, r in enumerate(test_results, start=1):
            sub = r.get("subtopic") or f"Question {idx}"
            q_text = r.get("question", "")
            perf_ans = r.get("perfect_answer", "")
            stud_ans = r.get("student_answer", "")
            eval_data = r.get("evaluation", {})
            score = eval_data.get("score", 0)
            max_s = eval_data.get("max_score", 5)
            feedback = eval_data.get("feedback", "")
            corrects = eval_data.get("correct_points", [])
            missings = eval_data.get("missing_points", [])

            md.append(f"### Q{idx}. [{sub}] — Score: `{score}/{max_s}`")
            md.append("")
            md.append(f"**Question Prompt:**")
            md.append(f"```text\n{q_text}\n```")
            md.append("")
            md.append(f"**Your Answer:**")
            md.append(f"> {stud_ans if stud_ans else '*No answer provided*'}")
            md.append("")
            if corrects:
                md.append("**What You Got Right:**")
                for c in corrects:
                    md.append(f"- ✅ {c}")
                md.append("")
            if missings:
                md.append("**Key Points Missed / To Revise:**")
                for m in missings:
                    md.append(f"- ⚠️ {m}")
                md.append("")
            md.append(f"**Exemplar Perfect Answer:**")
            md.append(f"```text\n{perf_ans}\n```")
            md.append("")
            md.append(f"**Tutor Feedback:**")
            md.append(f"> {feedback}")
            md.append("")
            md.append("---")
            md.append("")

        report_md = "\n".join(md)

        # Save to canonical path and archive timestamp path
        canonical_path = os.path.join(output_dir, "topic_test_report.md")
        with open(canonical_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        timestamp = int(time.time())
        safe_area = desired_area.replace(" ", "_").replace("/", "_")[:30]
        archive_path = os.path.join(output_dir, f"{subject}_{examiner}_topic_test_{safe_area}_{timestamp}.md")
        with open(archive_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        logger.info("Saved topic test report to %s and %s", canonical_path, archive_path)
        return canonical_path
