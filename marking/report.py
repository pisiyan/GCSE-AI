"""Marking report formatting and performance summary generation."""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from llm.client import LLMClient
from llm.prompts import load_prompt

logger = logging.getLogger(__name__)


def generate_performance_summary(
    subject: str,
    examiner: str,
    topic: str,
    marking_results: List[Dict[str, Any]],
    llm_client: Optional[LLMClient] = None,
) -> str:
    """Generate an executive performance summary (What Went Well & Areas for Improvement) via LLM call."""
    if not llm_client:
        try:
            llm_client = LLMClient()
        except Exception:
            return ""

    total_available = sum(int(item.get("marks", 0) or 0) for item in marking_results)
    total_awarded = sum(int(item.get("awarded_marks", 0) or 0) for item in marking_results)
    pct = (total_awarded / total_available * 100) if total_available > 0 else 0.0

    digest_lines = []
    for idx, item in enumerate(marking_results, start=1):
        label = item.get("label") or f"Question {idx}"
        q_text = (item.get("question") or "").replace("\n", " ")[:150]
        ans = (item.get("student_answer") or "No answer provided").replace("\n", " ")[:200]
        marks = item.get("marks", 1)
        awarded = item.get("awarded_marks", 0)
        feedback = (item.get("result") or item.get("evaluation") or "").replace("\n", " ")[:200]
        digest_lines.append(f"- [{label}] Marks: {awarded}/{marks} | Q: {q_text} | Student Answer: {ans} | Feedback: {feedback}")

    results_digest = "\n".join(digest_lines)

    try:
        prompt = load_prompt(
            "exam_feedback_summary",
            subject=subject,
            examiner=examiner,
            topic=topic,
            total_available=total_available,
            total_awarded=total_awarded,
            percentage=f"{pct:.1f}",
            results_digest=results_digest,
        )
    except Exception:
        prompt = load_prompt(
            "generate_performance_summary",
            subject=subject,
            examiner=examiner,
            topic=topic,
            results_digest=results_digest,
        )

    logger.info("Generating diagnostic feedback summary for %s (%s)...", subject, topic)
    try:
        summary_text = llm_client.invoke(prompt)
        return summary_text.strip()
    except Exception as e:
        logger.warning("Failed to generate performance summary via LLM: %s", e)
        return ""



def format_marking_as_markdown(
    subject: str,
    examiner: str,
    topic: str,
    marking_results: List[Dict[str, Any]],
    llm_client: Optional[LLMClient] = None,
) -> str:
    """Format exam marking results into a rich, structured markdown document."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# 📝 GCSE {subject} ({examiner}) - {topic} Marking Report")
    md.append("")
    md.append(f"**Date:** `{now_str}` | **Subject:** `{subject}` | **Exam Board:** `{examiner}` | **Paper/Topic:** `{topic}`")
    md.append("")

    total_marks_available = 0
    total_marks_awarded = 0
    has_scores = False

    for item in marking_results:
        m_avail = item.get("marks", 0)
        m_award = item.get("awarded_marks")
        if isinstance(m_avail, int):
            total_marks_available += m_avail
        if isinstance(m_award, int):
            total_marks_awarded += m_award
            has_scores = True

    if has_scores and total_marks_available > 0:
        pct = (total_marks_awarded / total_marks_available) * 100

        if pct >= 80:
            performance = "Grade 9 / Exceptional Performance 🌟"
        elif pct >= 70:
            performance = "Grade 8 / Strong Performance 🚀"
        elif pct >= 60:
            performance = "Grade 7 / Good Performance 👍"
        elif pct >= 50:
            performance = "Grade 6 / Solid Pass ✅"
        elif pct >= 40:
            performance = "Grade 4-5 / Standard Pass 📈"
        else:
            performance = "Grade 1-3 / Revision Required ⚠️"

        md.append("## 📊 Executive Score Summary")
        md.append("")
        md.append("| Total Available Marks | Total Awarded Marks | Overall Percentage | Estimated Performance |")
        md.append("| :---: | :---: | :---: | :---: |")
        md.append(f"| **{total_marks_available}** | **{total_marks_awarded}** | **{pct:.1f}%** | **{performance}** |")
        md.append("")

        if os.path.exists("test_outputs/marked_exam_booklet.pdf"):
            md.append("> 📄 **Annotated Exam Booklet**: Teacher pen markings (highlights, ticks, underlines, and margin cards) are compiled in [`test_outputs/marked_exam_booklet.pdf`](test_outputs/marked_exam_booklet.pdf).")
            md.append("")

        if llm_client:
            summary = generate_performance_summary(subject, examiner, topic, marking_results, llm_client)
            if summary:
                md.append("## 💡 Diagnostic Performance Analysis")
                md.append("")
                md.append(summary)
                md.append("")

        md.append("---")
        md.append("")

    md.append("## 📋 Question Overview & Mark Allocations")
    md.append("")
    md.append("| Question | Marks Available | Marks Awarded | Question Type | Feedback Notes |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")

    for idx, item in enumerate(marking_results, start=1):
        label = item.get("label") or f"Question {idx}"
        q_title = (item.get("question") or "").strip().replace("\n", " ")
        m_avail = item.get("marks", "")
        awarded = item.get("awarded_marks", 0)
        q_type = item.get("question_type") or ("Leveled" if int(m_avail or 0) >= 6 else "Discrete")
        teacher_summary = item.get("teacher_summary") or item.get("result", "")
        summary_one_line = teacher_summary.splitlines()[0] if teacher_summary else "Marked."
        summary_one_line = summary_one_line.replace("|", "/")[:80]

        question_col = f"**{label}**: {q_title}" if q_title else f"**{label}**"
        md.append(f"| {question_col} | {m_avail} | **{awarded}** | {str(q_type).capitalize()} | {summary_one_line} |")


    md.append("")
    md.append("> [!NOTE]")
    md.append("> Detailed sentence-level highlights, error underlines, checkmarks, and margin sticky notes are rendered directly on the question pages in the marked PDF booklet.")

    return "\n".join(md)



def save_marking_report_md(
    subject: str,
    examiner: str,
    topic: str,
    marking_results: List[Dict[str, Any]],
    output_dir: str = "test_outputs",
    llm_client: Optional[LLMClient] = None,
) -> str:
    """Format and save exam marking results to canonical and timestamped markdown files."""
    os.makedirs(output_dir, exist_ok=True)
    report_md = format_marking_as_markdown(subject, examiner, topic, marking_results, llm_client=llm_client)

    canonical_path = os.path.join(output_dir, "marking_report.md")
    with open(canonical_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    timestamp = int(time.time())
    archive_path = os.path.join(output_dir, f"{subject}_{examiner}_marking_{timestamp}.md")
    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("Saved marking markdown report to %s and %s", canonical_path, archive_path)
    return canonical_path
