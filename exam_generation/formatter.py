"""Markdown formatting and text cleaning utilities for exam questions and papers."""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_answer_lines(marks: Any, lines_per_mark: int = 2) -> str:
    """Generate clean answer space lines for students to write responses.

    Args:
        marks: Available marks for the question.
        lines_per_mark: Number of lines per mark.

    Returns:
        Formatted string containing answer lines.
    """
    try:
        m_val = int(marks)
    except (ValueError, TypeError):
        m_val = 1
    if m_val <= 0:
        m_val = 1

    num_lines = max(2, min(m_val * lines_per_mark, 16))
    line_str = "_" * 68
    return "\n".join(["  " + line_str for _ in range(num_lines)])


def fix_ocr_formatting(text: str) -> str:
    """Reconstruct split words and clean up OCR spacing artifacts from raw PDF text."""
    if not text:
        return ""

    cleaned = text

    # Remove non-standard OCR symbol artifacts and bullet boxes
    cleaned = re.sub(
        r"[\u2022\u25a0\u25a1\u2571\u2572\u274c\u2713\ufffd\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\u2069\u206a-\u206f\u2027\u2219\u25aa\u25ab\u25c6\u25c7\u25a2\u25a3\u25a4\u25a5\u25a6\u25a7\u25a8\u25a9\u25aa\u25ab\u25ac\u25ad\u25ae\u25af\u25b0\u25b1\u25b2\u25b3\u25b4\u25b5\u25b6\u25b7\u25b8\u25b9\u25ba\u25bb\u25bc\u25bd\u25be\u25bf\u25c0\u25c1\u25c2\u25c3\u25c4\u25c5\u25c6\u25c7\u25c8\u25c9\u25ca\u25cb\u25cc\u25cd\u25ce\u25cf\u25d0-\u25ff\uf000-\uf8ff]",
        "",
        cleaned,
    )

    # Reconstruct words split across line breaks
    cleaned = re.sub(r"(\b[a-zA-Z]{3,})\s*[\r\n]+\s*([a-zA-Z]{1,3}\b)", r"\1\2", cleaned)
    cleaned = re.sub(r"(\b[a-zA-Z]+)\s*-\s*[\r\n]+\s*([a-zA-Z]+)\b", r"\1-\2", cleaned)

    # Reconstruct hyphenated words split by spaces
    cleaned = re.sub(r"(\b[a-zA-Z]+)\s*-\s*([a-zA-Z]+)\b", r"\1-\2", cleaned)

    # Reconstruct single/double letter split words inside lines
    cleaned = re.sub(r"\b([a-zA-Z]{3,})\s+([b-zB-Z]{1,2})\b", r"\1\2", cleaned)
    cleaned = re.sub(r"\b([b-hj-zB-HJ-Z]{1})\s+([a-zA-Z]{3,})\b", r"\1\2", cleaned)

    # General glued words fixes
    glued_fixes = {
        "Figur e": "Figure", "figur e": "figure",
        "in vasive": "invasive", "indica tes": "indicates", "indica te": "indicate", "fertil iser": "fertiliser",
        "reac tion": "reaction", "concen tration": "concentration", "tempe rature": "temperature",
        "expe riment": "experiment", "investiga tion": "investigation",
        r"\bcolourof\b": "colour of", r"\bshownin\b": "shown in", r"\btestingfor\b": "testing for",
        r"\bcanbe\b": "can be", r"\bsolutionof\b": "solution of", r"\bsolutionsof\b": "solutions of",
        r"\bsolutionw as\b": "solution was", r"\bwateris\b": "water is",
    }
    for wrong, right in glued_fixes.items():
        cleaned = re.sub(wrong, right, cleaned, flags=re.IGNORECASE)

    # Format Multiple Choice options cleanly into bullets if detected
    mc_pattern = r"\bA\s+([^\nB]+?)\s+B\s+([^\nC]+?)\s+C\s+([^\nD]+?)\s+D\s+([^\n]+)"
    match = re.search(mc_pattern, cleaned)
    if match:
        a_opt = match.group(1).strip()
        b_opt = match.group(2).strip()
        c_opt = match.group(3).strip()
        d_opt = match.group(4).strip()
        mc_formatted = f"\n\n  * **A** {a_opt}\n  * **B** {b_opt}\n  * **C** {c_opt}\n  * **D** {d_opt}"
        cleaned = cleaned[:match.start()] + mc_formatted + cleaned[match.end():]

    return cleaned


def clean_question_text(text: str) -> str:
    """Clean raw question content by removing exam board boilerplate, codes, and junk text."""
    if not text:
        return ""

    cleaned = text

    # Remove margin instructions with OCR spacing glitches
    cleaned = re.sub(r"(?i)DO\s*NO?\s*T?\s*WRITE?\s*IN?\s*THIS?\s*(?:AREA|MARGIN|PAGE).*", "", cleaned)
    cleaned = re.sub(r"(?i)\(?\s*Total\s+for\s+Question.*", "", cleaned)
    cleaned = re.sub(r"(?i)\(?\s*Total\s+\d+\s+marks?\s*\)?", "", cleaned)

    # Remove exam paper instructions & boilerplate
    cleaned = re.sub(r"(?i)Answer\s+ALL\s+questions.*", "", cleaned)
    cleaned = re.sub(r"(?i)Write\s+your\s+answers\s+in\s+the\s+spaces\s+provided.*", "", cleaned)
    cleaned = re.sub(r"(?i)Some\s+questions\s+must\s+be\s+answered\s+with\s+a\s+cross.*", "", cleaned)
    cleaned = re.sub(r"(?i)If\s+you\s+change\s+your\s+mind.*", "", cleaned)
    cleaned = re.sub(r"(?i)mark\s+your\s+new\s+answer.*", "", cleaned)
    cleaned = re.sub(r"(?i)\bTurn\s+over\b", "", cleaned)
    cleaned = re.sub(r"(?i)Pearson\s+Edexcel.*", "", cleaned)
    cleaned = re.sub(r"\b[A-Z]\d{4,}[A-Z0-9]*\b", "", cleaned)
    cleaned = re.sub(r"\*\s*[A-Z0-9]{5,}\s*\*", "", cleaned)

    # Remove long dotted or underlined prompt lines
    cleaned = re.sub(r"\.{3,}", "", cleaned)
    cleaned = re.sub(r"_{3,}", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\(\d+\)\s*$", "", cleaned)

    # Reconstruct OCR words & split text
    cleaned = fix_ocr_formatting(cleaned)

    # Collapse multiple blank lines & spaces
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines)


def _safe_marks_str(marks: Any) -> str:
    """Format marks string safely, handling None or non-integer values gracefully."""
    if marks is None:
        return ""
    try:
        m = int(marks)
        return f" *({m} mark{'s' if m > 1 else ''})*"
    except (ValueError, TypeError):
        return ""


def format_single_question_as_markdown(subject: str, examiner: str, q: dict) -> str:
    """Format a single generated question dictionary into clean markdown."""
    md = []
    marks = q.get("marks")
    subtopic = q.get("subtopic", "")
    matched_subtopic = q.get("matched_subtopic", "")
    sim_score = q.get("similarity_score")

    marks_str = _safe_marks_str(marks)
    md.append(f"# 📝 GCSE {subject} ({examiner}) - Single Question{marks_str}")
    md.append("")
    md.append(f"**Subject:** `{subject}` | **Exam Board:** `{examiner}`")

    if subtopic:
        md.append(f"* **Target Subtopic:** `{subtopic}`")
    if matched_subtopic:
        md.append(f"* **Matched Question Subtopic:** `{matched_subtopic}`")
    if sim_score is not None:
        sim_pct = f"{sim_score * 100:.1f}%" if isinstance(sim_score, float) and sim_score <= 1.0 else f"{sim_score}"
        md.append(f"* **Semantic Similarity:** `{sim_pct}`")
    md.append("")

    if "sub_questions" in q:
        p_desc = q.get("parent_description", "")
        if p_desc:
            md.append("### Question Context")
            md.append(p_desc)
            md.append("")
        md.append("### Sub-Questions")
        for sq in q.get("sub_questions", []):
            label = sq.get("label", "")
            sq_text = sq.get("text", "")
            sq_marks = sq.get("marks")
            md.append(f"* **{label}** {sq_text}{_safe_marks_str(sq_marks)}")
            md.append("")
            md.append(generate_answer_lines(sq_marks))
            md.append("")
    else:
        q_text = q.get("text", "")
        md.append("### Question")
        md.append(q_text)
        md.append("")
        md.append(generate_answer_lines(marks))
        md.append("")

    return "\n".join(md)


def format_exam_as_markdown(subject: str, examiner: str, exam_type: str, exam_data: dict) -> str:
    """Format a generated exam dictionary into a clean markdown document."""
    md = []
    md.append(f"# GCSE {subject} ({examiner}) - {exam_type} Exam")
    md.append("")

    total_marks = 0
    questions_by_topic = exam_data.get("questions", {})
    for topic, q_list in questions_by_topic.items():
        for q in q_list:
            if "sub_questions" in q:
                if q.get("marks") is not None:
                    total_marks += (q.get("marks") or 0)
                else:
                    for sq in q.get("sub_questions", []):
                        if "sub_parts" in sq:
                            for gq in sq.get("sub_parts", []):
                                total_marks += (gq.get("marks") or 0)
                        else:
                            total_marks += (sq.get("marks") or 0)
            else:
                total_marks += (q.get("marks") or 0)

    md.append(f"**Total Marks:** {total_marks}")
    md.append("")

    spec_trees = exam_data.get("spec_trees", {})
    if spec_trees:
        md.append("## Topic & Specification Breakdown")
        md.append("")
        for topic, tree_data in spec_trees.items():
            if isinstance(tree_data, dict):
                spec_code = tree_data.get("spec_code", "")
                subtopic_names = tree_data.get("subtopics", [])
                if not subtopic_names and "subtopics" not in tree_data:
                    subtopic_names = [k for k in tree_data.keys() if not k.startswith("_")]
            elif isinstance(tree_data, list):
                spec_code = ""
                subtopic_names = tree_data
            else:
                spec_code = ""
                subtopic_names = []

            code_str = f" (Specification Code: {spec_code})" if spec_code else ""
            md.append(f"### Topic: {topic}{code_str}")
            for st_name in subtopic_names:
                md.append(f"- **Subtopic:** {st_name}")
            md.append("")
        md.append("---")
        md.append("")

    for topic, q_list in questions_by_topic.items():
        md.append(f"## Topic: {topic}")
        md.append("")
        for q in q_list:
            q_num = q.get("number", "")
            subtopic = q.get("subtopic", "")
            matched_subtopic = q.get("matched_subtopic", "")
            similarity_score = q.get("similarity_score")

            subtopic_info = []
            if subtopic:
                subtopic_info.append(f"* **Target Subtopic:** {subtopic}")
            if matched_subtopic:
                subtopic_info.append(f"* **Matched Question Subtopic:** {matched_subtopic}")
            if similarity_score is not None:
                sim_pct = f"{similarity_score * 100:.1f}%" if isinstance(similarity_score, float) and similarity_score <= 1.0 else f"{similarity_score}"
                subtopic_info.append(f"* **Semantic Similarity:** `{sim_pct}`")

            subtopic_block = "\n".join(subtopic_info) if subtopic_info else ""

            if "sub_questions" in q:
                parent_desc = q.get("parent_description", "")
                marks_str = _safe_marks_str(q.get("marks"))
                md.append(f"### Question {q_num}{marks_str}")
                if subtopic_block:
                    md.append(subtopic_block)
                    md.append("")
                if parent_desc:
                    md.append(parent_desc)
                    md.append("")

                for sq in q.get("sub_questions", []):
                    label = sq.get("label", "")
                    sq_subtopic = sq.get("subtopic", "")
                    sq_subtopic_str = f" _(Subtopic: {sq_subtopic})_" if sq_subtopic and sq_subtopic != matched_subtopic else ""

                    if "sub_parts" in sq:
                        context = sq.get("context", "")
                        if context:
                            md.append(f"**{label}**{sq_subtopic_str} {context}")
                            md.append("")
                        else:
                            md.append(f"**{label}**{sq_subtopic_str}")
                            md.append("")
                        for gq in sq.get("sub_parts", []):
                            g_label = gq.get("label", "")
                            g_text = gq.get("text", "")
                            g_marks = gq.get("marks")
                            md.append(f"  * **{g_label}** {g_text}{_safe_marks_str(g_marks)}")
                            md.append("")
                            md.append(generate_answer_lines(g_marks))
                            md.append("")
                    else:
                        sq_text = sq.get("text", "")
                        sq_marks = sq.get("marks")
                        md.append(f"* **{label}**{sq_subtopic_str} {sq_text}{_safe_marks_str(sq_marks)}")
                        md.append("")
                        md.append(generate_answer_lines(sq_marks))
                        md.append("")
            else:
                q_text = q.get("text", "")
                q_marks = q.get("marks")
                md.append(f"### Question {q_num}{_safe_marks_str(q_marks)}")
                if subtopic_block:
                    md.append(subtopic_block)
                    md.append("")
                md.append(q_text)
                md.append("")
                md.append(generate_answer_lines(q_marks))
                md.append("")

    return "\n".join(md)
