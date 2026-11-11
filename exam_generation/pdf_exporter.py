"""PDF export module for compiling GCSE AI generated exams into authentic exam papers."""

import logging
import os
import re
from typing import Any, Dict, List, Optional
import pymupdf

logger = logging.getLogger(__name__)


def render_exam_pdf(
    exam_output: Dict[str, Any],
    output_pdf_path: str,
    subject: str = "Science",
    examiner: str = "GCSE",
) -> str:
    """Render a generated exam dictionary into a styled past-paper PDF document.

    Args:
        exam_output: Exam dict containing 'questions' topic groups.
        output_pdf_path: Filepath where PDF will be saved.
        subject: Subject title.
        examiner: Exam board name.

    Returns:
        Path of the rendered PDF file.
    """
    os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)
    doc = pymupdf.open()

    page_width = 595.0
    page_height = 842.0
    margin_left = 50.0
    margin_right = 545.0
    content_width = margin_right - margin_left

    current_page = doc.new_page(width=page_width, height=page_height)
    cursor_y = 60.0

    def check_page_space(needed_height: float) -> pymupdf.Page:
        nonlocal current_page, cursor_y
        if cursor_y + needed_height > page_height - 60.0:
            current_page = doc.new_page(width=page_width, height=page_height)
            cursor_y = 60.0
        return current_page

    # Header
    current_page.insert_text(
        pymupdf.Point(margin_left, cursor_y),
        f"GCSE {subject.upper()} ({examiner.upper()}) EXAMINATION",
        fontsize=16,
        fontname="helv",
    )
    cursor_y += 24.0

    # Calculate total marks
    total_marks = 0
    questions_data = exam_output.get("questions", {})
    for topic, q_list in questions_data.items():
        for q in q_list:
            if "sub_questions" in q:
                if q.get("marks"):
                    total_marks += q.get("marks")
                else:
                    for sq in q.get("sub_questions", []):
                        total_marks += sq.get("marks", 0)
            else:
                total_marks += q.get("marks", 0)

    current_page.insert_text(
        pymupdf.Point(margin_left, cursor_y),
        f"Time allowed: Standard | Total marks available: {total_marks}",
        fontsize=10,
        fontname="helv",
    )
    cursor_y += 15.0

    current_page.draw_line(
        pymupdf.Point(margin_left, cursor_y),
        pymupdf.Point(margin_right, cursor_y),
        width=1.0,
    )
    cursor_y += 20.0

    # Render questions
    q_counter = 1
    for topic, q_list in questions_data.items():
        for q in q_list:
            check_page_space(60.0)

            # Render Question stem / parent context
            if "sub_questions" in q:
                p_marks = q.get("marks", "")
                marks_label = f" ({p_marks} marks)" if p_marks else ""
                q_heading = f"Question {q_counter}{marks_label} [{topic}]"
                current_page.insert_text(
                    pymupdf.Point(margin_left, cursor_y),
                    q_heading,
                    fontsize=12,
                    fontname="helv",
                )
                cursor_y += 18.0

                p_desc = q.get("parent_description", "")
                if p_desc:
                    rect = pymupdf.Rect(margin_left, cursor_y, margin_right, cursor_y + 120.0)
                    rc = current_page.insert_textbox(rect, p_desc, fontsize=10, fontname="helv")
                    cursor_y += 40.0

                # Embed diagram if available
                img_paths = q.get("image_paths", [])
                if img_paths:
                    for img_p in img_paths:
                        if os.path.exists(img_p):
                            check_page_space(160.0)
                            img_rect = pymupdf.Rect(margin_left + 40, cursor_y, margin_right - 40, cursor_y + 140)
                            try:
                                current_page.insert_image(img_rect, filename=img_p)
                                cursor_y += 150.0
                            except Exception as e:
                                logger.warning("Failed to insert diagram image %s: %s", img_p, e)

                for sq in q.get("sub_questions", []):
                    check_page_space(70.0)
                    sq_label = sq.get("label", "")
                    sq_text = sq.get("text", "")
                    sq_marks = sq.get("marks", 1)

                    full_sq_text = f"{sq_label} {sq_text} ({sq_marks} mark{'s' if sq_marks > 1 else ''})"
                    rect = pymupdf.Rect(margin_left + 15, cursor_y, margin_right, cursor_y + 80.0)
                    current_page.insert_textbox(rect, full_sq_text, fontsize=10, fontname="helv")
                    cursor_y += 35.0

                    # Answer space lines
                    num_lines = max(2, min(sq_marks * 2, 10))
                    for _ in range(num_lines):
                        check_page_space(18.0)
                        current_page.draw_line(
                            pymupdf.Point(margin_left + 25, cursor_y),
                            pymupdf.Point(margin_right - 10, cursor_y),
                            width=0.5,
                            color=(0.7, 0.7, 0.7),
                        )
                        cursor_y += 16.0
                    cursor_y += 10.0

            else:
                q_marks = q.get("marks", 1)
                q_text = q.get("text", "")
                q_heading = f"Question {q_counter} ({q_marks} mark{'s' if q_marks > 1 else ''}) [{topic}]"
                current_page.insert_text(
                    pymupdf.Point(margin_left, cursor_y),
                    q_heading,
                    fontsize=12,
                    fontname="helv",
                )
                cursor_y += 18.0

                rect = pymupdf.Rect(margin_left, cursor_y, margin_right, cursor_y + 80.0)
                current_page.insert_textbox(rect, q_text, fontsize=10, fontname="helv")
                cursor_y += 35.0

                # Answer lines
                num_lines = max(2, min(q_marks * 2, 12))
                for _ in range(num_lines):
                    check_page_space(18.0)
                    current_page.draw_line(
                        pymupdf.Point(margin_left + 15, cursor_y),
                        pymupdf.Point(margin_right - 10, cursor_y),
                        width=0.5,
                        color=(0.7, 0.7, 0.7),
                    )
                    cursor_y += 16.0
                cursor_y += 12.0

            q_counter += 1

    doc.save(output_pdf_path)
    doc.close()
    logger.info("Successfully rendered past paper PDF to: %s", output_pdf_path)
    return output_pdf_path
