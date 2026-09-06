"""Marked Exam Booklet PDF Assembly Module.

Stitches marked question page images and an executive pedagogical summary
cover page into a unified, high-quality marked_exam_booklet.pdf using PyMuPDF.
"""

import io
import logging
import os
from typing import Any, Dict, List, Optional

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from marking.canvas_typesetter import _get_font, _wrap_text

logger = logging.getLogger(__name__)


def render_summary_cover_image(
    subject: str,
    examiner: str,
    topic: str,
    total_available: int,
    total_awarded: int,
    question_results: List[Dict[str, Any]],
    executive_summary_text: str = "",
    width: int = 1100,
    min_height: int = 1400,
) -> Image.Image:
    """Render a clean, high-impact pedagogical summary cover page as a PIL image."""
    font_large = _get_font(26)
    font_subtitle = _get_font(18)
    font_section = _get_font(16)
    font_body = _get_font(13)
    font_bold = _get_font(13)
    font_table = _get_font(12)

    pad_x = 50
    pct = (total_awarded / total_available * 100) if total_available > 0 else 0.0

    if pct >= 80:
        grade = "Grade 9 / Exceptional Performance 🌟"
        grade_color = (39, 174, 96)
    elif pct >= 70:
        grade = "Grade 8 / Strong Performance 🚀"
        grade_color = (39, 174, 96)
    elif pct >= 60:
        grade = "Grade 7 / Good Performance 👍"
        grade_color = (41, 128, 185)
    elif pct >= 50:
        grade = "Grade 6 / Solid Pass ✅"
        grade_color = (41, 128, 185)
    elif pct >= 40:
        grade = "Grade 4-5 / Standard Pass 📈"
        grade_color = (211, 84, 0)
    else:
        grade = "Grade 1-3 / Revision Required ⚠️"
        grade_color = (192, 57, 43)

    # Estimate height based on summary lines and table
    summary_lines = _wrap_text(executive_summary_text, 115) if executive_summary_text else []
    table_h = len(question_results) * 26 + 60
    summary_h = max(250, len(summary_lines) * 20 + 60)
    total_h = max(min_height, 220 + summary_h + table_h + 80)

    canvas = Image.new("RGB", (width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Header Card
    draw.rectangle([(0, 0), (width, 130)], fill=(15, 23, 42))
    draw.text((pad_x, 28), f"GCSE {subject} ({examiner}) — Feedback Report", fill=(255, 255, 255), font=font_large)
    draw.text((pad_x, 72), f"Topic: {topic}  |  Detailed Annotations Booklet", fill=(148, 163, 184), font=font_subtitle)

    # Score Banner
    score_y = 150
    banner_w = width - (pad_x * 2)
    draw.rectangle([(pad_x, score_y), (pad_x + banner_w, score_y + 80)], fill=(248, 250, 252), outline=(226, 232, 240), width=1)
    # Accent strip
    draw.rectangle([(pad_x, score_y), (pad_x + 6, score_y + 80)], fill=grade_color)

    score_str = f"Final Score: {total_awarded} / {total_available} Marks ({pct:.1f}%)"
    draw.text((pad_x + 25, score_y + 18), score_str, fill=(15, 23, 42), font=font_section)
    draw.text((pad_x + 25, score_y + 48), f"Estimated Standard: {grade}", fill=grade_color, font=font_bold)

    y_cursor = score_y + 105

    # Executive Diagnostic Summary Section
    if executive_summary_text:
        draw.text((pad_x, y_cursor), "Diagnostic Examiner Analysis", fill=(30, 41, 59), font=font_section)
        y_cursor += 28

        # Draw summary card
        card_w = width - (pad_x * 2)
        card_h = len(summary_lines) * 20 + 24
        draw.rectangle([(pad_x, y_cursor), (pad_x + card_w, y_cursor + card_h)], fill=(255, 255, 255), outline=(226, 232, 240), width=1)

        sy = y_cursor + 12
        for line in summary_lines:
            if line.startswith("### "):
                # Section heading inside summary
                draw.text((pad_x + 16, sy), line.replace("### ", ""), fill=(15, 23, 42), font=font_bold)
                sy += 22
            else:
                draw.text((pad_x + 20, sy), line, fill=(51, 65, 85), font=font_body)
                sy += 20
        y_cursor += card_h + 30

    # Question Breakdown Table
    draw.text((pad_x, y_cursor), "Question-by-Question Mark Breakdown", fill=(30, 41, 59), font=font_section)
    y_cursor += 28

    table_w = width - (pad_x * 2)
    # Table Header
    draw.rectangle([(pad_x, y_cursor), (pad_x + table_w, y_cursor + 30)], fill=(241, 245, 249))
    draw.text((pad_x + 16, y_cursor + 7), "Question", fill=(71, 85, 105), font=font_bold)
    draw.text((pad_x + 220, y_cursor + 7), "Marks Available", fill=(71, 85, 105), font=font_bold)
    draw.text((pad_x + 390, y_cursor + 7), "Marks Awarded", fill=(71, 85, 105), font=font_bold)
    draw.text((pad_x + 560, y_cursor + 7), "Status / Type", fill=(71, 85, 105), font=font_bold)
    y_cursor += 30

    # Rows
    for idx, q_res in enumerate(question_results):
        row_bg = (255, 255, 255) if idx % 2 == 0 else (248, 250, 252)
        draw.rectangle([(pad_x, y_cursor), (pad_x + table_w, y_cursor + 26)], fill=row_bg, outline=(241, 245, 249), width=1)

        lbl = q_res.get("label") or f"Question {idx+1}"
        m_avail = str(q_res.get("marks", 0))
        m_award = str(q_res.get("awarded_marks", 0))
        q_type = q_res.get("question_type") or ("Leveled" if int(q_res.get("marks", 0) or 0) >= 6 else "Discrete")

        draw.text((pad_x + 16, y_cursor + 5), lbl, fill=(15, 23, 42), font=font_table)
        draw.text((pad_x + 240, y_cursor + 5), m_avail, fill=(71, 85, 105), font=font_table)

        # Highlight awarded mark in green/red
        award_col = (39, 174, 96) if m_award == m_avail else ((214, 48, 49) if m_award == "0" else (211, 84, 0))
        draw.text((pad_x + 410, y_cursor + 5), m_award, fill=award_col, font=font_bold)
        draw.text((pad_x + 560, y_cursor + 5), str(q_type).capitalize(), fill=(100, 116, 139), font=font_table)
        y_cursor += 26

    # Bottom notice
    draw.text((pad_x, y_cursor + 20), "👉 Turn to subsequent pages to view detailed teacher pen annotations on each question.", fill=(100, 116, 139), font=font_body)

    return canvas


def assemble_marked_booklet_pdf(
    subject: str,
    examiner: str,
    topic: str,
    marked_image_paths: List[str],
    question_results: List[Dict[str, Any]],
    executive_summary_text: str = "",
    output_pdf_path: str = "test_outputs/marked_exam_booklet.pdf",
) -> str:
    """Stitch marked question pages and an executive summary page into a PDF booklet."""
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    doc = pymupdf.open()

    total_avail = sum(int(q.get("marks", 0) or 0) for q in question_results)
    total_award = sum(int(q.get("awarded_marks", 0) or 0) for q in question_results)

    # 1. Generate & Insert Executive Summary Cover Page
    cover_img = render_summary_cover_image(
        subject=subject,
        examiner=examiner,
        topic=topic,
        total_available=total_avail,
        total_awarded=total_award,
        question_results=question_results,
        executive_summary_text=executive_summary_text,
    )
    cover_buf = io.BytesIO()
    cover_img.save(cover_buf, format="PNG")

    cover_w, cover_h = cover_img.size
    p_cover = doc.new_page(width=cover_w, height=cover_h)
    p_cover.insert_image(p_cover.rect, stream=cover_buf.getvalue())

    # 2. Insert each marked question page image
    for img_path in marked_image_paths:
        if not os.path.exists(img_path):
            continue
        try:
            with Image.open(img_path) as q_img:
                qw, qh = q_img.size
            p_q = doc.new_page(width=qw, height=qh)
            p_q.insert_image(p_q.rect, filename=img_path)
        except Exception as e:
            logger.warning("Failed to insert question page %s into PDF: %s", img_path, e)

    doc.save(output_pdf_path)
    doc.close()
    logger.info("Successfully compiled marked exam booklet PDF to %s", output_pdf_path)
    return output_pdf_path
