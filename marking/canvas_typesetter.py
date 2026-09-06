"""Canvas typesetter module for GCSE AI.

Renders typed student text and question prompts into clean, modern
feedback pages optimized for teacher visual annotations (highlights,
ticks, underlines, and margin cards).
"""

import base64
import io
import logging
import os
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Attempt to load a clean TTF font, falling back to default."""
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "segoeui.ttf",
        "calibri.ttf",
        "DejaVuSans.ttf",
    ]
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, max_chars_per_line: int) -> list[str]:
    """Wrap text into lines given a character limit per line, preserving paragraphs."""
    paragraphs = text.splitlines()
    wrapped_lines = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            wrapped_lines.append("")
            continue
        words = para.split()
        curr: list[str] = []
        for w in words:
            curr.append(w)
            if len(" ".join(curr)) > max_chars_per_line:
                if len(curr) > 1:
                    wrapped_lines.append(" ".join(curr[:-1]))
                    curr = [w]
                else:
                    wrapped_lines.append(" ".join(curr))
                    curr = []
        if curr:
            wrapped_lines.append(" ".join(curr))
    return wrapped_lines


def typeset_student_answer_page(
    question_label: str,
    question_prompt: str,
    student_answer_text: str,
    marks: int = 4,
    body_width: int = 760,
    min_height: int = 650,
) -> Image.Image:
    """Typeset typed student text into a clean feedback page canvas.

    Args:
        question_label: Label e.g. 'Question 1(a)'
        question_prompt: Prompt text e.g. 'Describe how...'
        student_answer_text: Student's typed response text
        marks: Total marks available
        body_width: Pixel width of the question/answer area
        min_height: Minimum page height

    Returns:
        PIL Image of the clean feedback page.
    """
    font_title = _get_font(18)
    font_bold = _get_font(15)
    font_body = _get_font(14)
    font_small = _get_font(12)

    # Word wrap prompt and answer
    prompt_chars_per_line = 78
    answer_chars_per_line = 72

    prompt_lines = _wrap_text(question_prompt or "No question prompt provided.", prompt_chars_per_line)
    answer_lines = _wrap_text(student_answer_text or "No student answer provided.", answer_chars_per_line)

    # Calculate layout metrics
    pad_x = 35
    y_cursor = 30

    # Question Header
    y_cursor += 35

    # Prompt card
    prompt_line_height = 20
    prompt_box_h = max(40, len(prompt_lines) * prompt_line_height + 24)
    prompt_box_y = y_cursor
    y_cursor += prompt_box_h + 25

    # Student answer section
    ans_line_height = 28  # Generous line spacing for legible teacher annotations
    ans_start_y = y_cursor + 25
    ans_total_h = max(200, len(answer_lines) * ans_line_height + 40)
    total_page_h = max(min_height, ans_start_y + ans_total_h + 40)

    # Create canvas
    canvas = Image.new("RGB", (body_width, total_page_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 1. Header Bar: Question Label & Marks Badge
    draw.rectangle([(pad_x, 20), (pad_x + 130, 48)], fill=(30, 41, 59))
    draw.text((pad_x + 12, 24), question_label, fill=(255, 255, 255), font=font_bold)

    badge_text = f"[{marks} Marks]" if marks else ""
    draw.rectangle([(pad_x + 140, 20), (pad_x + 235, 48)], fill=(241, 245, 249), outline=(203, 213, 225), width=1)
    draw.text((pad_x + 152, 25), badge_text, fill=(71, 85, 105), font=font_bold)

    # 2. Question Prompt Card
    prompt_box_w = body_width - (pad_x * 2)
    draw.rectangle(
        [(pad_x, prompt_box_y), (pad_x + prompt_box_w, prompt_box_y + prompt_box_h)],
        fill=(248, 250, 252),
        outline=(226, 232, 240),
        width=1,
    )
    # Blue accent bar on left of prompt
    draw.line([(pad_x, prompt_box_y), (pad_x, prompt_box_y + prompt_box_h)], fill=(59, 130, 246), width=4)

    p_y = prompt_box_y + 12
    for p_line in prompt_lines:
        draw.text((pad_x + 16, p_y), p_line, fill=(30, 41, 59), font=font_body)
        p_y += prompt_line_height

    # 3. Student Answer Header
    draw.text((pad_x, ans_start_y - 22), "STUDENT ANSWER", fill=(100, 116, 139), font=font_small)

    # 4. Student Answer Lined Box
    ans_box_w = body_width - (pad_x * 2)
    ans_box_bottom = ans_start_y + ans_total_h
    draw.rectangle([(pad_x, ans_start_y), (pad_x + ans_box_w, ans_box_bottom)], fill=(255, 255, 255), outline=(226, 232, 240), width=1)

    # Draw faint guide lines
    for line_idx in range(len(answer_lines) + 2):
        guide_y = ans_start_y + 10 + (line_idx * ans_line_height) + 18
        if guide_y < ans_box_bottom:
            draw.line([(pad_x + 10, guide_y), (pad_x + ans_box_w - 10, guide_y)], fill=(245, 247, 250), width=1)

    # Draw student answer text
    text_y = ans_start_y + 10
    for a_line in answer_lines:
        draw.text((pad_x + 14, text_y), a_line, fill=(15, 23, 42), font=font_body)
        text_y += ans_line_height

    return canvas


def typeset_student_answer_to_b64(
    question_label: str,
    question_prompt: str,
    student_answer_text: str,
    marks: int = 4,
    body_width: int = 760,
    min_height: int = 650,
) -> str:
    """Typeset answer page and return as base64-encoded PNG string."""
    img = typeset_student_answer_page(
        question_label=question_label,
        question_prompt=question_prompt,
        student_answer_text=student_answer_text,
        marks=marks,
        body_width=body_width,
        min_height=min_height,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
