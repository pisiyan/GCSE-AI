"""Exam generation package for GCSE AI."""

from exam_generation.structure import ExamStructureBuilder
from exam_generation.generator import QuestionGenerator
from exam_generation.pdf_exporter import render_exam_pdf
from exam_generation.formatter import (
    generate_answer_lines,
    fix_ocr_formatting,
    clean_question_text,
    format_single_question_as_markdown,
    format_exam_as_markdown,
)

__all__ = [
    "ExamStructureBuilder",
    "QuestionGenerator",
    "render_exam_pdf",
    "generate_answer_lines",
    "fix_ocr_formatting",
    "clean_question_text",
    "format_single_question_as_markdown",
    "format_exam_as_markdown",
]
