"""Exam marking package for GCSE AI."""

from marking.marker import ExamMarker
from marking.segmenter import ExamSegmenter
from marking.report import (
    format_marking_as_markdown,
    generate_performance_summary,
    save_marking_report_md,
)
from marking.visual_marker import VisualExamMarker
from marking.visual_renderer import render_visual_annotations
from marking.canvas_typesetter import typeset_student_answer_page, typeset_student_answer_to_b64
from marking.booklet import assemble_marked_booklet_pdf, render_summary_cover_image

__all__ = [
    "ExamMarker",
    "ExamSegmenter",
    "format_marking_as_markdown",
    "generate_performance_summary",
    "save_marking_report_md",
    "VisualExamMarker",
    "render_visual_annotations",
    "typeset_student_answer_page",
    "typeset_student_answer_to_b64",
    "assemble_marked_booklet_pdf",
    "render_summary_cover_image",
]


