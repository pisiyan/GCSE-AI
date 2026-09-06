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

__all__ = [
    "ExamMarker",
    "ExamSegmenter",
    "format_marking_as_markdown",
    "generate_performance_summary",
    "save_marking_report_md",
    "VisualExamMarker",
    "render_visual_annotations",
]

