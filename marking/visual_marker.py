"""Visual exam script marking and spatial annotation orchestrator.

Implements the 3-stage visual marking pipeline:
- Step 1: Pre-marking policy generation from mark scheme (Text LLM, blind to student answer).
- Step 2: Diagnostic student handwritten script evaluation (Multimodal Vision LLM).
- Step 3: Spatial coordinate grounding & annotation synthesis (Multimodal Vision LLM).
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from llm.client import LLMClient
from llm.prompts import load_prompt
from llm.tracker import global_tracker

logger = logging.getLogger(__name__)


class VisualExamMarker:
    """Orchestrates 3-step visual marking and teacher annotations for handwritten answers."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        subject: str = "General",
        examiner: str = "AQA",
    ) -> None:
        self.llm = llm_client or LLMClient()
        self.subject = subject
        self.examiner = examiner

    def generate_marking_policy(
        self, question: str, marks: int, mark_scheme: str
    ) -> Dict[str, Any]:
        """Step 1: Formulate marking tool instructions and rubric blind to student answer."""
        logger.info("Step 1: Generating marking tool policy for %d marks...", marks)
        prompt = load_prompt(
            "generate_marking_policy",
            subject=self.subject,
            examiner=self.examiner,
            marks=marks,
            question=question,
            mark_scheme=mark_scheme,
        )
        try:
            policy = self.llm.invoke_json(prompt)
            if not isinstance(policy, dict):
                policy = {"marking_rules": [], "raw": str(policy)}
            return policy
        except Exception as e:
            logger.warning("Step 1 policy generation JSON parse failed, falling back: %s", e)
            return {
                "marking_rules": [
                    {
                        "mark_code": f"M{i+1}",
                        "description": f"Mark point {i+1}",
                        "tool_to_use": "green_highlight + tick_mark",
                        "keywords_to_look_for": [],
                        "reject_rules": {"disallowed_terms": [], "tool_to_use": "red_underline"},
                        "if_missing_tool": "margin_callout",
                    }
                    for i in range(marks)
                ],
                "leveled_rules": {
                    "applicable_if_leveled": marks >= 6,
                    "level_3_criteria": "Tool: green_highlight on coherent scientific mechanisms",
                    "level_2_criteria": "Tool: amber_highlight on isolated facts",
                    "level_1_criteria": "Tool: amber_highlight on brief points",
                },
                "teacher_guidance": "Follow official mark scheme strictly.",
            }

    def evaluate_student_script(
        self, image_b64: str, question: str, marks: int, mark_scheme: str
    ) -> str:
        """Step 2: Inspect handwritten image against mark scheme to produce diagnostic evaluation."""
        logger.info("Step 2: Performing multimodal diagnostic script evaluation...")
        prompt = load_prompt(
            "evaluate_handwritten_script",
            subject=self.subject,
            examiner=self.examiner,
            marks=marks,
            question=question,
            mark_scheme=mark_scheme,
        )
        evaluation_text = self.llm.invoke_with_image(prompt, image_b64)
        logger.info("Step 2 completed: %s", evaluation_text[:120].replace("\n", " "))
        return evaluation_text

    def synthesize_visual_annotations(
        self,
        image_b64: str,
        policy: Dict[str, Any],
        evaluation_text: str,
        marks: int,
    ) -> Dict[str, Any]:
        """Step 3: Ground evaluation into spatial 2D coordinates and map tools according to policy."""
        logger.info("Step 3: Grounding visual annotations onto script image...")
        policy_str = json.dumps(policy, indent=2)
        prompt = load_prompt(
            "ground_annotations",
            marks=marks,
            policy_json=policy_str,
            evaluation_text=evaluation_text,
        )
        try:
            annotations_data = self.llm.invoke_with_image_json(prompt, image_b64)
            if not isinstance(annotations_data, dict):
                annotations_data = {"annotations": [], "raw": str(annotations_data)}
        except Exception as e:
            logger.warning("Step 3 annotation grounding JSON parse failed: %s", e)
            annotations_data = {
                "question_type": "point_based",
                "total_marks": marks,
                "awarded_marks": 0,
                "annotations": [],
                "teacher_summary": "Marking evaluation completed.",
            }

        # Normalize annotations and coordinates
        annotations_data = self._validate_and_sanitize_annotations(annotations_data, marks)
        return annotations_data

    def _validate_and_sanitize_annotations(
        self, data: Dict[str, Any], marks: int
    ) -> Dict[str, Any]:
        """Validate and clamp annotation coordinates and tool values."""
        if not isinstance(data.get("annotations"), list):
            data["annotations"] = []

        valid_tools = {
            "green_highlight",
            "tick_mark",
            "red_underline",
            "cross_mark",
            "amber_highlight",
            "margin_callout",
            "level_descriptor_badge",
        }

        sanitized_annotations = []
        for ann in data["annotations"]:
            if not isinstance(ann, dict):
                continue
            tool = ann.get("tool", "green_highlight")
            if tool not in valid_tools:
                tool = "green_highlight"

            box = ann.get("box_2d", [0, 0, 100, 100])
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(isinstance(v, (int, float)) for v in box)
            ):
                box = [100, 100, 200, 500]

            # Clamp box to 0..1000
            ymin = max(0, min(1000, int(box[0])))
            xmin = max(0, min(1000, int(box[1])))
            ymax = max(ymin, min(1000, int(box[2])))
            xmax = max(xmin, min(1000, int(box[3])))

            sanitized_annotations.append({
                "tool": tool,
                "box_2d": [ymin, xmin, ymax, xmax],
                "text_snippet": str(ann.get("text_snippet", "")),
                "mark_code": ann.get("mark_code"),
                "teacher_comment": str(ann.get("teacher_comment", "")),
            })

        data["annotations"] = sanitized_annotations
        data["total_marks"] = marks

        # Score sanity checking
        if "awarded_marks" not in data or not isinstance(data["awarded_marks"], (int, float)):
            # Count ticks or extract from evaluation
            data["awarded_marks"] = sum(
                1 for a in sanitized_annotations if a["tool"] in ("tick_mark", "green_highlight")
            )
        data["awarded_marks"] = max(0, min(marks, int(data["awarded_marks"])))

        return data

    def mark_handwritten_answer(
        self,
        image_b64: str,
        question: str,
        marks: int,
        mark_scheme: str,
        output_json_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run complete 3-stage visual marking pipeline."""
        session_name = (
            f"Visual Marking: {self.subject} ({self.examiner}) - Q ({marks} Marks)"
        )
        global_tracker.start_session(session_name)
        try:
            # Step 1: Pre-marking policy
            policy = self.generate_marking_policy(question, marks, mark_scheme)

            # Step 2: Multimodal diagnostic evaluation
            evaluation = self.evaluate_student_script(image_b64, question, marks, mark_scheme)

            # Step 3: Spatial visual grounding
            annotations_data = self.synthesize_visual_annotations(
                image_b64=image_b64,
                policy=policy,
                evaluation_text=evaluation,
                marks=marks,
            )

            result = {
                "subject": self.subject,
                "examiner": self.examiner,
                "question": question,
                "marks": marks,
                "mark_scheme": mark_scheme,
                "policy": policy,
                "evaluation": evaluation,
                "annotations_data": annotations_data,
                "awarded_marks": annotations_data.get("awarded_marks", 0),
                "question_type": annotations_data.get("question_type", "point_based"),
                "level_band": annotations_data.get("level_band"),
                "annotations": annotations_data.get("annotations", []),
                "teacher_summary": annotations_data.get("teacher_summary", ""),
            }

            if output_json_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                logger.info("Saved visual marking annotations JSON to %s", output_json_path)

            return result
        finally:
            global_tracker.end_session()
