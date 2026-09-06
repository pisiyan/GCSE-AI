"""Unit tests for 3-stage visual marking and teacher annotation pipeline."""

import base64
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from marking.visual_marker import VisualExamMarker
from marking.visual_renderer import render_visual_annotations


class TestVisualMarking(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.marker = VisualExamMarker(
            llm_client=self.mock_llm,
            subject="Biology",
            examiner="Edexcel",
        )

        # Create a small base64 test image
        img = Image.new("RGB", (300, 200), color=(240, 240, 240))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.test_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    def test_generate_marking_policy_success(self):
        """Step 1 returns parsed marking policy JSON."""
        expected_policy = {
            "marking_rules": [
                {
                    "mark_code": "M1",
                    "description": "Mentions cell membrane",
                    "tool_to_use": "green_highlight + tick_mark",
                    "keywords_to_look_for": ["cell membrane", "partially permeable"],
                    "reject_rules": {"disallowed_terms": ["cell wall"], "tool_to_use": "red_underline"},
                    "if_missing_tool": "margin_callout",
                }
            ],
            "leveled_rules": {
                "applicable_if_leveled": False,
                "level_3_criteria": "Coherent multi-step scientific mechanisms",
            },
            "teacher_guidance": "Animal cells lack cell walls.",
        }
        self.mock_llm.invoke_json.return_value = expected_policy

        policy = self.marker.generate_marking_policy(
            question="Describe the structure of an animal cell.",
            marks=2,
            mark_scheme="1 mark for cell membrane. Do not accept cell wall.",
        )
        self.assertEqual(policy, expected_policy)
        self.mock_llm.invoke_json.assert_called_once()

    def test_generate_marking_policy_fallback_on_error(self):
        """Step 1 provides graceful fallback policy if LLM fails JSON parsing."""
        self.mock_llm.invoke_json.side_effect = ValueError("Malformed JSON")

        policy = self.marker.generate_marking_policy(
            question="Calculate velocity.", marks=3, mark_scheme="v = s / t"
        )
        self.assertIn("marking_rules", policy)
        self.assertEqual(len(policy["marking_rules"]), 3)
        self.assertEqual(policy["marking_rules"][0]["mark_code"], "M1")

    def test_evaluate_student_script_step2(self):
        """Step 2 sends image and mark scheme to multimodal vision model."""
        self.mock_llm.invoke_with_image.return_value = (
            "1. Transcription: The student wrote cell membrane.\n"
            "2. Marks awarded: M1 awarded for cell membrane.\n"
            "Score: [1/2]"
        )

        res = self.marker.evaluate_student_script(
            image_b64=self.test_img_b64,
            question="Describe animal cell",
            marks=2,
            mark_scheme="1 mark for cell membrane",
        )
        self.assertIn("Score: [1/2]", res)
        self.mock_llm.invoke_with_image.assert_called_once()

    def test_synthesize_visual_annotations_step3_point_based(self):
        """Step 3 grounds coordinates and clamps boxes to 0..1000."""
        self.mock_llm.invoke_with_image_json.return_value = {
            "question_type": "point_based",
            "total_marks": 2,
            "awarded_marks": 1,
            "annotations": [
                {
                    "tool": "green_highlight",
                    "box_2d": [100, 50, 160, 400],
                    "text_snippet": "cell membrane",
                    "mark_code": "M1",
                    "teacher_comment": "✓ Correctly identified cell membrane",
                },
                {
                    "tool": "red_underline",
                    "box_2d": [-20, 420, 250, 1200],  # Out of range, should be clamped
                    "text_snippet": "cell wall",
                    "mark_code": "REJECT",
                    "teacher_comment": "✗ Animal cells do not have cell walls",
                },
            ],
            "teacher_summary": "Good effort.",
        }

        data = self.marker.synthesize_visual_annotations(
            image_b64=self.test_img_b64,
            policy={"marking_rules": []},
            evaluation_text="Evaluated notes",
            marks=2,
        )

        self.assertEqual(data["question_type"], "point_based")
        self.assertEqual(data["awarded_marks"], 1)
        self.assertEqual(len(data["annotations"]), 2)

        # Verify coordinate clamping
        clamped_box = data["annotations"][1]["box_2d"]
        self.assertEqual(clamped_box[0], 0)     # -20 clamped to 0
        self.assertEqual(clamped_box[3], 1000)  # 1200 clamped to 1000

    def test_synthesize_visual_annotations_step3_leveled(self):
        """Step 3 handles leveled question format with level band justification."""
        self.mock_llm.invoke_with_image_json.return_value = {
            "question_type": "leveled",
            "total_marks": 6,
            "awarded_marks": 4,
            "level_band": {
                "level": 2,
                "range": [3, 4],
                "rationale": "Clear scientific reasoning for step 1, lacking mechanism for step 2.",
            },
            "annotations": [
                {
                    "tool": "level_descriptor_badge",
                    "box_2d": [50, 50, 500, 900],
                    "text_snippet": "Full paragraph",
                    "mark_code": "LEVEL_2",
                    "teacher_comment": "Level 2: Clear discussion of collision theory",
                }
            ],
            "teacher_summary": "Solid Level 2 answer.",
        }

        data = self.marker.synthesize_visual_annotations(
            image_b64=self.test_img_b64,
            policy={"leveled_rules": {}},
            evaluation_text="Leveled evaluation",
            marks=6,
        )

        self.assertEqual(data["question_type"], "leveled")
        self.assertEqual(data["awarded_marks"], 4)
        self.assertIsNotNone(data["level_band"])
        self.assertEqual(data["level_band"]["level"], 2)

    def test_render_visual_annotations(self):
        """Visual renderer applies highlights, ticks, underlines, and margin gutter."""
        base_img = Image.new("RGB", (400, 300), color=(255, 255, 255))
        annotations_data = {
            "question_type": "point_based",
            "total_marks": 3,
            "awarded_marks": 2,
            "annotations": [
                {
                    "tool": "green_highlight",
                    "box_2d": [100, 100, 150, 300],
                    "mark_code": "M1",
                    "teacher_comment": "Awarded M1",
                },
                {
                    "tool": "tick_mark",
                    "box_2d": [100, 310, 150, 350],
                    "mark_code": "✓ M1",
                    "teacher_comment": "+1 mark",
                },
                {
                    "tool": "red_underline",
                    "box_2d": [200, 100, 240, 300],
                    "mark_code": "REJECT",
                    "teacher_comment": "Reject this term",
                },
                {
                    "tool": "margin_callout",
                    "box_2d": [260, 100, 290, 300],
                    "mark_code": "M3_MISSING",
                    "teacher_comment": "Remember to state that reaction rate increases.",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "marked_script.png")
            annotated_img = render_visual_annotations(
                image_input=base_img,
                annotations_data=annotations_data,
                output_path=out_file,
                add_margin_gutter=True,
                gutter_width=300,
            )

            self.assertTrue(os.path.exists(out_file))
            self.assertEqual(annotated_img.size, (400 + 300, 300))
            # Verify file has non-zero bytes
            self.assertGreater(os.path.getsize(out_file), 500)

    def test_full_pipeline_end_to_end(self):
        """End-to-end execution of 3-stage visual marking with JSON export."""
        self.mock_llm.invoke_json.return_value = {
            "marking_rules": [{"mark_code": "M1", "tool_to_use": "green_highlight"}]
        }
        self.mock_llm.invoke_with_image.return_value = "Diagnostic: Good response. Score: [1/1]"
        self.mock_llm.invoke_with_image_json.return_value = {
            "question_type": "point_based",
            "total_marks": 1,
            "awarded_marks": 1,
            "annotations": [
                {
                    "tool": "green_highlight",
                    "box_2d": [100, 100, 200, 300],
                    "mark_code": "M1",
                    "teacher_comment": "Spot on",
                }
            ],
            "teacher_summary": "Full marks achieved!",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = os.path.join(tmp_dir, "annotations.json")
            result = self.marker.mark_handwritten_answer(
                image_b64=self.test_img_b64,
                question="Define speed.",
                marks=1,
                mark_scheme="Distance divided by time",
                output_json_path=out_json,
            )

            self.assertEqual(result["awarded_marks"], 1)
            self.assertEqual(result["question_type"], "point_based")
            self.assertTrue(os.path.exists(out_json))


if __name__ == "__main__":
    unittest.main()
