"""Visual annotation renderer for exam answer scripts.

Renders teacher-style visual feedback directly onto student answer images:
- Translucent colored highlights (green, amber).
- Red underlinings and error cross marks.
- Teacher tick marks (e.g. [✓ M1]) and score badges.
- Level band descriptor badges (e.g. [Level 2 - 4/6 Marks]).
- Right-hand examiner feedback margin cards with connector lines.
"""

import base64
import io
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Attempt to load a clean TTF font, falling back to default."""
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "seguiemj.ttf",
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


def render_visual_annotations(
    image_input: Union[str, Image.Image, bytes],
    annotations_data: Dict[str, Any],
    output_path: Optional[str] = None,
    add_margin_gutter: bool = True,
    gutter_width: int = 340,
) -> Image.Image:
    """Render teacher annotations onto student answer image.

    Args:
        image_input: File path, base64 string, bytes, or PIL Image.
        annotations_data: Dict containing 'annotations', 'awarded_marks', 'total_marks', etc.
        output_path: Optional path to save rendered image (PNG/JPG).
        add_margin_gutter: If True, adds an examiner margin panel on the right for clean teacher notes.
        gutter_width: Pixel width of examiner margin panel.

    Returns:
        Annotated PIL Image.
    """
    # Load base image
    if isinstance(image_input, Image.Image):
        base_img = image_input.convert("RGBA")
    elif isinstance(image_input, bytes):
        base_img = Image.open(io.BytesIO(image_input)).convert("RGBA")
    elif isinstance(image_input, str):
        if os.path.exists(image_input):
            base_img = Image.open(image_input).convert("RGBA")
        else:
            # Assume base64 string
            clean_b64 = image_input.split(",")[-1] if "," in image_input else image_input
            img_bytes = base64.b64decode(clean_b64)
            base_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    else:
        raise ValueError("Unsupported image_input type.")

    orig_w, orig_h = base_img.size

    # Canvas dimensions
    canvas_w = orig_w + (gutter_width if add_margin_gutter else 0)
    canvas_h = orig_h

    # Create composite canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    canvas.paste(base_img, (0, 0))

    # Translucent overlay layer for highlights
    overlay = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Main draw layer for crisp text, underlines, and icons
    draw = ImageDraw.Draw(canvas)

    # Fonts
    font_large = _get_font(20)
    font_mid = _get_font(15)
    font_small = _get_font(12)
    font_badge = _get_font(13)

    # Color Palette (Teacher Pen Colors)
    c_green_hl = (46, 204, 113, 85)       # Soft translucent green highlight
    c_amber_hl = (243, 156, 18, 85)       # Soft translucent amber highlight
    c_tick = (39, 174, 96, 255)           # Teacher green ink
    c_red_pen = (214, 48, 49, 255)        # Teacher red ink
    c_purple = (142, 68, 173, 255)        # Examiner guidance purple
    c_blue = (41, 128, 185, 255)          # Level badge blue
    c_card_bg = (248, 249, 250, 240)      # Gutter card background
    c_card_border = (200, 205, 210, 255)  # Card border

    # Draw Examiner Gutter Separator Line
    if add_margin_gutter:
        draw.line([(orig_w, 0), (orig_w, canvas_h)], fill=(220, 224, 230, 255), width=2)
        # Margin Header
        draw.rectangle([(orig_w, 0), (canvas_w, 35)], fill=(240, 243, 246, 255))
        draw.text((orig_w + 15, 8), "EXAMINER FEEDBACK & MARKS", fill=(100, 110, 120, 255), font=font_small)

    annotations = annotations_data.get("annotations", [])
    total_marks = annotations_data.get("total_marks", 4)
    awarded_marks = annotations_data.get("awarded_marks", 0)
    level_band = annotations_data.get("level_band")

    # Draw Score Banner in Top Right
    score_text = f"Mark: [{awarded_marks}/{total_marks}]"
    if add_margin_gutter:
        draw.rectangle([(canvas_w - 180, 42), (canvas_w - 15, 78)], fill=(230, 245, 235, 255), outline=c_tick, width=2)
        draw.text((canvas_w - 165, 48), score_text, fill=c_tick, font=font_large)

    # Track gutter vertical positions to prevent overlapping margin notes
    gutter_y_cursor = 88

    for ann in annotations:
        tool = ann.get("tool", "green_highlight")
        box = ann.get("box_2d", [0, 0, 100, 100])
        code = ann.get("mark_code") or ""
        comment = ann.get("teacher_comment") or ""

        # Map normalized 0..1000 to pixel dimensions
        ymin = int((box[0] / 1000.0) * orig_h)
        xmin = int((box[1] / 1000.0) * orig_w)
        ymax = int((box[2] / 1000.0) * orig_h)
        xmax = int((box[3] / 1000.0) * orig_w)

        # Pad bounding box slightly
        box_pad = 3
        b_ymin = max(0, ymin - box_pad)
        b_xmin = max(0, xmin - box_pad)
        b_ymax = min(orig_h, ymax + box_pad)
        b_xmax = min(orig_w, xmax + box_pad)

        if tool == "green_highlight":
            overlay_draw.rectangle([b_xmin, b_ymin, b_xmax, b_ymax], fill=c_green_hl)
            if code:
                # Small badge on highlight
                draw.rectangle([b_xmin, b_ymin - 16, b_xmin + 45, b_ymin], fill=c_tick)
                draw.text((b_xmin + 4, b_ymin - 15), str(code), fill=(255, 255, 255, 255), font=font_small)

        elif tool == "amber_highlight":
            overlay_draw.rectangle([b_xmin, b_ymin, b_xmax, b_ymax], fill=c_amber_hl)

        elif tool == "tick_mark":
            # Draw checkmark icon near word
            tick_x = min(orig_w - 40, b_xmax + 6)
            tick_y = b_ymin
            tick_label = f"✓ {code}".strip() if code else "✓"
            draw.rectangle([tick_x, tick_y - 2, tick_x + 52, tick_y + 18], fill=(235, 247, 238, 240), outline=c_tick, width=1)
            draw.text((tick_x + 6, tick_y), tick_label, fill=c_tick, font=font_badge)

        elif tool == "red_underline":
            # Draw wavy or solid red underline under text
            line_y = min(orig_h - 2, b_ymax + 1)
            draw.line([(b_xmin, line_y), (b_xmax, line_y)], fill=c_red_pen, width=2)
            draw.line([(b_xmin, line_y + 2), (b_xmax, line_y + 2)], fill=(250, 150, 150, 180), width=1)
            if code == "REJECT":
                draw.rectangle([b_xmax + 4, b_ymin, b_xmax + 54, b_ymin + 16], fill=c_red_pen)
                draw.text((b_xmax + 7, b_ymin + 1), "REJECT", fill=(255, 255, 255, 255), font=font_small)

        elif tool == "cross_mark":
            cross_x = min(orig_w - 30, b_xmax + 6)
            cross_y = b_ymin
            draw.text((cross_x, cross_y), "✗", fill=c_red_pen, font=font_large)

        elif tool == "level_descriptor_badge":
            # Draw level badge box around section
            overlay_draw.rectangle([b_xmin, b_ymin, b_xmax, b_ymax], outline=c_blue, width=2)
            lbl = f"[{code or 'Level Band'}]"
            draw.rectangle([b_xmin, b_ymin - 20, b_xmin + 110, b_ymin], fill=c_blue)
            draw.text((b_xmin + 6, b_ymin - 18), lbl, fill=(255, 255, 255, 255), font=font_small)

        # Margin Callout Card
        if add_margin_gutter and comment:
            card_x = orig_w + 15
            card_w = gutter_width - 30
            card_y = max(gutter_y_cursor, b_ymin - 10)

            # Determine card theme
            if tool in ("red_underline", "cross_mark"):
                border_col = c_red_pen
                header_text = f"✗ {code or 'Correction'}"
            elif tool in ("tick_mark", "green_highlight"):
                border_col = c_tick
                header_text = f"✓ {code or 'Mark Awarded'}"
            elif tool == "level_descriptor_badge":
                border_col = c_blue
                header_text = f"⭐ {code or 'Level Feedback'}"
            else:
                border_col = c_purple
                header_text = f"💡 {code or 'Examiner Tip'}"

            # Wrap comment text into lines
            max_chars_per_line = 34
            words = comment.split()
            lines = []
            curr = []
            for w in words:
                curr.append(w)
                if len(" ".join(curr)) > max_chars_per_line:
                    lines.append(" ".join(curr[:-1]))
                    curr = [w]
            if curr:
                lines.append(" ".join(curr))

            card_h = 28 + (len(lines) * 16) + 8

            # Draw connector line from script box to margin card
            draw.line([(b_xmax, b_ymin + 10), (card_x, card_y + 12)], fill=(180, 190, 200, 200), width=1)
            draw.ellipse([(b_xmax - 2, b_ymin + 8), (b_xmax + 4, b_ymin + 14)], fill=border_col)

            # Draw card
            draw.rectangle([(card_x, card_y), (card_x + card_w, card_y + card_h)], fill=c_card_bg, outline=border_col, width=1)
            draw.rectangle([(card_x, card_y), (card_x + card_w, card_y + 20)], fill=(240, 242, 245, 255))
            draw.text((card_x + 6, card_y + 3), header_text, fill=border_col, font=font_small)

            for l_idx, line_str in enumerate(lines):
                draw.text((card_x + 8, card_y + 24 + (l_idx * 16)), line_str, fill=(40, 45, 50, 255), font=font_small)

            gutter_y_cursor = card_y + card_h + 10

    # Composite translucent highlights over base canvas
    canvas = Image.alpha_composite(canvas, overlay)

    # Save output if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Convert to RGB before saving JPEG/PNG
        final_img = canvas.convert("RGB")
        final_img.save(output_path)
        logger.info("Saved annotated visual marking image to %s", output_path)

    return canvas
