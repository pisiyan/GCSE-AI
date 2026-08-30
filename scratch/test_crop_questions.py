import os
import re
import pymupdf as fitz

pdf_path = "data/Biology/Edexcel/Exam-Types/Higher/QuestionPapers/Biology-Edexcel-QuestionPaper-Higher-1June18.pdf"
doc = fitz.open(pdf_path)

output_dir = "test_outputs/question_images_test"
os.makedirs(output_dir, exist_ok=True)

pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

# 1. Collect all block boundaries across pages (skipping page 0 cover)
question_spans = [] # list of dicts: {"q_num": int, "start_page": int, "start_y": float, "end_page": int, "end_y": float}

current_q = None

for page_idx in range(1, len(doc)):
    page = doc[page_idx]
    blocks = page.get_text("blocks")
    
    for b in blocks:
        x0, y0, x1, y1, text, b_num, b_type = b[:7]
        clean_text = text.strip()
        
        # Check for Question start: e.g. "1 (a)", "Question 1", "2 (a)"
        q_start_match = re.search(r"^(?:Question\s*(\d+)|(\d+)\s*\([a-z]\))", clean_text, re.IGNORECASE)
        # Check for Question end: e.g. "(Total for Question 1 = 6 marks)"
        q_end_match = re.search(r"\(Total\s+for\s+Question\s*(\d+)", clean_text, re.IGNORECASE)
        
        if q_start_match:
            q_num = int(q_start_match.group(1) or q_start_match.group(2))
            if current_q is None or current_q["q_num"] != q_num:
                if current_q and current_q["end_page"] is None:
                    # Close previous question at top of this block or end of previous page
                    current_q["end_page"] = page_idx
                    current_q["end_y"] = y0
                    question_spans.append(current_q)
                current_q = {
                    "q_num": q_num,
                    "start_page": page_idx,
                    "start_y": y0,
                    "end_page": None,
                    "end_y": None
                }
        elif q_end_match:
            end_q_num = int(q_end_match.group(1))
            if current_q and current_q["q_num"] == end_q_num:
                current_q["end_page"] = page_idx
                current_q["end_y"] = y1
                question_spans.append(current_q)
                current_q = None

if current_q:
    current_q["end_page"] = len(doc) - 1
    current_q["end_y"] = doc[-1].rect.height
    question_spans.append(current_q)

print(f"Detected {len(question_spans)} question spans:")
for q in question_spans:
    print(f"  Q{q['q_num']}: Page {q['start_page']+1} (y={q['start_y']:.1f}) -> Page {q['end_page']+1} (y={q['end_y']:.1f})")

# 2. Render clips for each question span
rendered_images = {}
for q in question_spans:
    q_num = q["q_num"]
    rendered_images[q_num] = []
    
    start_p = q["start_page"]
    end_p = q["end_page"]
    
    for p_idx in range(start_p, end_p + 1):
        page = doc[p_idx]
        page_h = page.rect.height
        page_w = page.rect.width
        
        y_top = max(0, q["start_y"] - 15) if p_idx == start_p else 30
        y_bot = min(page_h, q["end_y"] + 15) if p_idx == end_p else (page_h - 30)
        
        if y_bot > y_top + 10:
            rect = fitz.Rect(0, y_top, page_w, y_bot)
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2.0, 2.0))
            img_filename = f"{pdf_basename}_q{q_num}_p{p_idx+1}.png"
            img_path = os.path.join(output_dir, img_filename)
            pix.save(img_path)
            rendered_images[q_num].append(img_path)
            print(f"    Saved: {img_path}")

doc.close()
