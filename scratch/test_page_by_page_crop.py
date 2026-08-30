import os
import sys
import re
import pymupdf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pdf_path = "data/Biology/Edexcel/Exam-Types/Higher/QuestionPapers/Biology-Edexcel-QuestionPaper-Higher-1June18.pdf"
doc = pymupdf.open(pdf_path)

output_dir = "test_outputs/question_images_page_by_page"
os.makedirs(output_dir, exist_ok=True)
pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

print(f"Loaded PDF: {pdf_path} ({len(doc)} pages)")

# Track question images: q_num -> list of image paths
question_images = {}

current_q_num = None

for page_idx in range(1, len(doc)): # skip cover page 0
    page = doc[page_idx]
    page_num = page_idx + 1
    blocks = page.get_text("blocks")
    
    # Identify question headers on this page
    # Each item: (q_num, start_block_idx, y0)
    q_headers_on_page = []
    
    for b_idx, b in enumerate(blocks):
        x0, y0, x1, y1, text, b_num, b_type = b[:7]
        clean_text = text.strip()
        
        # Match question start e.g. "Question 1", "1 (a)", "2 (a)", "1."
        q_start = re.search(r"^(?:Question\s*(\d+)|(\d+)\s*\([a-z]\)|(\d+)\.)", clean_text, re.IGNORECASE)
        if q_start:
            q_num = int(q_start.group(1) or q_start.group(2) or q_start.group(3))
            q_headers_on_page.append((q_num, b_idx, y0))
            
    print(f"\n--- Page {page_num} ({len(blocks)} blocks) ---")
    if q_headers_on_page:
        print(f"  Question headers found: {q_headers_on_page}")
    else:
        print(f"  Continuing question: Q{current_q_num}")

    # Now calculate bounding boxes for each question segment on this page
    if not q_headers_on_page:
        # Whole page belongs to current_q_num (if active)
        if current_q_num is not None:
            # Crop page content area
            y_top = 30.0
            # Check for Total marks at page bottom
            y_bot = page.rect.height - 30.0
            for b in blocks:
                if "Total for Question" in b[4]:
                    y_bot = min(page.rect.height, b[3] + 10.0)
                    break
            
            rect = pymupdf.Rect(0, y_top, page.rect.width, y_bot)
            pix = page.get_pixmap(clip=rect, matrix=pymupdf.Matrix(2.0, 2.0))
            img_name = f"{pdf_basename}_q{current_q_num}_p{page_num}.png"
            img_path = os.path.join(output_dir, img_name)
            pix.save(img_path)
            question_images.setdefault(current_q_num, []).append(img_path)
            print(f"    Rendered continuation for Q{current_q_num}: {img_name}")
    else:
        # There are 1 or more question starts on this page
        for h_idx, (q_num, b_idx, q_y0) in enumerate(q_headers_on_page):
            current_q_num = q_num
            y_top = max(0.0, q_y0 - 10.0)
            
            # y_bottom is start of next question header on this page, or page footer
            if h_idx + 1 < len(q_headers_on_page):
                next_y0 = q_headers_on_page[h_idx + 1][2]
                y_bot = max(y_top + 20.0, next_y0 - 8.0)
            else:
                y_bot = page.rect.height - 30.0
                for b in blocks[b_idx:]:
                    if "Total for Question" in b[4]:
                        y_bot = min(page.rect.height, b[3] + 10.0)
                        break
                        
            if y_bot > y_top + 15.0:
                rect = pymupdf.Rect(0, y_top, page.rect.width, y_bot)
                pix = page.get_pixmap(clip=rect, matrix=pymupdf.Matrix(2.0, 2.0))
                img_name = f"{pdf_basename}_q{q_num}_p{page_num}.png"
                img_path = os.path.join(output_dir, img_name)
                pix.save(img_path)
                question_images.setdefault(q_num, []).append(img_path)
                print(f"    Rendered Q{q_num} (header): {img_name} [y_top={y_top:.1f}, y_bot={y_bot:.1f}]")

doc.close()
print("\nFinal Rendered Images per Question:")
for q_n, img_list in sorted(question_images.items()):
    print(f"Question {q_n}: {img_list}")
