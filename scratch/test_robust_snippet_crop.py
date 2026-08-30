import os
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pymupdf

def crop_question_screenshots_for_pdf(pdf_path: str, questions_raw_text: list[str], output_dir: str) -> list[list[str]]:
    if not os.path.exists(pdf_path):
        return [[] for _ in questions_raw_text]
        
    os.makedirs(output_dir, exist_ok=True)
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    doc = pymupdf.open(pdf_path)
    
    # 1. Locate start page and y0 for each question
    question_starts = [] # list of (start_page, y0)
    
    for idx, q_text in enumerate(questions_raw_text):
        clean_q = re.sub(r"\s+", " ", q_text).strip()
        snippet = clean_q[:50]
        found = False
        
        for p_idx in range(1, len(doc)): # skip cover page 0
            page = doc[p_idx]
            rects = page.search_for(snippet[:30])
            if rects:
                question_starts.append((p_idx, rects[0].y0))
                found = True
                break
            # Fallback: search for first 20 non-space chars
            if len(snippet) > 20:
                rects = page.search_for(snippet[:20])
                if rects:
                    question_starts.append((p_idx, rects[0].y0))
                    found = True
                    break
                    
        if not found:
            # Fallback: assign relative page estimation
            est_p = min(1 + idx, len(doc) - 1)
            question_starts.append((est_p, 90.0))
            
    print(f"Mapped {len(question_starts)} question start locations:")
    for i, (p, y) in enumerate(question_starts):
        print(f"  Q{i+1}: Page {p+1}, y0={y:.1f}")

    # 2. Compute bounding spans and render clip images
    all_image_paths = []
    
    for i in range(len(question_starts)):
        start_p, start_y = question_starts[i]
        if i + 1 < len(question_starts):
            next_p, next_y = question_starts[i+1]
        else:
            next_p = len(doc) - 1
            next_y = doc[next_p].rect.height - 30
            
        q_img_paths = []
        for p_idx in range(start_p, next_p + 1):
            page = doc[p_idx]
            p_h = page.rect.height
            p_w = page.rect.width
            
            y_top = max(0, start_y - 12) if p_idx == start_p else 30
            y_bot = min(p_h, next_y + 10) if p_idx == next_p else (p_h - 30)
            
            if y_bot > y_top + 15:
                rect = pymupdf.Rect(0, y_top, p_w, y_bot)
                pix = page.get_pixmap(clip=rect, matrix=pymupdf.Matrix(2.0, 2.0))
                img_name = f"{pdf_basename}_q{i+1}_p{p_idx+1}.png"
                img_path = os.path.normpath(os.path.join(output_dir, img_name))
                pix.save(img_path)
                q_img_paths.append(img_path)
                
        all_image_paths.append(q_img_paths)
        
    doc.close()
    return all_image_paths


if __name__ == "__main__":
    pdf_path = "data/Biology/Edexcel/Exam-Types/Higher/QuestionPapers/Biology-Edexcel-QuestionPaper-Higher-1June18.pdf"
    from load_and_store import PdfFile
    pdf_file = PdfFile(pdf_path, "Biology", "Edexcel", "QuestionPapers", "Higher")
    docs = pdf_file.load_pdf()
    full_text = pdf_file.pdf_to_text(docs)
    raw_qs = re.split(pdf_file.question_pattern, full_text)
    raw_qs = [q for q in raw_qs if q.strip()]
    
    img_paths_list = crop_question_screenshots_for_pdf(pdf_path, raw_qs, "test_outputs/question_images_robust")
    print(f"\nRendered images for {len(img_paths_list)} questions:")
    for i, paths in enumerate(img_paths_list):
        print(f"Q{i+1}: {paths}")
