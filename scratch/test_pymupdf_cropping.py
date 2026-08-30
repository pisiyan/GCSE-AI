import os
import re
import pymupdf as fitz

pdf_path = "data/Biology/Edexcel/Exam-Types/Higher/QuestionPapers/Biology-Edexcel-QuestionPaper-Higher-1June18.pdf"
if not os.path.exists(pdf_path):
    print("PDF not found:", pdf_path)
    exit(1)

doc = fitz.open(pdf_path)
print(f"Opened PDF: {pdf_path}, Total Pages: {len(doc)}")

# Iterate through pages (skipping cover page 0)
for page_num in range(1, len(doc)):
    page = doc[page_num]
    blocks = page.get_text("blocks")
    print(f"\n--- Page {page_num + 1} ({len(blocks)} blocks) ---")
    for b in blocks:
        # b is (x0, y0, x1, y1, text, block_no, block_type)
        x0, y0, x1, y1, text, b_num, b_type = b[:7]
        clean_t = text.strip().replace("\n", " ")
        if len(clean_t) > 60:
            clean_t = clean_t[:60] + "..."
        if "Question" in clean_t or re.search(r"^\d+\s*\(", clean_t) or re.search(r"^\d+\.", clean_t):
            print(f"  [Q Start Block {b_num}] y0={y0:.1f}, y1={y1:.1f} | {clean_t}")

doc.close()
