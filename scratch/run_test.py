import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_and_store import Question, crop_question_screenshots
from exam_generator import render_exam_pdf

print("Testing Question model...")
q = Question("Sample question text", "basic_question", "Biology", "Photosynthesis", 3, "1June18", "Higher", image_paths=["sample.png"])
print("Question dict:", q.to_dict())

print("Testing render_exam_pdf...")
test_pdf = "test_outputs/test_demo.pdf"
sample_exam = {
    "questions": {
        "Biology": [
            {
                "number": "1)",
                "text": "What is photosynthesis?",
                "marks": 2,
                "subtopic": "Plants",
                "image_paths": []
            }
        ]
    }
}
res = render_exam_pdf(sample_exam, test_pdf, "Biology", "Edexcel")
print("PDF output path:", res, "Exists:", os.path.exists(res), "Size:", os.path.getsize(res))
print("ALL TESTS COMPLETED SUCCESSFULLY!")
