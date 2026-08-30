import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from load_and_store import Question, crop_question_screenshots
from exam_generator import render_exam_pdf


class TestDiagramIngestion(unittest.TestCase):
    def test_question_model(self):
        q = Question(
            question_content="Explain the process of photosynthesis.",
            q_type="basic_question",
            topic="Bioenergetics",
            subtopic="Photosynthesis",
            marks=4,
            exam="1June18",
            exam_type="Higher",
            image_paths=["data/Biology/Edexcel/question_images/sample_q1_p1.png"]
        )
        d = q.to_dict()
        self.assertEqual(d["question_content"], "Explain the process of photosynthesis.")
        self.assertEqual(d["marks"], 4)
        self.assertEqual(d["image_paths"], ["data/Biology/Edexcel/question_images/sample_q1_p1.png"])

    def test_crop_question_screenshots_valid_pdf(self):
        pdf_path = "data/Biology/Edexcel/Exam-Types/Higher/QuestionPapers/Biology-Edexcel-QuestionPaper-Higher-1June18.pdf"
        if not os.path.exists(pdf_path):
            self.skipTest("Sample PDF not found")
            
        raw_qs = [
            "1 (a) Photosynthesis occurs in chloroplasts...",
            "2 (a) Obesity increases the risk of heart disease..."
        ]
        results = crop_question_screenshots(pdf_path, "Biology", "Edexcel", raw_qs)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], list)

    def test_render_exam_pdf_creation(self):
        out_pdf = "test_outputs/test_render_exam_pdf.pdf"
        if os.path.exists(out_pdf):
            os.remove(out_pdf)

        sample_exam_output = {
            "questions": {
                "Bioenergetics": [
                    {
                        "number": "1)",
                        "text": "Describe the function of chlorophyll.",
                        "marks": 3,
                        "subtopic": "Photosynthesis",
                        "image_paths": []
                    }
                ]
            }
        }
        res_path = render_exam_pdf(sample_exam_output, out_pdf, subject="Biology", examiner="Edexcel")
        self.assertTrue(os.path.exists(res_path))
        self.assertGreater(os.path.getsize(res_path), 0)


if __name__ == '__main__':
    unittest.main()
