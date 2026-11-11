import os
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant import GcseAssistant
from ingestion.spec_summary import build_spec_summary


class TestSpecLoading(unittest.TestCase):
    @patch('assistant.load_subject_config')
    @patch('assistant.HuggingFaceEmbeddings')
    @patch('assistant.load_vectorstore')
    @patch('assistant.GcseAssistant._load_from_vectorstore')
    def setUp(self, mock_load_vdb, mock_faiss, mock_hf, mock_config):
        mock_load_vdb.return_value = ([], [])
        
        with patch('assistant.GcseAssistant._load_full_specification', return_value="Dummy Spec"):
            self.assistant = GcseAssistant("Physics", "Edexcel")

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open, read_data="Cached Spec Text")
    def test_load_full_specification_from_cache(self, mock_file, mock_listdir, mock_exists):
        mock_exists.side_effect = lambda path: True
        mock_listdir.return_value = ["Physics-Edexcel-Specification.pdf"]

        spec_text = self.assistant._load_full_specification()

        self.assertEqual(spec_text, "Cached Spec Text")
        mock_file.assert_called_with(
            os.path.normpath("data/Physics/Edexcel/Specification/Physics-Edexcel-Specification.txt"),
            "r",
            encoding="utf-8"
        )

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('pypdf.PdfReader')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_full_specification_from_pdf(self, mock_file, mock_pdf_reader_cls, mock_listdir, mock_exists):
        mock_exists.side_effect = lambda path: path.endswith("Specification") or path.endswith(".pdf")
        mock_listdir.return_value = ["Physics-Edexcel-Specification.pdf"]

        mock_pdf_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Extracted Page Content"
        mock_pdf_reader.pages = [mock_page]
        mock_pdf_reader_cls.return_value = mock_pdf_reader

        spec_text = self.assistant._load_full_specification()

        self.assertEqual(spec_text, "Extracted Page Content")
        mock_file.assert_called_with(
            os.path.normpath("data/Physics/Edexcel/Specification/Physics-Edexcel-Specification.txt"),
            "w",
            encoding="utf-8"
        )
        mock_file().write.assert_called_once_with("Extracted Page Content")


class TestBuildSpecSummary(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_build_spec_summary_single_llm_call(self, mock_makedirs, mock_file):
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke_json.return_value = {
            "subject": "Physics",
            "examiner": "Edexcel",
            "exam_types": [
                {
                    "exam_type": "Paper 1",
                    "topics": [
                        {
                            "topic": "Topic 1 Motion and Forces",
                            "spec_code": "Topic 1 (1.1 - 1.10)",
                            "subtopics": ["Speed", "Velocity"]
                        }
                    ]
                }
            ]
        }

        spec_text = "Topic 1 Motion and Forces\n- Speed\n- Velocity"
        res = build_spec_summary("Physics", "Edexcel", spec_text, ["Paper 1"], llm_client=mock_llm_instance)

        self.assertEqual(mock_llm_instance.invoke_json.call_count, 1)
        self.assertEqual(res["subject"], "Physics")
        self.assertEqual(res["examiner"], "Edexcel")
        self.assertEqual(len(res["exam_types"]), 1)
        self.assertEqual(res["exam_types"][0]["topics"][0]["topic"], "Topic 1 Motion and Forces")


if __name__ == '__main__':
    unittest.main()
