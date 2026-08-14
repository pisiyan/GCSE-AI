import os
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generate_content import GcseAssistant

class TestSpecLoading(unittest.TestCase):
    @patch('generate_content.load_subject_config')
    @patch('generate_content.HuggingFaceEmbeddings')
    @patch('generate_content.FAISS.load_local')
    @patch('generate_content.RetrievalQA.from_chain_type')
    @patch('generate_content.GcseAssistant._load_from_vectorstore')
    @patch('generate_content.GcseAssistant._load_templates')
    def setUp(self, mock_load_templates, mock_load_vdb, mock_qa, mock_faiss, mock_hf, mock_config):
        # Prevent any VDB or template loads during setup
        mock_load_templates.return_value = {}
        mock_load_vdb.return_value = ([], [])
        
        with patch('generate_content.GcseAssistant._load_full_specification', return_value="Dummy Spec"):
            self.assistant = GcseAssistant("Physics", "Edexcel")

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open, read_data="Cached Spec Text")
    def test_load_full_specification_from_cache(self, mock_file, mock_listdir, mock_exists):
        # Setup paths to mock existences
        # 1. Spec dir exists
        # 2. PDF file exists
        # 3. TXT file exists
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
        # 1. Spec dir exists
        # 2. PDF file exists
        # 3. TXT cache does NOT exist (must trigger extraction)
        mock_exists.side_effect = lambda path: path.endswith("Specification") or path.endswith(".pdf")
        mock_listdir.return_value = ["Physics-Edexcel-Specification.pdf"]

        # Mock PDF reader page extraction
        mock_pdf_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Extracted Page Content"
        mock_pdf_reader.pages = [mock_page]
        mock_pdf_reader_cls.return_value = mock_pdf_reader

        spec_text = self.assistant._load_full_specification()

        self.assertEqual(spec_text, "Extracted Page Content")
        # Verify it writes to cache
        mock_file.assert_called_with(
            os.path.normpath("data/Physics/Edexcel/Specification/Physics-Edexcel-Specification.txt"),
            "w",
            encoding="utf-8"
        )
        mock_file().write.assert_called_once_with("Extracted Page Content")

class TestBuildSpecSummary(unittest.TestCase):
    @patch('build_spec_summary.get_specification_text')
    @patch('build_spec_summary.discover_exam_types')
    @patch('build_spec_summary.LLMClient')
    @patch('build_spec_summary.load_subject_config')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_generate_specification_summary_single_llm_call(
        self, mock_makedirs, mock_file, mock_config, mock_llm_cls, mock_discover_exam_types, mock_get_spec_text
    ):
        from build_spec_summary import generate_specification_summary

        mock_get_spec_text.return_value = "Topic 1 Motion and Forces\n- Speed\n- Velocity"
        mock_discover_exam_types.return_value = ["Paper 1"]

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke_json.return_value = {
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
        mock_llm_cls.return_value = mock_llm_instance

        res = generate_specification_summary("Physics", "Edexcel")

        # Verify only 1 LLM call was invoked for all topics/exam types
        self.assertEqual(mock_llm_instance.invoke_json.call_count, 1)
        self.assertEqual(res["subject"], "Physics")
        self.assertEqual(res["examiner"], "Edexcel")
        self.assertEqual(len(res["exam_types"]), 1)
        self.assertEqual(res["exam_types"][0]["topics"][0]["topic"], "Topic 1 Motion and Forces")
        self.assertEqual(res["exam_types"][0]["topics"][0]["subtopics"], ["Speed", "Velocity"])

    @patch('os.path.exists', return_value=True)
    @patch('build_spec_summary.LLMClient')
    def test_generate_specification_summary_skip_if_exists(self, mock_llm_cls, mock_exists):
        from build_spec_summary import generate_specification_summary
        import json

        dummy_data = {"subject": "Physics", "examiner": "Edexcel", "exam_types": []}
        with patch('builtins.open', mock_open(read_data=json.dumps(dummy_data))):
            res = generate_specification_summary("Physics", "Edexcel", skip_if_exists=True)

        # Verify LLM was NOT invoked because file exists
        mock_llm_cls.assert_not_called()
        self.assertEqual(res, dummy_data)


if __name__ == '__main__':
    unittest.main()

