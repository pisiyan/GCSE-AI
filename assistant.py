"""GCSE AI Assistant — High-level orchestrator facade.

Wires together configuration, LLM utilities, exam generation,
marking, topic testing, and curriculum guidance modules.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import SubjectConfig, load_subject_config
from llm.client import LLMClient
from llm.similarity import SimilarityEngine
from exam_generation.structure import ExamStructureBuilder
from exam_generation.generator import QuestionGenerator
from exam_generation.pdf_exporter import render_exam_pdf
from marking.marker import ExamMarker
from content.topic_tester import TopicTester
from content.revision import generate_revision_materials
from content.guidance import explain_command_word, get_spec_breakdown
from ingestion.load_and_store import load_vectorstore

logger = logging.getLogger(__name__)

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"), override=True)


class GcseAssistant:
    """Main GCSE AI assistant facade coordinating all system modules."""

    def __init__(self, subject: str, examiner: str) -> None:
        logger.info("Initializing GCSE Assistant for %s (%s)...", subject, examiner)

        self.subject = subject
        self.examiner = examiner
        self.config: SubjectConfig = load_subject_config(subject, examiner)

        # Shared embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load FAISS vector store
        self.vectorstore = load_vectorstore(subject, examiner, self.embedding_model)

        # Shared LLM client & similarity engine
        self.llm_client = LLMClient()
        self.similarity = SimilarityEngine(self.llm_client, self.embedding_model)

        # Load question and mark scheme items from docstore
        self.questions, self.mark_schemes = self._load_from_vectorstore()

        # Load full specification text
        self.specification_text = self._load_full_specification()

        # Module sub-systems
        self.structure_builder = ExamStructureBuilder(
            config=self.config,
            questions=self.questions,
        )
        self.question_generator = QuestionGenerator(
            config=self.config,
            llm_client=self.llm_client,
            similarity_engine=self.similarity,
            questions=self.questions,
            embedding_model=self.embedding_model,
            specification_text=self.specification_text,
        )
        self.exam_marker = ExamMarker(
            config=self.config,
            llm_client=self.llm_client,
            similarity_engine=self.similarity,
            questions=self.questions,
            mark_schemes=self.mark_schemes,
            subject=subject,
            examiner=examiner,
            embedding_model=self.embedding_model,
            specification_text=self.specification_text,
        )
        self.topic_tester = TopicTester(
            config=self.config,
            llm_client=self.llm_client,
            specification_text=self.specification_text,
        )

        logger.info("GCSE Assistant initialization complete for %s (%s)", subject, examiner)

    def _load_from_vectorstore(self) -> Tuple[List[dict], List[dict]]:
        """Load questions and mark schemes directly from FAISS database docstore."""
        questions = []
        mark_schemes = []
        all_docs = list(self.vectorstore.docstore._dict.values())
        logger.info("Loaded %d total documents from vector database", len(all_docs))

        for doc in all_docs:
            meta = doc.metadata.copy()
            doc_type = meta.get("type", "")

            if doc_type == "QuestionPaper":
                q_dict = meta.copy()
                q_dict["question_content"] = doc.page_content
                if "q_type" in q_dict:
                    q_dict["type"] = q_dict.pop("q_type")

                # De-serialize nested JSON fields
                for json_field in ("parent_question_structure", "sub_questions", "image_paths"):
                    if json_field in q_dict and isinstance(q_dict[json_field], str):
                        try:
                            q_dict[json_field] = json.loads(q_dict[json_field])
                        except json.JSONDecodeError:
                            pass
                questions.append(q_dict)

            elif doc_type == "MarkScheme" or meta.get("doc_class") == "mark_scheme":
                ms_dict = meta.copy()
                ms_dict["question_content"] = doc.page_content
                mark_schemes.append(ms_dict)

        logger.info("Extracted %d questions and %d mark schemes", len(questions), len(mark_schemes))
        return questions, mark_schemes

    def _load_full_specification(self) -> str:
        """Find specification PDF, extract text using pypdf if not cached, and return it."""
        spec_dir = f"data/{self.subject}/{self.examiner}/Specification"
        if not os.path.exists(spec_dir):
            root_dir = os.path.dirname(os.path.abspath(__file__))
            spec_dir = os.path.join(root_dir, "data", self.subject, self.examiner, "Specification")

        if not os.path.exists(spec_dir):
            logger.warning("Specification directory not found under %s", spec_dir)
            return "No specification details found."

        pdf_file = None
        for filename in os.listdir(spec_dir):
            if filename.lower().endswith(".pdf"):
                pdf_file = os.path.normpath(os.path.join(spec_dir, filename))
                break

        if not pdf_file:
            logger.warning("No specification PDF found in %s", spec_dir)
            return "No specification details found."

        txt_file = os.path.normpath(pdf_file.rsplit(".", 1)[0] + ".txt")
        if os.path.exists(txt_file):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    logger.info("Loaded specification text from cache: %s", txt_file)
                    return f.read()
            except Exception as e:
                logger.error("Failed to read cached spec text: %s", e)

        logger.info("Extracting text from specification PDF: %s", pdf_file)
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_file)
            text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
            full_text = "\n\n".join(text_parts)
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            logger.info("Saved specification text cache to: %s", txt_file)
            return full_text
        except Exception as e:
            logger.error("Failed to extract text from specification PDF %s: %s", pdf_file, e)
            return "No specification details found."

    # --- Delegated module operations ---

    @staticmethod
    def get_valid_exam_types(subject: str, examiner: str) -> List[str]:
        """Discover valid exam types for a given subject and examiner."""
        root_dir = os.path.dirname(os.path.abspath(__file__))
        for base in ("data", os.path.join(root_dir, "data")):
            exam_types_dir = os.path.join(base, subject, examiner, "Exam-Types")
            if os.path.isdir(exam_types_dir):
                return sorted(
                    name for name in os.listdir(exam_types_dir)
                    if os.path.isdir(os.path.join(exam_types_dir, name))
                )
        return []

    def make_exam(
        self,
        exam_type: str,
        total_marks: int,
        topics: List[str],
        user_preferences: Optional[Any] = None,
        num_questions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a complete GCSE exam paper."""
        return self.question_generator.generate_exam(
            exam_type=exam_type,
            total_marks=total_marks,
            topics=topics,
            subject=self.subject,
            structure_builder=self.structure_builder,
            user_preferences=user_preferences,
            num_questions=num_questions,
        )

    def export_exam_pdf(self, exam_output: Dict[str, Any], output_pdf_path: str) -> str:
        """Export a generated exam dict into a PDF paper."""
        return render_exam_pdf(
            exam_output=exam_output,
            output_pdf_path=output_pdf_path,
            subject=self.subject,
            examiner=self.examiner,
        )

    def mark_question(self) -> str:
        """Mark question(s) from documents placed in user_data/."""
        return self.exam_marker.mark_question_from_files()

    def mark_exam(self, questions: List[dict], topic: str) -> List[dict]:
        """Mark a full list of questions."""
        return self.exam_marker.mark_exam(questions, topic)

    def revision_materials(self, topic: str) -> str:
        """Generate revision materials for a topic."""
        return generate_revision_materials(
            topic=topic,
            subject=self.subject,
            examiner=self.examiner,
            llm_client=self.llm_client,
            specification_text=self.specification_text,
        )

    def create_topic_test_session(self, desired_area: str) -> Tuple[dict, List[dict]]:
        """Start a topic content test session."""
        scope = self.topic_tester.compile_target_subtopics(desired_area, self.subject, self.examiner)
        qa_list = self.topic_tester.generate_questions_and_answers(scope, self.subject, self.examiner)
        return scope, qa_list

    def evaluate_topic_test_answer(
        self, question: str, perfect_answer: str, student_answer: str, subtopic: str = ""
    ) -> dict:
        """Evaluate a student's answer against the exemplar answer."""
        return self.topic_tester.evaluate_student_answer(question, perfect_answer, student_answer, subtopic)

    def save_topic_test_report(
        self, desired_area: str, results: List[dict], output_dir: str = "test_outputs"
    ) -> str:
        """Save performance report for a topic test session."""
        return self.topic_tester.generate_performance_report(
            self.subject, self.examiner, desired_area, results, output_dir
        )

    def generate_single_question(
        self,
        topic: str,
        marks: int = 4,
        subtopic: str = "",
        q_type: str = "basic",
        exam_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a single standalone or parent question."""
        et = exam_type or self.examiner
        mark_struct = [marks, q_type] if q_type == "parent" else marks
        task = {
            "mark_structure": mark_struct,
            "exam_type": et,
            "subtopic": subtopic or topic,
            "subject": self.subject,
            "number": 1
        }
        return self.question_generator._execute_generation_task(task)

    def generate_mark_scheme(self, question: str, marks: int = 4, topic: str = "General") -> str:
        """Create a mark scheme for a single question."""
        return self.exam_marker.create_mark_scheme(question=question, marks=marks, topic=topic)

    def generate_model_answer(
        self, question: str, mark_scheme: str = "", marks: int = 4, topic: str = "General"
    ) -> str:
        """Generate a full-mark exemplar model answer for a question."""
        if not mark_scheme or not mark_scheme.strip():
            mark_scheme = self.generate_mark_scheme(question=question, marks=marks, topic=topic)
        return self.exam_marker.generate_model_answer(question=question, mark_scheme=mark_scheme)

    def get_spec_breakdown(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve syllabus breakdown."""
        return get_spec_breakdown(self.subject, self.examiner, topic)

    def explain_command_word(self, command_word: str = "", question: str = "") -> str:
        """Explain GCSE command word requirements."""
        return explain_command_word(
            command_word=command_word,
            subject=self.subject,
            examiner=self.examiner,
            llm_client=self.llm_client,
            question=question,
        )
