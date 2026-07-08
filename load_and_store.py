"""PDF loading, parsing, and vector store management for GCSE AI.

Handles loading exam papers and specifications from PDF files,
extracting question structures, and storing them in FAISS vector databases.
"""

import json
import logging
import os
import re
from typing import Optional

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from config import SubjectConfig, load_subject_config

logger = logging.getLogger(__name__)

# Map folder/type names to config prefix keys
DOC_TYPE_KEYS = {
    "specification": "spec",
    "markschemes": "ms",
    "questionpapers": "spec",
}


class PdfFile:
    """Represents a PDF file for processing and question extraction.

    Handles parsing of exam papers, mark schemes, and specifications,
    extracting question structures based on configurable regex patterns.
    """

    def __init__(
        self, name: str, subject: str, examiner: str, doc_type: str
    ) -> None:
        """Initialize a PdfFile.

        Args:
            name: Path to the PDF file.
            subject: Subject name (e.g., "Biology").
            examiner: Exam board name (e.g., "Edexcel").
            doc_type: Document type ("Specification", "MarkSchemes", or "QuestionPapers").
        """
        self.subject = subject
        self.examiner = examiner
        self.config = load_subject_config(subject, examiner)
        self.name = name
        self.meta_data = self._get_metadata()
        self.info = self.meta_data

        config_prefix = DOC_TYPE_KEYS[doc_type.lower()]
        self.splitter = CharacterTextSplitter(
            chunk_size=getattr(self.config, f"{config_prefix}_chunk_size"),
            chunk_overlap=getattr(self.config, f"{config_prefix}_chunk_overlap"),
            separator="",
        )
        self.marks_pattern = self.config.mark_pattern
        self.letter_pattern = self.config.letter_pattern
        self.roman_pattern = self.config.roman_pattern
        self.question_pattern = self.config.question_pattern

    def first_marker_type(self, text: str) -> Optional[str]:
        """Determine whether letters or roman numerals appear first in text.

        Args:
            text: The text to search for markers.

        Returns:
            "letter", "roman", or None if neither is found.
        """
        if self.letter_pattern and self.letter_pattern != "None":
            letter_match = re.search(self.letter_pattern, text)
        else:
            letter_match = None

        if self.roman_pattern and self.roman_pattern != "None":
            roman_match = re.search(self.roman_pattern, text)
        else:
            roman_match = None

        letter_pos = letter_match.start() if letter_match else None
        roman_pos = roman_match.start() if roman_match else None

        if letter_pos is None and roman_pos is None:
            return None
        elif letter_pos is None:
            return "roman"
        elif roman_pos is None:
            return "letter"
        else:
            return "letter" if letter_pos < roman_pos else "roman"

    def extract_mark(self, text: str, pattern: str) -> Optional[int]:
        """Extract a mark value from text using a regex pattern.

        Finds the last match in the text and extracts the numeric value.

        Args:
            text: The text to search.
            pattern: Regex pattern with a capture group for the mark number.

        Returns:
            The extracted mark as an integer, or None if not found.
        """
        regex = re.compile(pattern)
        matches = list(regex.finditer(text))
        if matches:
            last_match = matches[-1]
            if last_match.groups():
                return int(last_match.group(1))
            return int(last_match.group())
        return None

    def add_metadata(self, chunks: list[Document]) -> list[Document]:
        """Add metadata to document chunks.

        Args:
            chunks: List of Document objects to annotate.

        Returns:
            The same chunks with metadata added.
        """
        for doc in chunks:
            doc.metadata.update(self.meta_data)
        return chunks

    def load_pdf(self) -> list[Document]:
        """Load a PDF file and return its pages as Documents.

        Returns:
            List of Document objects, one per page.
        """
        loader = PyPDFLoader(self.name)
        document = loader.load()
        logger.info("Loaded PDF: %s", self.name)
        return document

    def pdf_to_text(self, pages: list[Document]) -> str:
        """Convert PDF pages to a single text string.

        Skips the first page (typically a cover page with exam board info).

        Args:
            pages: List of Document objects from the PDF.

        Returns:
            Concatenated text from all pages except the first.
        """
        # Skip first page (cover page with exam board info/instructions)
        return " ".join(pages[i + 1].page_content for i in range(len(pages) - 1))

    @staticmethod
    def flatten(lst: list) -> list:
        """Flatten a list one level deep.

        Args:
            lst: A potentially nested list.

        Returns:
            A flattened list.
        """
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    def is_parent_question_valid(
        self, parent_question: dict, questions: list[dict]
    ) -> bool:
        """Check if a parent question's structure matches its child questions.

        Args:
            parent_question: The parent question dict.
            questions: All questions from the same exam.

        Returns:
            True if the structure matches the child question marks.
        """
        structure = self.flatten(parent_question["parent_question_structure"])
        child_marks = [
            q["marks"]
            for q in questions
            if (
                q["marks"] is not None
                and q["parent_question_description"]
                == parent_question["parent_question_description"]
            )
        ]
        return structure == child_marks

    def extract_question_documents(self, content: str) -> list[Document]:
        """Extract questions from exam paper content and return as Document chunks.

        Parses the content into individual questions, validates parent questions,
        and constructs Document objects.

        Args:
            content: The full text content of the exam paper.

        Returns:
            List of chunked Document objects.
        """
        topic = self.meta_data["topic"]
        exam = self.meta_data["time"]
        questions = re.split(self.question_pattern, content)
        question_info = self.process_questions(questions, topic, exam)

        docs = []
        for question in question_info:
            parent_valid = False
            if question["type"] == "parent_question":
                parent_valid = self.is_parent_question_valid(question, question_info)
            if question["marks"] is not None or parent_valid:
                content_str = question.pop("question_content") or ""
                meta = question.copy()
                meta["q_type"] = meta.pop("type")
                if meta.get("parent_question_structure"):
                    meta["parent_question_structure"] = json.dumps(meta["parent_question_structure"])
                docs.append(Document(page_content=str(content_str), metadata=meta))

        logger.info("Extracted %d question chunks", len(docs))
        return docs

    def extract_ms_documents(self, content: str) -> list[Document]:
        """Extract mark scheme questions from content and return as Document chunks.

        Args:
            content: The full text content of the mark scheme.

        Returns:
            List of chunked Document objects.
        """
        ms_pattern = self.config.ms_pattern
        ms_mark_pattern = self.config.ms_mark_pattern
        mark_schemes = re.split(ms_pattern, content)

        docs = []
        for mark_scheme in mark_schemes:
            marks = self.extract_mark(mark_scheme, ms_mark_pattern)
            meta = {
                "marks": marks,
                "doc_class": "mark_scheme"
            }
            docs.append(Document(page_content=str(mark_scheme), metadata=meta))

        logger.info("Extracted %d mark scheme chunks", len(docs))
        return docs

    def split_document(
        self, document: list[Document]
    ) -> Optional[list[Document]]:
        """Split a document into chunks for vector storage.

        For question papers and mark schemes, extracts and returns structured question documents.
        For specifications, splits into overlapping text chunks.

        Args:
            document: List of Document pages from a PDF.

        Returns:
            List of chunked Documents.
        """
        doc_type = self.meta_data.get("type", "").lower()

        if doc_type == "questionpaper":
            full_text = self.pdf_to_text(document)
            return self.extract_question_documents(full_text)
        elif doc_type == "markscheme":
            full_text = self.pdf_to_text(document)
            return self.extract_ms_documents(full_text)
        else:
            merged_text = "\n".join(str(doc.page_content) for doc in document)
            merged_doc = Document(page_content=merged_text)
            chunks = self.splitter.split_documents([merged_doc])
            logger.info("Split document into %d chunks", len(chunks))
            return chunks

    def _get_metadata(self) -> dict[str, str]:
        """Extract metadata from the PDF filename.

        Expects filenames in format: subject-examiner-type-topic-time.pdf

        Returns:
            Dict with keys: subject, examiner, type, topic, time.
        """
        keys = ["subject", "examiner", "type", "topic", "time"]
        basename = os.path.basename(self.name)
        basename = basename.replace(".PDF", "").replace(".pdf", "")
        details = basename.split("-")

        meta_data = {}
        for i, value in enumerate(details):
            if i < len(keys):
                meta_data[keys[i]] = value

        logger.debug("Extracted metadata: %s", meta_data)
        return meta_data

    def process_questions(
        self, questions: list[str], topic: str, exam: str
    ) -> list[dict]:
        """Process raw question text splits into structured question dicts.

        Handles three levels of question nesting:
        - basic_question: standalone questions
        - parent_question: questions with letter-labeled sub-parts
        - grandchild_question: sub-parts with roman numeral sub-sub-parts

        Args:
            questions: List of raw question text strings.
            topic: The exam topic.
            exam: The exam identifier (time period).

        Returns:
            List of structured question dictionaries.
        """
        questions_info: list[dict] = []

        for question in questions:
            first_marker = self.first_marker_type(question)

            if first_marker is not None:
                # This question has sub-parts
                if first_marker == "letter":
                    sub_questions = re.split(self.letter_pattern, question)
                else:
                    sub_questions = re.split(self.roman_pattern, question)

                parent_description = sub_questions.pop(0)
                question_structure: list = []

                for sub_question in sub_questions:
                    sub_marker = self.first_marker_type(sub_question)

                    if sub_marker is None:
                        # Simple sub-question
                        marks = self.extract_mark(sub_question, self.marks_pattern)
                        question_structure.append(marks)
                        questions_info.append(
                            {
                                "type": "child_question",
                                "topic": topic,
                                "marks": marks,
                                "question_content": sub_question,
                                "parent_question_structure": None,
                                "parent_question_description": parent_description,
                            }
                        )
                    else:
                        # Sub-question with its own sub-parts (grandchildren)
                        if sub_marker == "letter":
                            sub_sub_questions = re.split(
                                self.letter_pattern, sub_question
                            )
                        else:
                            sub_sub_questions = re.split(
                                self.roman_pattern, sub_question
                            )

                        parent_child_description = sub_sub_questions.pop(0)
                        sub_structure: list = []

                        for sub_sub_question in sub_sub_questions:
                            marks = self.extract_mark(
                                sub_sub_question, self.marks_pattern
                            )
                            sub_structure.append(marks)
                            questions_info.append(
                                {
                                    "type": "grandchild_question",
                                    "topic": topic,
                                    "marks": marks,
                                    "question_content": sub_sub_question,
                                    "parent_question_structure": None,
                                    "parent_question_description": parent_description,
                                }
                            )

                        questions_info.append(
                            {
                                "type": "parent_child_question",
                                "topic": topic,
                                "marks": None,
                                "question_content": None,
                                "parent_question_structure": sub_structure,
                                "parent_question_description": parent_description,
                            }
                        )
                        question_structure.append(sub_structure)

                questions_info.append(
                    {
                        "type": "parent_question",
                        "topic": topic,
                        "marks": None,
                        "question_content": question,
                        "parent_question_structure": question_structure,
                        "parent_question_description": parent_description,
                    }
                )
            else:
                # Basic standalone question
                questions_info.append(
                    {
                        "type": "basic_question",
                        "topic": topic,
                        "marks": self.extract_mark(question, self.marks_pattern),
                        "question_content": question,
                        "parent_question_structure": None,
                        "parent_question_description": None,
                    }
                )

        # Add exam identifier to all questions
        for q_info in questions_info:
            q_info["exam"] = exam

        return questions_info


class VectorStore:
    """Manages FAISS vector store creation and updates."""

    def __init__(self, name: str) -> None:
        """Initialize a VectorStore.

        Args:
            name: Path/name for the vector database.
        """
        self.vector_database_name = name
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def create_new_database(self, chunks: list[Document]) -> None:
        """Create a new FAISS vector database from document chunks.

        Args:
            chunks: List of Document objects to embed and store.
        """
        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        vectorstore.save_local(self.vector_database_name)
        logger.info(
            "Created new vector database: %s", self.vector_database_name
        )

    def add_to_existing_database(self, chunks: list[Document]) -> None:
        """Add document chunks to an existing FAISS vector database.

        Args:
            chunks: List of Document objects to embed and add.
        """
        db = FAISS.load_local(
            self.vector_database_name,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        db.add_documents(chunks)
        db.save_local(self.vector_database_name)
        logger.info(
            "Added %d chunks to %s", len(chunks), self.vector_database_name
        )


class DatabaseManager:
    """Orchestrates PDF processing and vector database management."""

    def __init__(self, subject: str, examiner: str) -> None:
        self.subject = subject
        self.examiner = examiner

    def store_to_database(self, pdf: PdfFile, database: VectorStore) -> None:
        """Load a PDF, split it, and store chunks in the vector database.

        Args:
            pdf: A PdfFile instance to process.
            database: A VectorStore instance to store chunks in.
        """
        document = pdf.load_pdf()
        chunks = pdf.split_document(document)

        if chunks is not None:
            chunks = pdf.add_metadata(chunks)
            try:
                database.add_to_existing_database(chunks)
            except RuntimeError:
                logger.info("Database not found, creating new one")
                database.create_new_database(chunks)

    def add_folder_database(self, folder: str, database_path: str) -> None:
        """Process all PDFs in a folder and add them to a vector database.

        Args:
            folder: Path to the folder containing PDF files.
            database_path: Path for the FAISS vector database.
        """
        vdb = VectorStore(database_path)
        doc_type = os.path.basename(folder)

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                logger.info("Processing: %s", file_path)
                pdf_file = PdfFile(
                    file_path, self.subject, self.examiner, doc_type
                )
                self.store_to_database(pdf_file, vdb)
