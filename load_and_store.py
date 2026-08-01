"""PDF loading, parsing, and vector store management for GCSE AI.

Handles loading exam papers and specifications from PDF files,
extracting question structures, and storing them in FAISS vector databases.
"""

import json
import logging
import os
import re
from typing import Optional

from langchain_core.documents import Document
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
        self, name: str, subject: str, examiner: str, doc_type: str,
        exam_type: str = ""
    ) -> None:
        """Initialize a PdfFile.

        Args:
            name: Path to the PDF file.
            subject: Subject name (e.g., "Biology").
            examiner: Exam board name (e.g., "Edexcel").
            doc_type: Document type ("Specification", "MarkSchemes", or "QuestionPapers").
            exam_type: Optional exam type/tier (e.g. "Higher", "Christianity"). When
                provided it is stored in metadata and used to tag documents ingested
                from an Exam-Types subfolder.
        """
        self.subject = subject
        self.examiner = examiner
        self.exam_type_override = exam_type
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
        self.sub_question_pattern = self.config.sub_question_pattern
        self.sub_sub_question_pattern = self.config.sub_sub_question_pattern
        self.question_pattern = self.config.question_pattern

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

    def extract_question_documents(self, content: str) -> list[Document]:
        """Extract questions from exam paper content and return as Document chunks.

        Parses the content into individual questions and constructs Document objects.

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
            content_str = question.pop("question_content") or ""
            meta = question.copy()
            meta["q_type"] = meta.pop("type")
            if meta.get("parent_question_structure"):
                meta["parent_question_structure"] = json.dumps(meta["parent_question_structure"])
            if meta.get("sub_questions"):
                meta["sub_questions"] = json.dumps(meta["sub_questions"])
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
        """Extract metadata from the PDF filename and folder path.

        For files under an ``Exam-Types/{ExamType}/`` subtree the exam type is
        detected from the path and stored as ``exam_type`` in the metadata dict.
        The filename convention is still: subject-examiner-type-topic-time.pdf

        Returns:
            Dict with keys: subject, examiner, type, topic, time, and optionally
            exam_type.
        """
        keys = ["subject", "examiner", "type", "topic", "time"]
        basename = os.path.basename(self.name)
        basename = basename.replace(".PDF", "").replace(".pdf", "")
        details = basename.split("-")

        meta_data = {}
        for i, value in enumerate(details):
            if i < len(keys):
                meta_data[keys[i]] = value

        # Detect exam type from folder path (e.g. .../Exam-Types/Christianity/questionPapers/)
        exam_type = self.exam_type_override
        if not exam_type:
            norm_path = os.path.normpath(self.name)
            parts = norm_path.split(os.sep)
            try:
                et_idx = next(
                    i for i, p in enumerate(parts)
                    if p.lower() == "exam-types"
                )
                # The directory immediately after "Exam-Types" is the exam type
                if et_idx + 1 < len(parts):
                    exam_type = parts[et_idx + 1]
            except StopIteration:
                pass

        if exam_type:
            meta_data["exam_type"] = exam_type

        logger.debug("Extracted metadata: %s", meta_data)
        return meta_data

    def _get_specification_structure(self) -> dict[str, list[str]]:
        """Load and parse specification topics and subtopics for this subject/examiner."""
        subject = getattr(self, "subject", "")
        examiner = getattr(self, "examiner", "")
        if not subject or not examiner:
            return {}

        spec_dir = os.path.normpath(f"data/{subject}/{examiner}/Specification")
        if not os.path.exists(spec_dir):
            return {}

        txt_files = [f for f in os.listdir(spec_dir) if f.lower().endswith(".txt")]
        spec_text = ""
        if txt_files:
            try:
                with open(os.path.join(spec_dir, txt_files[0]), "r", encoding="utf-8") as f:
                    spec_text = f.read()
            except Exception:
                pass

        if not spec_text:
            return {}

        structure: dict[str, list[str]] = {}
        topic_pattern = r"(?:Topic\s+\d+|Section\s+\d+|Paper\s+\d+)\s*[–\-:\s]+([^\n]+)"
        topics_found = re.findall(topic_pattern, spec_text, re.IGNORECASE)

        for top in topics_found[:15]:
            top_clean = top.strip()
            if top_clean and len(top_clean) > 3:
                structure[top_clean] = []

        sub_pattern = r"(\d+\.\d+)\s+([^\n]+)"
        sub_matches = re.findall(sub_pattern, spec_text)

        for code, sub_title in sub_matches:
            prefix = code.split(".")[0]
            target_key = None
            for key in structure.keys():
                if f"topic {prefix}" in key.lower() or f"{prefix}." in key:
                    target_key = key
                    break
            if not target_key and structure:
                target_key = list(structure.keys())[0]

            if target_key:
                structure[target_key].append(f"{code} {sub_title.strip()[:60]}")

        return structure

    def classify_question_topic_and_subtopic(
        self, question_text: str, file_topic: str
    ) -> tuple[str, str]:
        """Classify a question into its specification topic first, and then subtopic.

        Args:
            question_text: Full text of the question.
            file_topic: Default topic derived from filename metadata.

        Returns:
            Tuple of (classified_topic, classified_subtopic).
        """
        spec_structure = self._get_specification_structure()
        if not spec_structure:
            return file_topic, file_topic

        # Step 1: Define Topic using specification structure
        matched_topic = file_topic
        for topic_name in spec_structure.keys():
            if topic_name.lower() in file_topic.lower() or file_topic.lower() in topic_name.lower():
                matched_topic = topic_name
                break

        if matched_topic == file_topic:
            best_topic_score = 0
            for topic_name in spec_structure.keys():
                score = sum(1 for word in topic_name.lower().split() if len(word) > 3 and word in question_text.lower())
                if score > best_topic_score:
                    best_topic_score = score
                    matched_topic = topic_name

        # Step 2: Define Subtopic under the identified Topic
        subtopics = spec_structure.get(matched_topic, [])
        matched_subtopic = matched_topic

        if subtopics:
            best_subtopic_score = -1
            for subtopic_name in subtopics:
                keywords = [w for w in re.findall(r'\w+', subtopic_name.lower()) if len(w) > 3]
                score = sum(1 for kw in keywords if kw in question_text.lower())
                if score > best_subtopic_score:
                    best_subtopic_score = score
                    matched_subtopic = subtopic_name

        return matched_topic, matched_subtopic

    def process_questions(
        self, questions: list[str], topic: str, exam: str
    ) -> list[dict]:
        """Process raw question text splits into structured question dicts.

        Handles nested questions:
        - basic_question: standalone questions
        - parent_question: questions with sub-parts and grandchild sub-sub-parts

        Args:
            questions: List of raw question text strings.
            topic: The exam topic.
            exam: The exam identifier (time period).

        Returns:
            List of structured question dictionaries.
        """
        questions_info: list[dict] = []
        
        letters = "abcdefghijklmn"
        romans = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

        sub_q_pat = self.sub_question_pattern
        sub_sub_q_pat = self.sub_sub_question_pattern

        for question in questions:
            question_stripped = question.strip()
            if not question_stripped:
                continue

            # Classify official specification topic first, then subtopic
            classified_topic, classified_subtopic = self.classify_question_topic_and_subtopic(
                question_stripped, topic
            )

            has_sub = False
            sub_questions_list = []
            question_structure = []
            parent_description = ""

            if sub_q_pat and sub_q_pat != "None":
                splits = re.split(sub_q_pat, question)
                if len(splits) > 1:
                    has_sub = True
                    parent_description = splits[0].strip()
                    sub_question_texts = splits[1:]

                    for idx, sq_text in enumerate(sub_question_texts):
                        label = f"{letters[idx]})"
                        # Check if this sub-question has sub-sub-questions
                        has_sub_sub = False
                        sub_sub_texts = []
                        sq_intro = ""
                        
                        if sub_sub_q_pat and sub_sub_q_pat != "None":
                            ss_splits = re.split(sub_sub_q_pat, sq_text)
                            if len(ss_splits) > 1:
                                has_sub_sub = True
                                sq_intro = ss_splits[0].strip()
                                sub_sub_texts = ss_splits[1:]

                        if has_sub_sub:
                            sub_sub_list = []
                            sub_sub_structure = []
                            for ss_idx, ss_text in enumerate(sub_sub_texts):
                                ss_marks = self.extract_mark(ss_text, self.marks_pattern)
                                r_label = f"{romans[ss_idx]})"
                                sub_sub_structure.append(ss_marks)
                                sub_sub_list.append({
                                    "label": r_label,
                                    "text": ss_text.strip(),
                                    "marks": ss_marks
                                })
                            question_structure.append(sub_sub_structure)
                            sub_questions_list.append({
                                "label": label,
                                "context": sq_intro,
                                "sub_parts": sub_sub_list
                            })
                        else:
                            sq_marks = self.extract_mark(sq_text, self.marks_pattern)
                            question_structure.append(sq_marks)
                            sub_questions_list.append({
                                "label": label,
                                "text": sq_text.strip(),
                                "marks": sq_marks
                            })

            meta_dict = getattr(self, "meta_data", {})
            exam_type = meta_dict.get("exam_type", getattr(self, "exam_type_override", ""))
            if has_sub:
                questions_info.append({
                    "type": "parent_question",
                    "topic": classified_topic,
                    "subtopic": classified_subtopic,
                    "marks": None,
                    "question_content": question,
                    "parent_question_structure": question_structure,
                    "parent_description": parent_description,
                    "sub_questions": sub_questions_list,
                    "exam": exam,
                    "exam_type": exam_type
                })
            else:
                marks = self.extract_mark(question, self.marks_pattern)
                questions_info.append({
                    "type": "basic_question",
                    "topic": classified_topic,
                    "subtopic": classified_subtopic,
                    "marks": marks,
                    "question_content": question,
                    "parent_question_structure": None,
                    "exam": exam,
                    "exam_type": exam_type
                })

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

        Detects whether the folder lives inside an ``Exam-Types/{ExamType}``
        subtree and, when it does, passes the exam type through to ``PdfFile``
        so it is stored in every document's metadata.

        Args:
            folder: Path to the folder containing PDF files.
            database_path: Path for the FAISS vector database.
        """
        vdb = VectorStore(database_path)
        doc_type = os.path.basename(folder)

        # Detect exam type from the folder path
        exam_type = ""
        norm_folder = os.path.normpath(folder)
        parts = norm_folder.split(os.sep)
        try:
            et_idx = next(
                i for i, p in enumerate(parts)
                if p.lower() == "exam-types"
            )
            if et_idx + 1 < len(parts):
                exam_type = parts[et_idx + 1]
        except StopIteration:
            pass

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                if not filename.lower().endswith(".pdf"):
                    logger.info("Skipping non-PDF file: %s", file_path)
                    continue
                logger.info("Processing: %s", file_path)
                pdf_file = PdfFile(
                    file_path, self.subject, self.examiner, doc_type,
                    exam_type=exam_type,
                )
                self.store_to_database(pdf_file, vdb)
