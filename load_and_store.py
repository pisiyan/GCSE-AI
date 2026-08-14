"""PDF loading, parsing, and vector store management for GCSE AI.

Handles loading exam papers and specifications from PDF files,
extracting question structures, and storing them in FAISS vector databases.
"""

import json
import logging
import os
import re
from typing import Any, Optional

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


def clean_ingested_question_text(text: str) -> str:
    """Clean raw question content during ingestion by removing OCR glitches, margins, and exam boilerplate."""
    if not text:
        return ""

    cleaned = text

    # Remove margin instructions & boilerplate with OCR spacing glitches
    cleaned = re.sub(r"(?i)DO\s*NO?\s*T?\s*WRITE?\s*IN?\s*THIS?\s*(?:AREA|MARGIN|PAGE).*", "", cleaned)
    cleaned = re.sub(r"(?i)DO\s*NOT\s*WRITE\s*ON\s*THIS\s*PAGE.*", "", cleaned)
    cleaned = re.sub(r"(?i)\(?\s*Total\s+for\s+Question.*", "", cleaned)
    cleaned = re.sub(r"(?i)\(?\s*Total\s+\d+\s+marks?\s*\)?", "", cleaned)

    # Remove exam paper instructions & boilerplate
    cleaned = re.sub(r"(?i)Answer\s+ALL\s+questions.*", "", cleaned)
    cleaned = re.sub(r"(?i)Write\s+your\s+answers\s+in\s+the\s+spaces\s+provided.*", "", cleaned)
    cleaned = re.sub(r"(?i)Some\s+questions\s+must\s+be\s+answered\s+with\s+a\s+cross.*", "", cleaned)
    cleaned = re.sub(r"(?i)If\s+you\s+change\s+your\s+mind.*", "", cleaned)
    cleaned = re.sub(r"(?i)mark\s+your\s+new\s+answer.*", "", cleaned)
    cleaned = re.sub(r"(?i)\bTurn\s+over\b(?:\s+for\s+next\s+question)?", "", cleaned)
    cleaned = re.sub(r"(?i)\bBLANK\s+PAGE\b", "", cleaned)
    cleaned = re.sub(r"(?i)Pearson\s+Edexcel.*", "", cleaned)
    cleaned = re.sub(r"\b[A-Z]\d{4,}[A-Z0-9]*\b", "", cleaned)
    cleaned = re.sub(r"\*\s*[A-Z0-9]{5,}\s*\*", "", cleaned)

    # Remove long dotted or underlined prompt lines
    cleaned = re.sub(r"\.{3,}", "", cleaned)
    cleaned = re.sub(r"_{3,}", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\(\d+\)\s*$", "", cleaned)

    # Collapse multiple blank lines & spaces
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines() if line.strip()]

    # Filter out standalone page number lines or single digits
    filtered_lines = []
    for line in lines:
        if re.match(r"^\d+$", line) or re.match(r"(?i)^Page\s+\d+$", line):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def load_specification_summary(subject: str, examiner: str) -> dict:
    """Load pre-generated specification summary JSON for a subject/examiner."""
    if not subject or not examiner:
        return {}
    path = f"data/{subject}/{examiner}/{subject}-{examiner}-specification_summary.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read specification summary JSON %s: %s", path, e)
    return {}


def classify_question_with_llm(
    llm: Optional[Any],
    spec_summary: dict,
    question_content: str,
    file_topic: str,
    exam_type: str = "",
    is_parent: bool = False,
    sub_questions: Optional[list[dict]] = None
) -> tuple[str, Any]:
    """Classify 1 question (basic or parent) using LLM and specification summary hierarchy.

    Returns:
        For basic question: tuple of (topic_name, subtopic_name)
        For parent question: tuple of (topic_name, list_of_subtopic_names_for_sub_questions)
    """
    if not spec_summary or not llm:
        return file_topic, (file_topic if not is_parent else [file_topic] * len(sub_questions or []))

    allowed_topics_map: dict[str, list[str]] = {}
    exam_types_data = spec_summary.get("exam_types", [])

    target_et = exam_type.strip().lower() if exam_type else ""
    matching_et_item = None
    for et_item in exam_types_data:
        if target_et and target_et in str(et_item.get("exam_type", "")).strip().lower():
            matching_et_item = et_item
            break

    if not matching_et_item and exam_types_data:
        matching_et_item = exam_types_data[0]

    if matching_et_item:
        for t_item in matching_et_item.get("topics", []):
            top_name = t_item.get("topic", "").strip()
            subs = t_item.get("subtopics", [])
            allowed_topics_map[top_name] = [str(s).strip() for s in subs]

    if not allowed_topics_map:
        return file_topic, (file_topic if not is_parent else [file_topic] * len(sub_questions or []))

    hierarchy_str = json.dumps(allowed_topics_map, indent=2, ensure_ascii=False)

    if not is_parent:
        prompt = f"""You are an expert GCSE exam classifier.
Analyze the question below and assign its official Topic and Subtopic.

Allowed Topics & Subtopics Hierarchy:
{hierarchy_str}

Question Content:
{question_content[:3000]}

You MUST choose EXACTLY ONE topic from the allowed topics list, and EXACTLY ONE subtopic under that chosen topic.

Return strictly a JSON object:
{{
  "topic": "Selected Topic Name",
  "subtopic": "Selected Subtopic Name"
}}
Do not return any markdown fences or explanation, return only raw JSON."""

        try:
            res = llm.invoke_json(prompt)
            chosen_top = str(res.get("topic", "")).strip()
            chosen_sub = str(res.get("subtopic", "")).strip()

            if chosen_top in allowed_topics_map:
                if chosen_sub in allowed_topics_map[chosen_top]:
                    return chosen_top, chosen_sub
                elif allowed_topics_map[chosen_top]:
                    return chosen_top, allowed_topics_map[chosen_top][0]
                return chosen_top, chosen_top

            for valid_top, valid_subs in allowed_topics_map.items():
                if valid_top.lower() in chosen_top.lower() or chosen_top.lower() in valid_top.lower():
                    chosen_sub_final = valid_subs[0] if valid_subs else valid_top
                    return valid_top, chosen_sub_final

        except Exception as e:
            logger.warning("LLM classification failed for basic question: %s", e)

        default_top = list(allowed_topics_map.keys())[0]
        default_sub = allowed_topics_map[default_top][0] if allowed_topics_map[default_top] else default_top
        return default_top, default_sub

    else:
        sq_descriptions = []
        for sq in (sub_questions or []):
            lbl = sq.get("label", "")
            txt = sq.get("text", "") or sq.get("context", "")
            sq_descriptions.append(f"- {lbl} {txt[:500]}")
        sq_text_block = "\n".join(sq_descriptions)

        prompt = f"""You are an expert GCSE exam classifier.
Analyze the parent question and its sub-parts below and assign its official Topic and Subtopics.

Allowed Topics & Subtopics Hierarchy:
{hierarchy_str}

Parent Question Context:
{question_content[:2000]}

Sub-questions:
{sq_text_block}

Choose EXACTLY ONE overall topic for the parent question from the allowed topics list.
For each sub-question, choose EXACTLY ONE subtopic under that chosen topic.

Return strictly a JSON object:
{{
  "topic": "Selected Topic Name",
  "subtopics": [
    {{"label": "a)", "subtopic": "Subtopic Name for a)"}}
  ]
}}
Do not return any markdown fences or explanation, return only raw JSON."""

        try:
            res = llm.invoke_json(prompt)
            chosen_top = str(res.get("topic", "")).strip()
            raw_subs = res.get("subtopics", [])

            valid_top = list(allowed_topics_map.keys())[0]
            if chosen_top in allowed_topics_map:
                valid_top = chosen_top
            else:
                for t_key in allowed_topics_map.keys():
                    if t_key.lower() in chosen_top.lower() or chosen_top.lower() in t_key.lower():
                        valid_top = t_key
                        break

            allowed_subs = allowed_topics_map.get(valid_top, [valid_top])
            subtopic_list = []
            if isinstance(raw_subs, list):
                for item in raw_subs:
                    if isinstance(item, dict) and item.get("subtopic"):
                        s_name = str(item.get("subtopic")).strip()
                        subtopic_list.append(s_name if s_name in allowed_subs else (allowed_subs[0] if allowed_subs else valid_top))

            while len(subtopic_list) < len(sub_questions or []):
                subtopic_list.append(allowed_subs[0] if allowed_subs else valid_top)

            return valid_top, subtopic_list

        except Exception as e:
            logger.warning("LLM classification failed for parent question: %s", e)

        default_top = list(allowed_topics_map.keys())[0]
        default_subs = allowed_topics_map.get(default_top, [default_top])
        return default_top, [default_subs[0] if default_subs else default_top] * len(sub_questions or [])


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
        """Extract a mark value from text using a regex pattern with fallbacks.

        Args:
            text: The text to search.
            pattern: Regex pattern with a capture group for the mark number.

        Returns:
            The extracted mark as an integer, or None if not found.
        """
        if not text:
            return None

        # 1. Try subject-configured primary pattern
        try:
            regex = re.compile(pattern)
            matches = list(regex.finditer(text))
            if matches:
                last_match = matches[-1]
                if last_match.groups():
                    return int(last_match.group(1))
                return int(last_match.group())
        except (ValueError, TypeError, re.error):
            pass

        # 2. Try common mark pattern fallbacks: (1), (2), [2 marks], [2]
        fallback_patterns = [
            r"\((\d{1,2})\)",
            r"\[\s*(\d{1,2})\s*(?:marks?)?\s*\]",
            r"(?i)(\d{1,2})\s*marks?"
        ]
        for fb_pat in fallback_patterns:
            fb_matches = list(re.finditer(fb_pat, text))
            if fb_matches:
                try:
                    return int(fb_matches[-1].group(1))
                except (ValueError, TypeError):
                    pass
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
        spec_summary = load_specification_summary(getattr(self, "subject", ""), getattr(self, "examiner", ""))
        classifier_llm = None
        if spec_summary:
            try:
                from llm_client import LLMClient
                model_name = getattr(self.config, "LLM_MODEL", "gpt-5.4-mini")
                classifier_llm = LLMClient(model=model_name)
            except Exception as e:
                logger.warning("Could not initialize LLMClient for question classification: %s", e)

        for question in questions:
            question_cleaned = clean_ingested_question_text(question)
            if not question_cleaned:
                continue

            # Extract raw basic mark before clean text strips mark numbers
            raw_basic_mark = self.extract_mark(question, self.marks_pattern)

            has_sub = False
            sub_questions_list = []
            question_structure = []
            parent_description = ""

            if sub_q_pat and sub_q_pat != "None":
                splits = re.split(sub_q_pat, question)
                if len(splits) > 1:
                    has_sub = True
                    parent_description = clean_ingested_question_text(splits[0])
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
                                sq_intro = clean_ingested_question_text(ss_splits[0])
                                sub_sub_texts = ss_splits[1:]

                        if has_sub_sub:
                            sub_sub_list = []
                            sub_sub_structure = []
                            for ss_idx, ss_text in enumerate(sub_sub_texts):
                                ss_marks = self.extract_mark(ss_text, self.marks_pattern) or 1
                                ss_clean = clean_ingested_question_text(ss_text)
                                r_label = f"{romans[ss_idx]})"
                                sub_sub_structure.append(ss_marks)
                                sub_sub_list.append({
                                    "label": r_label,
                                    "text": ss_clean,
                                    "marks": ss_marks
                                })
                            question_structure.append(sub_sub_structure)
                            sub_questions_list.append({
                                "label": label,
                                "context": sq_intro,
                                "sub_parts": sub_sub_list
                            })
                        else:
                            sq_marks = self.extract_mark(sq_text, self.marks_pattern) or 1
                            sq_clean = clean_ingested_question_text(sq_text)
                            question_structure.append(sq_marks)
                            sub_questions_list.append({
                                "label": label,
                                "text": sq_clean,
                                "marks": sq_marks
                            })

            meta_dict = getattr(self, "meta_data", {})
            exam_type = meta_dict.get("exam_type", getattr(self, "exam_type_override", ""))

            # Classify using LLM and specification summary
            if has_sub:
                classified_topic, subtopic_names = classify_question_with_llm(
                    classifier_llm, spec_summary, parent_description or question_cleaned, topic, exam_type, is_parent=True, sub_questions=sub_questions_list
                )
                for idx, sq_part in enumerate(sub_questions_list):
                    sq_part["subtopic"] = subtopic_names[idx] if idx < len(subtopic_names) else (subtopic_names[0] if subtopic_names else classified_topic)
                classified_subtopic = subtopic_names[0] if isinstance(subtopic_names, list) and subtopic_names else classified_topic
            else:
                classified_topic, classified_subtopic = classify_question_with_llm(
                    classifier_llm, spec_summary, question_cleaned, topic, exam_type, is_parent=False
                )

            if has_sub:
                # Calculate total parent question marks from question structure
                flat_marks = []
                def _flatten(item):
                    if isinstance(item, list):
                        for sub in item:
                            _flatten(sub)
                    elif isinstance(item, int) and item > 0:
                        flat_marks.append(item)
                _flatten(question_structure)
                parent_total_marks = sum(flat_marks) if flat_marks else None

                questions_info.append({
                    "type": "parent_question",
                    "topic": classified_topic,
                    "subtopic": classified_subtopic,
                    "marks": parent_total_marks,
                    "question_content": question_cleaned,
                    "parent_question_structure": question_structure,
                    "parent_description": parent_description,
                    "sub_questions": sub_questions_list,
                    "exam": exam,
                    "exam_type": exam_type
                })
            else:
                marks = raw_basic_mark or 1
                questions_info.append({
                    "type": "basic_question",
                    "topic": classified_topic,
                    "subtopic": classified_subtopic,
                    "marks": marks,
                    "question_content": question_cleaned,
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

    def dump_database_to_string(self, subject: str, examiner: str) -> str:
        """Export the full vector database contents as a human-readable text string."""
        if not os.path.exists(self.vector_database_name):
            return f"Vector database '{self.vector_database_name}' does not exist."

        db = FAISS.load_local(
            self.vector_database_name,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        all_docs = list(db.docstore._dict.values())

        lines = [
            "=" * 80,
            f"VECTOR DATABASE CONTENTS: {subject} ({examiner})",
            f"Database Path: {self.vector_database_name}",
            f"Total Ingested Document Chunks: {len(all_docs)}",
            "=" * 80,
            ""
        ]

        for idx, doc in enumerate(all_docs, start=1):
            meta = doc.metadata or {}
            doc_type = meta.get("type") or meta.get("doc_type") or "Chunk"
            q_type = meta.get("q_type") or meta.get("type", "")
            topic = meta.get("topic", "N/A")
            subtopic = meta.get("subtopic", "N/A")
            exam_type = meta.get("exam_type", "N/A")
            marks = meta.get("marks", "N/A")
            exam = meta.get("exam", "N/A")

            lines.append("-" * 80)
            lines.append(f"DOCUMENT #{idx}")
            lines.append(f"Document Type: {doc_type} | Question Type: {q_type}")
            lines.append(f"Subject: {subject} | Examiner: {examiner} | Exam Type: {exam_type} | Exam Period: {exam}")
            lines.append(f"Topic: {topic}")
            lines.append(f"Subtopic: {subtopic}")
            lines.append(f"Marks: {marks}")

            if "parent_question_structure" in meta:
                lines.append(f"Parent Question Structure: {meta['parent_question_structure']}")
            if "sub_questions" in meta:
                lines.append(f"Sub Questions JSON: {meta['sub_questions']}")

            lines.append("\n[Page Content]:")
            lines.append(doc.page_content.strip() if doc.page_content else "(empty)")
            lines.append("-" * 80)
            lines.append("")

        return "\n".join(lines)


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
