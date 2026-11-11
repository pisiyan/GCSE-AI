"""PDF loading, parsing, and vector store management for GCSE AI.

Handles loading exam papers and specifications from PDF files,
extracting question structures, cropping diagram images, and storing them in FAISS vector databases.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import SubjectConfig, load_subject_config
from llm.client import LLMClient
from llm.prompts import load_prompt

logger = logging.getLogger(__name__)

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

    # Remove margin instructions & boilerplate with OCR glitches
    cleaned = re.sub(r"(?i)DO\s*NO?\s*T?\s*WRITE?\s*(?:IN?\s*THIS?\s*(?:AREA|MARGIN|PAGE)|OUTSIDE\s*(?:THE\s*)?BOX|ON\s*THIS\s*PAGE)?.*", "", cleaned)
    cleaned = re.sub(r"(?i)\boutside\s*the\s*box\b.*", "", cleaned)
    cleaned = re.sub(r"(?i)\boutside\s*the\b.*", "", cleaned)
    cleaned = re.sub(r"(?i)^\s*box\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"(?i)\(?\s*Total\s+for\s+Question.*", "", cleaned)
    cleaned = re.sub(r"(?i)\(?\s*Total\s+\d+\s+marks?\s*\)?", "", cleaned)

    # Remove exam paper instructions, codes, & boilerplate
    cleaned = re.sub(r"(?i)TOTAL\s+FOR\s+PAPER\s*=\s*\d+\s*MARKS.*", "", cleaned)
    cleaned = re.sub(r"(?i)Every\s+effort\s+has\s+been\s+made\s+to\s+contact\s+copyright\s+holders.*", "", cleaned)
    cleaned = re.sub(r"(?i)Answer\s+ALL\s+questions.*", "", cleaned)
    cleaned = re.sub(r"(?i)Write\s+your\s+answers\s+in\s+the\s+spaces\s+provided.*", "", cleaned)
    cleaned = re.sub(r"(?i)Some\s+questions\s+must\s+be\s+answered\s+with\s+a\s+cross.*", "", cleaned)
    cleaned = re.sub(r"(?i)If\s+you\s+change\s+your\s+mind.*", "", cleaned)
    cleaned = re.sub(r"(?i)mark\s+your\s+new\s+answer.*", "", cleaned)
    cleaned = re.sub(r"(?i)\bTurn\s+over\b(?:\s+for\s+next\s+question)?", "", cleaned)
    cleaned = re.sub(r"(?i)\bBLANK\s+PAGE\b", "", cleaned)
    cleaned = re.sub(r"(?i)\bExtra\s+space\b.*", "", cleaned)
    cleaned = re.sub(r"(?i)\bEND\s+OF\s+QUESTIONS\b.*", "", cleaned)
    cleaned = re.sub(r"(?i)\bThere\s+are\s+no\s+questions\b.*", "", cleaned)
    cleaned = re.sub(r"(?i)\bAdditional\s+page\b.*", "", cleaned)
    cleaned = re.sub(r"(?i)\bQuestion\s+(?:number|\d+\s+continues\s+on\s+the\s+next\s+page)\b.*", "", cleaned)
    cleaned = re.sub(r"(?is)\bCopyright\b.*", "", cleaned)
    cleaned = re.sub(r"(?i)Pearson\s+Edexcel.*", "", cleaned)
    cleaned = re.sub(r"\b[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+\b", "", cleaned)
    cleaned = re.sub(r"\b[A-Z0-9_-]*\d[A-Z0-9_-]{5,}\b", "", cleaned)
    cleaned = re.sub(r"\*\s*[-A-Za-z0-9_/]+\s*\*", "", cleaned)

    # Remove prompt lines and decorative glyphs
    cleaned = re.sub(r"[\u2022\u25ba\u25cf\u25aa\u25ab\uf0fc]", " • ", cleaned)
    cleaned = re.sub(r"\.{2,}", "", cleaned)
    cleaned = re.sub(r"_{2,}", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\(\d+\)\s*$", "", cleaned)

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines() if line.strip()]
    filtered_lines = [l for l in lines if not (re.match(r"^[\d\.\_\-\s\u2022\u25ba\u25cf\u25aa\u25ab\*]+$", l) or re.match(r"(?i)^Page\s+\d+$", l))]
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
    sub_questions: Optional[List[dict]] = None,
) -> Tuple[str, Any]:
    """Classify 1 question using LLM and specification summary hierarchy."""
    if not spec_summary or not llm:
        return file_topic, (file_topic if not is_parent else [file_topic] * len(sub_questions or []))

    allowed_topics_map: Dict[str, List[str]] = {}
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
        prompt = load_prompt(
            "classify_question_basic",
            hierarchy_str=hierarchy_str,
            question_content=question_content[:3000],
        )
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

        prompt = load_prompt(
            "classify_question_parent",
            hierarchy_str=hierarchy_str,
            parent_context=question_content[:2000],
            sub_questions_text=sq_text_block[:2000],
        )
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


class Question:
    """Class representing an ingested question with metadata and screenshot image path(s)."""

    def __init__(
        self,
        question_content: str,
        q_type: str = "basic_question",
        topic: str = "",
        subtopic: str = "",
        marks: Optional[int] = None,
        exam: str = "",
        exam_type: str = "",
        parent_description: Optional[str] = None,
        parent_question_structure: Optional[Any] = None,
        sub_questions: Optional[list] = None,
        image_paths: Optional[list[str]] = None,
    ):
        self.question_content = question_content
        self.q_type = q_type
        self.topic = topic
        self.subtopic = subtopic
        self.marks = marks
        self.exam = exam
        self.exam_type = exam_type
        self.parent_description = parent_description
        self.parent_question_structure = parent_question_structure
        self.sub_questions = sub_questions
        self.image_paths = image_paths or []

    def to_dict(self) -> dict:
        return {
            "question_content": self.question_content,
            "type": self.q_type,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "marks": self.marks,
            "exam": self.exam,
            "exam_type": self.exam_type,
            "parent_description": self.parent_description,
            "parent_question_structure": self.parent_question_structure,
            "sub_questions": self.sub_questions,
            "image_paths": self.image_paths,
        }


def crop_question_screenshots(
    pdf_path: str,
    subject: str,
    examiner: str,
    questions_raw_text: list[str],
) -> list[list[str]]:
    """Render full question rectangle clips (text + diagrams) into PNG images page by page."""
    if not pdf_path or not os.path.exists(pdf_path) or not questions_raw_text:
        return [[] for _ in questions_raw_text]

    img_dir = os.path.normpath(f"data/{subject}/{examiner}/question_images")
    os.makedirs(img_dir, exist_ok=True)
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        logger.error("Failed to open PDF %s with PyMuPDF: %s", pdf_path, e)
        return [[] for _ in questions_raw_text]

    if len(doc) <= 1:
        doc.close()
        return [[] for _ in questions_raw_text]

    cfg = load_subject_config(subject, examiner)
    start_pattern = getattr(cfg, "question_start_pattern", None) or cfg.question_pattern
    end_pattern = getattr(cfg, "question_end_pattern", None)

    start_regex = re.compile(start_pattern, re.IGNORECASE) if start_pattern else None
    end_regex = re.compile(end_pattern, re.IGNORECASE) if end_pattern else None

    # 1. Map question boundaries across pages (skipping cover page 0)
    question_spans: dict[int, dict] = {}
    current_q_num = None

    for p_idx in range(1, len(doc)):
        page = doc[p_idx]
        blocks = page.get_text("blocks")

        for b in blocks:
            b_text = b[4].strip() if len(b) > 4 and isinstance(b[4], str) else ""
            b_type = b[6] if len(b) > 6 else 0
            if b_type != 0 or not b_text:
                continue

            m_start = start_regex.search(b_text) if start_regex else None
            m_end = end_regex.search(b_text) if end_regex else None

            if m_start and not re.search(r"(?i)continues", b_text) and not re.search(r"^\d+\.\d+", b_text):
                raw_num = m_start.group(1) if m_start.lastindex and m_start.lastindex >= 1 else (
                    m_start.group(2) if m_start.lastindex and m_start.lastindex >= 2 else None
                )
                try:
                    q_num = int(raw_num) if raw_num and raw_num.isdigit() else (len(question_spans) + 1)
                except Exception:
                    q_num = len(question_spans) + 1

                if current_q_num != q_num:
                    if current_q_num is not None and current_q_num in question_spans and "end_page" not in question_spans[current_q_num]:
                        # Check if previous question had content on this page above the new question
                        has_content_above = False
                        for prev_b in blocks:
                            if prev_b[3] > b[1]:
                                break
                            t = prev_b[4].strip() if len(prev_b) > 4 and isinstance(prev_b[4], str) else ""
                            if t and clean_ingested_question_text(t):
                                has_content_above = True
                                break

                        if has_content_above:
                            question_spans[current_q_num]["end_page"] = p_idx
                            question_spans[current_q_num]["end_y"] = b[1] - 4.0
                        else:
                            prev_p = max(question_spans[current_q_num]["start_page"], p_idx - 1)
                            question_spans[current_q_num]["end_page"] = prev_p
                            question_spans[current_q_num]["end_y"] = doc[prev_p].rect.height - 30.0

                    current_q_num = q_num
                    if q_num not in question_spans:
                        question_spans[q_num] = {"start_page": p_idx, "start_y": b[1]}

            if m_end:
                end_num = None
                if m_end.lastindex and m_end.lastindex >= 1:
                    try:
                        end_num = int(m_end.group(1))
                    except Exception:
                        pass
                target_q = end_num if end_num in question_spans else current_q_num
                if target_q is not None and target_q in question_spans:
                    question_spans[target_q]["end_page"] = p_idx
                    question_spans[target_q]["end_y"] = b[3]
                current_q_num = None

    # 2. Render clips for each detected question span
    all_image_paths = [[] for _ in questions_raw_text]

    for q_num, span in sorted(question_spans.items()):
        start_p = span.get("start_page")
        end_p = span.get("end_page", start_p)
        start_y = span.get("start_y", 50.0)
        end_y = span.get("end_y", doc[end_p].rect.height - 30.0)

        q_idx = q_num - 1
        if q_idx >= len(all_image_paths):
            continue

        for p_idx in range(start_p, end_p + 1):
            page = doc[p_idx]
            p_h = page.rect.height
            p_w = page.rect.width

            y_top = max(0.0, start_y - 12.0) if p_idx == start_p else 30.0
            y_bot = min(p_h, end_y + 10.0) if p_idx == end_p else (p_h - 30.0)

            if y_bot <= y_top + 25.0:
                continue

            rect = pymupdf.Rect(0.0, y_top, p_w, y_bot)

            # Filter out pages/clips that have no question content (e.g. empty ruled answer lines, blank pages)
            clip_text = page.get_text("text", clip=rect)
            if not clean_ingested_question_text(clip_text).strip():
                continue

            try:
                img_name = f"{pdf_basename}_q{q_num}_p{p_idx + 1}.png"
                img_path = os.path.normpath(os.path.join(img_dir, img_name))
                pix = page.get_pixmap(clip=rect, matrix=pymupdf.Matrix(2.0, 2.0))
                pix.save(img_path)
                all_image_paths[q_idx].append(img_path)
            except Exception as e:
                logger.warning("Failed rendering clip for Q%d p%d: %s", q_num, p_idx + 1, e)

    doc.close()
    return all_image_paths


class PdfFile:
    """Represents a PDF file for processing and question extraction."""

    def __init__(
        self, name: str, subject: str, examiner: str, doc_type: str,
        exam_type: str = ""
    ) -> None:
        self.subject = subject
        self.examiner = examiner
        self.exam_type_override = exam_type
        self.config = load_subject_config(subject, examiner)
        self.name = name
        self.doc_type = doc_type
        self.meta_data = self._get_metadata()
        self.info = self.meta_data

        self.marks_pattern = self.config.mark_pattern
        self.sub_question_pattern = self.config.sub_question_pattern
        self.sub_sub_question_pattern = self.config.sub_sub_question_pattern
        self.question_pattern = self.config.question_pattern
        self.question_split_pattern = getattr(self.config, "question_split_pattern", self.config.question_pattern)

    def extract_mark(self, text: str, pattern: str) -> Optional[int]:
        """Extract a mark value from text using regex with fallbacks."""
        if not text:
            return None
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

        fallback_patterns = [r"\((\d{1,2})\)", r"\[\s*(\d{1,2})\s*(?:marks?)?\s*\]", r"(?i)(\d{1,2})\s*marks?"]
        for fb_pat in fallback_patterns:
            fb_matches = list(re.finditer(fb_pat, text))
            if fb_matches:
                try:
                    return int(fb_matches[-1].group(1))
                except (ValueError, TypeError):
                    pass
        return None

    def add_metadata(self, chunks: list[Document]) -> list[Document]:
        """Add metadata to document chunks."""
        for doc in chunks:
            doc.metadata.update(self.meta_data)
        return chunks

    def load_pdf(self) -> list[Document]:
        """Load a PDF file and return its pages as Documents using PyMuPDF."""
        doc = pymupdf.open(self.name)
        pages = []
        for i, page in enumerate(doc):
            pages.append(Document(page_content=page.get_text(), metadata={"page": i}))
        doc.close()
        logger.info("Loaded PDF: %s", self.name)
        return pages

    def flatten(self, nested_list: Any) -> List[Any]:
        """Flatten a nested list of items."""
        flat = []
        if isinstance(nested_list, list):
            for item in nested_list:
                if isinstance(item, list):
                    flat.extend(self.flatten(item))
                else:
                    flat.append(item)
        else:
            flat.append(nested_list)
        return flat

    def process_questions(
        self, questions: List[str], topic: str, exam: str
    ) -> List[Dict[str, Any]]:
        """Process raw question text splits into structured question dicts."""
        questions_info: List[Dict[str, Any]] = []
        letters = "abcdefghijklmn"
        romans = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

        sub_q_pat = getattr(self, "sub_question_pattern", None)
        sub_sub_q_pat = getattr(self, "sub_sub_question_pattern", None)

        for question in questions:
            question_cleaned = clean_ingested_question_text(question)
            if not question_cleaned:
                continue

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
                        label = f"{letters[idx]})" if idx < len(letters) else f"part_{idx+1}"
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

            if has_sub:
                flat_marks = self.flatten(question_structure)
                parent_total_marks = sum(flat_marks) if flat_marks else None
                questions_info.append({
                    "type": "parent_question",
                    "topic": topic,
                    "subtopic": topic,
                    "marks": parent_total_marks,
                    "question_content": question_cleaned,
                    "parent_question_structure": question_structure,
                    "parent_description": parent_description,
                    "sub_questions": sub_questions_list,
                    "exam": exam,
                    "exam_type": getattr(self, "exam_type_override", "")
                })
            else:
                marks = raw_basic_mark or 1
                questions_info.append({
                    "type": "basic_question",
                    "topic": topic,
                    "subtopic": topic,
                    "marks": marks,
                    "question_content": question_cleaned,
                    "parent_question_structure": None,
                    "exam": exam,
                    "exam_type": getattr(self, "exam_type_override", "")
                })

        return questions_info

    def pdf_to_text(self, pages: Any) -> str:
        """Convert PDF pages to a single text string."""
        return "\n".join(page.page_content for page in pages)

    def _get_metadata(self) -> dict:
        """Extract metadata from filename and document path."""
        meta = {
            "type": self.doc_type,
            "doc_class": self.doc_type.lower(),
            "subject": self.subject,
            "examiner": self.examiner,
            "exam_type": self.exam_type_override or self.examiner,
            "exam": os.path.splitext(os.path.basename(self.name))[0],
            "file_name": os.path.basename(self.name),
        }
        return meta

    def split_document(self, document: pymupdf.Document) -> List[Document]:
        """Extract structured Document objects for QuestionPaper, MarkScheme, or Specification."""
        pages = [Document(page_content=page.get_text(), metadata={"page": i}) for i, page in enumerate(document)]
        full_text = self.pdf_to_text(pages)

        if "specification" in self.doc_type.lower():
            # Return full specification document (no chunking needed)
            return [Document(page_content=full_text, metadata={"type": "Specification", "doc_class": "specification"})]

        elif "mark" in self.doc_type.lower():
            ms_entries = re.split(self.config.ms_pattern, full_text)
            ms_docs = []
            for entry in ms_entries:
                cleaned = clean_ingested_question_text(entry)
                if cleaned:
                    marks = self.extract_mark(entry, self.config.ms_mark_pattern)
                    ms_docs.append(Document(
                        page_content=cleaned,
                        metadata={
                            "type": "MarkScheme",
                            "doc_class": "mark_scheme",
                            "marks": marks,
                            "exam": self.meta_data.get("exam", ""),
                            "exam_type": self.meta_data.get("exam_type", ""),
                        }
                    ))
            return ms_docs

        else:
            # Question paper
            splits = re.split(self.question_split_pattern, full_text)
            questions_raw = [s for s in splits if clean_ingested_question_text(s)]
            # Crop diagrams
            img_paths = crop_question_screenshots(self.name, self.subject, self.examiner, questions_raw)

            q_docs = []
            for idx, q_text in enumerate(questions_raw):
                cleaned = clean_ingested_question_text(q_text)
                marks = self.extract_mark(q_text, self.marks_pattern)
                q_imgs = img_paths[idx] if idx < len(img_paths) else []

                meta = {
                    "type": "QuestionPaper",
                    "doc_class": "question_paper",
                    "marks": marks,
                    "exam": self.meta_data.get("exam", ""),
                    "exam_type": self.meta_data.get("exam_type", ""),
                    "image_paths": json.dumps(q_imgs) if q_imgs else "[]",
                }
                q_docs.append(Document(page_content=cleaned, metadata=meta))
            return q_docs


class VectorStore:
    """Manages FAISS vector store creation and updates."""

    def __init__(self, name: str) -> None:
        self.vector_database_name = name
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def create_new_database(self, chunks: list[Document]) -> None:
        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        vectorstore.save_local(self.vector_database_name)
        logger.info("Created new vector database: %s", self.vector_database_name)

    def add_to_existing_database(self, chunks: list[Document]) -> None:
        db = FAISS.load_local(self.vector_database_name, self.embedding_model, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
        db.save_local(self.vector_database_name)
        logger.info("Added %d documents to %s", len(chunks), self.vector_database_name)

    def dump_database_to_string(self, subject: str, examiner: str) -> str:
        """Export the full vector database contents as a human-readable text string."""
        if not os.path.exists(self.vector_database_name):
            return f"Vector database '{self.vector_database_name}' does not exist."

        db = FAISS.load_local(self.vector_database_name, self.embedding_model, allow_dangerous_deserialization=True)
        all_docs = list(db.docstore._dict.values())

        lines = [
            "=" * 80,
            f"VECTOR DATABASE CONTENTS: {subject} ({examiner})",
            f"Database Path: {self.vector_database_name}",
            f"Total Ingested Documents: {len(all_docs)}",
            "=" * 80,
            ""
        ]

        for idx, doc in enumerate(all_docs, start=1):
            meta = doc.metadata or {}
            doc_type = meta.get("type") or "Document"
            marks = meta.get("marks", "N/A")
            lines.append("-" * 80)
            lines.append(f"DOCUMENT #{idx} | Type: {doc_type} | Marks: {marks}")
            lines.append(f"Content:\n{doc.page_content.strip()}")
            lines.append("-" * 80)
            lines.append("")

        return "\n".join(lines)


class DatabaseManager:
    """Orchestrates PDF processing and vector database management."""

    def __init__(self, subject: str, examiner: str) -> None:
        self.subject = subject
        self.examiner = examiner

    def store_to_database(self, pdf: PdfFile, database: VectorStore) -> None:
        document = pymupdf.open(pdf.name)
        chunks = pdf.split_document(document)
        document.close()

        if chunks:
            chunks = pdf.add_metadata(chunks)
            try:
                database.add_to_existing_database(chunks)
            except RuntimeError:
                logger.info("Database not found, creating new one")
                database.create_new_database(chunks)

    def add_folder_database(self, folder: str, database_path: str) -> None:
        vdb = VectorStore(database_path)
        doc_type = os.path.basename(folder)

        exam_type = ""
        norm_folder = os.path.normpath(folder)
        parts = norm_folder.split(os.sep)
        try:
            et_idx = next(i for i, p in enumerate(parts) if p.lower() == "exam-types")
            if et_idx + 1 < len(parts):
                exam_type = parts[et_idx + 1]
        except StopIteration:
            pass

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                if not filename.lower().endswith(".pdf"):
                    continue
                logger.info("Processing: %s", file_path)
                pdf_file = PdfFile(file_path, self.subject, self.examiner, doc_type, exam_type=exam_type)
                self.store_to_database(pdf_file, vdb)


def load_vectorstore(subject: str, examiner: str, embedding_model: Any = None) -> FAISS:
    """Load FAISS vector database directly from the data directory."""
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vdb_path = f"data/{subject}/{examiner}/{subject}-{examiner}-vectorDatabase"
    if not os.path.exists(vdb_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vdb_path = os.path.join(root_dir, "data", subject, examiner, f"{subject}-{examiner}-vectorDatabase")

    if not os.path.exists(vdb_path):
        raise FileNotFoundError(f"Vector database not found at {vdb_path}")

    return FAISS.load_local(vdb_path, embedding_model, allow_dangerous_deserialization=True)
