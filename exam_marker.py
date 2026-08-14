"""Exam marking module for GCSE AI.

Handles mark scheme creation, answer marking, and exam grading
using LLM-based analysis against specification content.
"""

import base64
import concurrent.futures
import json
import logging
import os
import re
from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

from config import SubjectConfig
from llm_client import LLMClient
from similarity import SimilarityEngine
from exam_generator import LocalSimilarityEngine
from exam_segmenter import ExamSegmenter

logger = logging.getLogger(__name__)


class ExamMarker:
    """Marks student answers against mark schemes using LLM analysis.

    Can create mark schemes from questions, generate model answers,
    and grade student responses from text or image input.
    """

    def __init__(
        self,
        config: SubjectConfig,
        llm_client: LLMClient,
        similarity_engine: SimilarityEngine,
        questions: list[dict],
        mark_schemes: list[dict],
        prompts: dict[str, str],
        queries: dict[str, str],
        spec_qa_chain: RetrievalQA,
        ms_qa_chain: RetrievalQA,
        subject: str,
        examiner: str,
        embedding_model: Any = None,
        specification_text: str = "",
    ):
        self.config = config
        self.llm = llm_client
        self.similarity = similarity_engine
        self.questions = questions
        self.mark_schemes = mark_schemes
        self.prompts = prompts
        self.queries = queries
        self.spec_qa_chain = spec_qa_chain
        self.ms_qa_chain = ms_qa_chain
        self.subject = subject
        self.examiner = examiner
        self.embedding_model = embedding_model
        self.specification_text = specification_text

        # Fast local embedding engine for similarity matching
        fallback_dim = getattr(config, "fallback_embedding_dim", 384)
        self.local_similarity = LocalSimilarityEngine(
            embedding_model, llm_client, fallback_dim=fallback_dim
        )

        # Cache for specification trees
        self.spec_trees_cache: dict[str, dict] = {}

        # Cache exam types to avoid recomputing
        self._exam_types: Optional[list[str]] = None

    @property
    def exam_types(self) -> list[str]:
        """Cached list of unique exam types from questions."""
        if self._exam_types is None:
            seen: set[str] = set()
            self._exam_types = []
            for question in self.questions:
                topic = question["topic"]
                if topic not in seen:
                    seen.add(topic)
                    self._exam_types.append(topic)
        return self._exam_types

    def _get_spec_tree_cached(self, topic: str, exam_type: str = "") -> dict:
        """Fetch specification tree by looking up specification_summary.json or falling back to LLM parsing."""
        et = exam_type if exam_type else self.examiner
        cache_key = f"{et}-{topic}"
        if cache_key in self.spec_trees_cache:
            return self.spec_trees_cache[cache_key]

        # Fast lookup from specification_summary.json
        if self.subject and self.examiner:
            sum_path = f"data/{self.subject}/{self.examiner}/{self.subject}-{self.examiner}-specification_summary.json"
            if os.path.exists(sum_path):
                try:
                    with open(sum_path, "r", encoding="utf-8") as f:
                        summary_data = json.load(f)

                    et_target = et.strip().lower()
                    et_items = summary_data.get("exam_types", [])
                    matched_et = None
                    for item in et_items:
                        if et_target and et_target in str(item.get("exam_type", "")).strip().lower():
                            matched_et = item
                            break
                    if not matched_et and et_items:
                        matched_et = et_items[0]

                    if matched_et:
                        topic_target = topic.strip().lower()
                        for t_item in matched_et.get("topics", []):
                            t_name = str(t_item.get("topic", "")).strip()
                            if topic_target in t_name.lower() or t_name.lower() in topic_target:
                                spec_code = t_item.get("spec_code", topic)
                                raw_subs = t_item.get("subtopics", [topic])
                                subtopics = []
                                for s in raw_subs:
                                    if isinstance(s, str):
                                        subtopics.append(s.strip())
                                    elif isinstance(s, dict) and s.get("name"):
                                        subtopics.append(str(s.get("name")).strip())
                                if not subtopics:
                                    subtopics = [topic]

                                tree = {"spec_code": spec_code, "subtopics": subtopics}
                                self.spec_trees_cache[cache_key] = tree
                                logger.info("Found spec summary lookup for marking %s: %s", cache_key, spec_code)
                                return tree
                except Exception as e:
                    logger.warning("Failed summary lookup for marking %s: %s", cache_key, e)

        tree = {"spec_code": topic, "subtopics": [topic]}
        self.spec_trees_cache[cache_key] = tree
        return tree

    def create_mark_scheme(self, question: str, marks: int, topic: str) -> str:
        """Create a mark scheme for a given question.

        Uses example mark schemes from past papers and specification content
        to generate a new mark scheme.

        Args:
            question: The question text.
            marks: Number of marks for the question.
            topic: The exam topic.

        Returns:
            The generated mark scheme text.
        """
        print(f"\033[94m  [Mark Scheme Creation] Step 1: Finding exemplar mark schemes for topic '{topic}' ({marks} marks)...\033[0m")
        example_mark_schemes = self.similarity.find_least_similar_objects(
            objects=self.mark_schemes,
            comparison=question,
            n=self.config.example_ms,
            topic_value=topic,
            marks_value=marks,
        )
        print(f"\033[96m    -> Selected {len(example_mark_schemes)} exemplar mark schemes.\033[0m")

        print(f"\033[94m  [Mark Scheme Creation] Step 2: Extracting structural pattern via LLM...\033[0m")
        structure_prompt = self.prompts["create_ms_structure"].format(
            subject=self.subject,
            random_mark_schemes=example_mark_schemes,
        )
        structure = self.llm.invoke(structure_prompt)
        print(f"\033[96m    -> Extracted Structure Template:\n{structure}\033[0m")

        # Retrieve specification tree hierarchy for topic context
        print(f"\033[94m  [Mark Scheme Creation] Step 3: Resolving specification tree hierarchy...\033[0m")
        spec_tree = self._get_spec_tree_cached(topic, self.examiner)
        spec_code = spec_tree.get("spec_code", topic)
        subtopics = ", ".join(spec_tree.get("subtopics", [topic]))
        print(f"\033[96m    -> Spec Code: {spec_code} | Subtopics: {subtopics}\033[0m")

        # Get relevant specification info for the question
        print(f"\033[94m  [Mark Scheme Creation] Step 4: Extracting syllabus requirements...\033[0m")
        if self.specification_text:
            prompt = (
                f"Using the following GCSE {self.subject} specification context (Code/Section: {spec_code}, Subtopics: {subtopics}), "
                f"extract the relevant syllabus requirements and information needed to answer this question: '{question}'\n\n"
                f"SPECIFICATION CONTENT:\n{self.specification_text}\n\nRelevant Information:"
            )
            info = self.llm.invoke(prompt)
        else:
            info_query = self.queries["get_question_related_info"].format(
                question=question
            )
            info = self.llm.invoke_qa(self.spec_qa_chain, info_query)
        print(f"\033[96m    -> Syllabus Information Retrieved:\n{info}\033[0m")

        print(f"\033[94m  [Mark Scheme Creation] Step 5: Generating final mark scheme...\033[0m")
        final_ms_prompt = self.prompts["create_new_markscheme"].format(
            question=question,
            subject=self.subject,
            structure=structure,
            info=info,
            marks=marks,
        )
        mark_scheme = self.llm.invoke(final_ms_prompt)
        print(f"\033[92m  [Mark Scheme Creation Complete]:\n{mark_scheme}\033[0m")
        return mark_scheme

    def mark_answer(
        self, answer: str, mark_scheme: str, question: str, marks: int
    ) -> str:
        """Mark a student's answer against a mark scheme.

        Args:
            answer: The student's answer text.
            mark_scheme: The mark scheme to mark against.
            question: The original question text.
            marks: Maximum marks available.

        Returns:
            Marking feedback with score.
        """
        print(f"\033[95m\n[Marking Execution] Starting marking workflow...\033[0m")
        print(f"\033[94m  * Question:\033[0m {question}")
        print(f"\033[94m  * Student Answer:\033[0m {answer}")
        print(f"\033[94m  * Allocated Marks:\033[0m {marks}")

        if mark_scheme and mark_scheme.strip():
            print(f"\033[96m  [Official Mark Scheme Used]: Using user_data mark scheme directly.\033[0m")
            formatted_ms = mark_scheme
        else:
            print(f"\033[94m  Step 1: Reformatting mark scheme for evaluation...\033[0m")
            format_ms_prompt = self.prompts["format_mark_scheme"].format(mark_scheme=question)
            formatted_ms = self.llm.invoke(format_ms_prompt)

        # Mark the student answer against the mark scheme
        print(f"\033[94m  Evaluating student answer against mark scheme...\033[0m")
        mark_prompt = self.prompts["mark_answer"].format(
            subject=self.subject,
            mark_scheme=formatted_ms,
            answer=answer,
            marks=marks,
        )
        result = self.llm.invoke(mark_prompt)
        print(f"\033[92m  [Marking Result]:\n{result}\033[0m")
        return result

    def get_marks_from_question(self, question: str) -> int:
        """Extract the mark allocation from a question text.

        Args:
            question: The question text containing mark information.

        Returns:
            The number of marks as an integer.
        """
        prompt = self.prompts["extract_marks"].format(question=question)
        result = self.llm.invoke(prompt)
        try:
            return int(result.strip())
        except ValueError:
            logger.warning("Could not parse marks from: %s", result)
            raise ValueError(f"Failed to extract marks from question. LLM returned: {result}")

    def identify_exam_type(self, question: str) -> str:
        """Identify which exam topic/type a question belongs to.

        Args:
            question: The question text.

        Returns:
            The exam type string.
        """
        prompt = self.prompts["exam_type_of_question"].format(
            exam_types=self.exam_types, question=question
        )
        if self.specification_text:
            full_prompt = f"Using the following GCSE {self.subject} specification:\n\nSPECIFICATION:\n{self.specification_text}\n\nTask: {prompt}"
            return self.llm.invoke(full_prompt)
        else:
            return self.llm.invoke_qa(self.spec_qa_chain, prompt)

    def generate_model_answer(self, question: str, mark_scheme: str) -> str:
        """Generate a model answer for a question.

        Args:
            question: The question text.
            mark_scheme: The mark scheme to target.

        Returns:
            A model answer targeting full marks.
        """
        return self.llm.invoke(
            self.prompts["model_answer"].format(
                ms=mark_scheme, question=question
            )
        )

    def image_to_text(self, img_path: str) -> str:
        """Extract text from an image of an exam page.

        Args:
            img_path: Path to the image file.

        Returns:
            Extracted text content.
        """
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = self.prompts["read_user_exam_page"]
        result = self.llm.invoke_with_image(prompt, img_b64)
        logger.info("Extracted text from image: %s", img_path)
        return result

    def pdf_to_text(self, filepath: str) -> str:
        """Extract text content from a PDF file, skipping cover/instruction pages.

        Args:
            filepath: Path to the PDF file.

        Returns:
            Extracted text string.
        """
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text_parts = []
            # Skip first page (cover page with exam board info/instructions) if multiple pages
            start_page = 1 if len(reader.pages) > 1 else 0
            for page in reader.pages[start_page:]:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            full_text = "\n\n".join(text_parts)
            logger.info("Extracted text from PDF file: %s (%d pages)", filepath, len(reader.pages))
            return full_text
        except Exception as e:
            logger.warning("pypdf extraction failed for %s: %s. Trying PyPDFLoader fallback...", filepath, e)
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                start_idx = 1 if len(docs) > 1 else 0
                return "\n\n".join(doc.page_content for doc in docs[start_idx:])
            except Exception as ex:
                logger.error("Failed to extract text from PDF %s: %s", filepath, ex)
                return ""

    def _fallback_llm_store_search(self, canonical_code: str, question_text: str, store_type: str, unmatched_items: list) -> Optional[dict]:
        """Invoke fallback LLM search to find a matching store item when code lookup fails."""
        try:
            prompt_path = os.path.join("prompts", "find_matching_store_item.txt")
            if not os.path.exists(prompt_path):
                return None
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            unmatched_json = json.dumps([{"code": item.get("canonical_code") or item.get("raw_code"), "content": item.get("content")} for item in unmatched_items if item.get("content")], indent=2)

            prompt = (prompt_template
                      .replace("{store_type}", store_type)
                      .replace("{canonical_code}", canonical_code)
                      .replace("{question_text}", question_text[:1000])
                      .replace("{unmatched_items_json}", unmatched_json[:10000]))

            res = self.llm.invoke(prompt)
            cleaned = res.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            data = json.loads(cleaned)
            if data.get("matched") and data.get("matched_content"):
                return {"content": data["matched_content"]}
        except Exception as e:
            logger.error("Fallback LLM store search error: %s", e)
        return None

    def mark_question_from_files(self) -> Optional[str]:
        """Mark a question or full paper from user_data/ directory.

        Converts all documents in user_data/ to text, uses 1 LLM call per file
        for item extraction and document classification, normalizes question codes
        with 1 post-processing LLM call, and marks all questions in parallel.

        Returns:
            Marking feedback markdown report string, or None if user_data is empty.
        """
        from user_data_ingestion import UserDataIngestion

        ingestion = UserDataIngestion(llm_client=self.llm)
        stores = ingestion.ingest_user_data("user_data")

        q_store = stores.get("QuestionStore", {})
        ms_store = stores.get("MarkSchemeStore", {})
        ans_store = stores.get("AnswerStore", {})

        if not q_store:
            logger.error("No question items found in user_data/")
            return None

        segmented_questions = []

        for canon_code, q_entry in q_store.items():
            ans_entry = ans_store.get(canon_code)
            ms_entry = ms_store.get(canon_code)

            answer_fallback_used = False
            ms_fallback_used = False

            # Fallback LLM search if answer missing for this question code
            if not ans_entry and ans_store:
                logger.info("Canonical code %s missing answer. Invoking fallback LLM search...", canon_code)
                print(f"\033[93m  [LLM Fallback Match]: Searching AnswerStore for missing answer to Question {canon_code}...\033[0m")
                matched_ans = self._fallback_llm_store_search(canon_code, q_entry["content"], "answer", list(ans_store.values()))
                if matched_ans:
                    ans_entry = matched_ans
                    answer_fallback_used = True
                    print(f"\033[92m  [LLM Fallback Match Success]: Matched answer for Question {canon_code}!\033[0m")

            # Fallback LLM search if mark scheme missing for this question code
            if not ms_entry and ms_store:
                logger.info("Canonical code %s missing mark scheme. Invoking fallback LLM search...", canon_code)
                print(f"\033[93m  [LLM Fallback Match]: Searching MarkSchemeStore for missing mark scheme to Question {canon_code}...\033[0m")
                matched_ms = self._fallback_llm_store_search(canon_code, q_entry["content"], "mark scheme", list(ms_store.values()))
                if matched_ms:
                    ms_entry = matched_ms
                    ms_fallback_used = True
                    print(f"\033[92m  [LLM Fallback Match Success]: Matched mark scheme for Question {canon_code}!\033[0m")

            ms_text = ms_entry["content"] if ms_entry else ""
            if ms_text:
                ms_source = "Official Mark Scheme (user_data via Fallback Match)" if ms_fallback_used else "Official Mark Scheme (user_data)"
            else:
                ms_source = "LLM Generated"

            segmented_questions.append({
                "label": f"Question {canon_code}",
                "code": canon_code,
                "question": q_entry["content"],
                "student_answer": ans_entry["content"] if ans_entry else "",
                "mark_scheme": ms_text,
                "marks": q_entry.get("marks") or 1,
                "mark_scheme_source": ms_source,
                "answer_fallback_used": answer_fallback_used,
                "ms_fallback_used": ms_fallback_used,
            })

        logger.info("Ingested and prepared %d questions for marking.", len(segmented_questions))
        print(f"\033[95m\n[Full Exam Marking Engine] Prepared {len(segmented_questions)} questions from flat user_data/ stores.\033[0m")
        print(f"\033[95m  * Executing parallel evaluation across worker threads...\033[0m")

        marking_results = self.mark_exam(segmented_questions, topic=self.examiner)

        from generate_content import save_marking_report_md, format_marking_as_markdown
        save_marking_report_md(
            subject=self.subject,
            examiner=self.examiner,
            topic="Full Exam Paper",
            marking_results=marking_results,
            llm_client=self.llm
        )
        return format_marking_as_markdown(
            subject=self.subject,
            examiner=self.examiner,
            topic="Full Exam Paper",
            marking_results=marking_results,
            llm_client=self.llm
        )

    def _mark_single_task(self, task: dict) -> dict:
        """Executes marking for a single question task in a worker thread."""
        idx = task["idx"]
        q_dict = task["question_dict"]
        topic = task["topic"]

        q_text = q_dict.get("question") or q_dict.get("text") or ""
        parent_desc = q_dict.get("parent_description", "")
        full_question = (parent_desc + "\n" + q_text).strip() if parent_desc else q_text.strip()
        
        marks = q_dict.get("marks")
        if marks is None or not isinstance(marks, int):
            try:
                marks = self.get_marks_from_question(full_question)
            except Exception:
                marks = 1

        student_answer = q_dict.get("answer") or q_dict.get("student_answer") or ""
        ms = q_dict.get("mark_scheme") or q_dict.get("ms")

        if not ms:
            ms = self.create_mark_scheme(full_question, marks, topic)

        mark_feedback = self.mark_answer(student_answer, ms, full_question, marks)

        # Try to extract awarded mark number [x/N] from feedback
        awarded_marks = None
        match = re.search(r"\[\s*(\d+)\s*/\s*\d+\s*\]", mark_feedback)
        if match:
            try:
                awarded_marks = int(match.group(1))
            except ValueError:
                awarded_marks = None

        return {
            "idx": idx,
            "label": q_dict.get("label") or f"Question {idx + 1}",
            "code": q_dict.get("code"),
            "question": full_question,
            "mark_scheme": ms,
            "student_answer": student_answer,
            "result": mark_feedback,
            "marks": marks,
            "awarded_marks": awarded_marks,
            "mark_scheme_source": ("Official Mark Scheme (user_data via Fallback Match)" if q_dict.get("ms_fallback_used") else "Official Mark Scheme (user_data)") if (q_dict.get("mark_scheme") and q_dict.get("mark_scheme").strip()) else "LLM Generated",
            "answer_fallback_used": q_dict.get("answer_fallback_used", False),
            "ms_fallback_used": q_dict.get("ms_fallback_used", False),
        }

    def mark_exam(
        self,
        questions: list[dict],
        topic: str,
    ) -> list[dict]:
        """Mark a full exam's worth of student answers in parallel using thread pool.

        Args:
            questions: List of dicts with 'parent_description', 'question'/'text',
                       'marks', and 'answer'/'student_answer' keys.
            topic: The exam topic.

        Returns:
            List of dicts with marking results, ordered by original question sequence.
        """
        if not questions:
            return []

        tasks = [
            {
                "idx": i,
                "question_dict": q,
                "topic": topic,
            }
            for i, q in enumerate(questions)
        ]

        max_workers = getattr(self.config, "max_parallel_workers", 10)
        if not isinstance(max_workers, int) or max_workers <= 0:
            max_workers = 10
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
            future_to_task = {executor.submit(self._mark_single_task, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logger.error("Marking task failed for question idx %d: %s", task["idx"], e, exc_info=True)
                    raise

        # Restore original order
        results.sort(key=lambda x: x["idx"])
        for res in results:
            res.pop("idx", None)

        return results

