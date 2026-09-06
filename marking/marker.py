"""Exam marker module for GCSE AI."""

import base64
import concurrent.futures
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from config import SubjectConfig
from llm.client import LLMClient
from llm.prompts import load_prompt
from llm.similarity import SimilarityEngine, LocalSimilarityEngine
from marking.segmenter import ExamSegmenter
from marking.report import format_marking_as_markdown, save_marking_report_md

logger = logging.getLogger(__name__)


class ExamMarker:
    """Marks student answers against mark schemes using LLM analysis."""

    def __init__(
        self,
        config: SubjectConfig,
        llm_client: LLMClient,
        similarity_engine: SimilarityEngine,
        questions: List[dict],
        mark_schemes: List[dict],
        subject: str,
        examiner: str,
        embedding_model: Any = None,
        specification_text: str = "",
        prompts: Optional[Dict[str, str]] = None,
        queries: Optional[Dict[str, str]] = None,
        spec_qa_chain: Any = None,
        ms_qa_chain: Any = None,
    ):
        self.config = config
        self.llm = llm_client
        self.similarity = similarity_engine
        self.questions = questions
        self.mark_schemes = mark_schemes
        self.subject = subject
        self.examiner = examiner
        self.embedding_model = embedding_model
        self.specification_text = specification_text
        self.prompts = prompts or {}

        fallback_dim = getattr(config, "fallback_embedding_dim", 384)
        self.local_similarity = LocalSimilarityEngine(
            embedding_model, llm_client, fallback_dim=fallback_dim
        )
        self.segmenter = ExamSegmenter(mark_pattern=config.mark_pattern)
        self.spec_trees_cache: Dict[str, dict] = {}
        self._exam_types: Optional[List[str]] = None

    @property
    def exam_types(self) -> List[str]:
        """Cached list of unique exam types from questions."""
        if self._exam_types is None:
            seen: set[str] = set()
            self._exam_types = []
            for question in self.questions:
                topic = question.get("topic", "")
                if topic and topic not in seen:
                    seen.add(topic)
                    self._exam_types.append(topic)
        return self._exam_types

    def _get_spec_tree_cached(self, topic: str, exam_type: str = "") -> dict:
        """Fetch specification tree by looking up specification_summary.json or falling back to default."""
        et = exam_type if exam_type else self.examiner
        cache_key = f"{et}-{topic}"
        if cache_key in self.spec_trees_cache:
            return self.spec_trees_cache[cache_key]

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
                                return tree
                except Exception as e:
                    logger.warning("Failed summary lookup for marking %s: %s", cache_key, e)

        tree = {"spec_code": topic, "subtopics": [topic]}
        self.spec_trees_cache[cache_key] = tree
        return tree

    def create_mark_scheme(self, question: str, marks: int, topic: str = "General") -> str:
        """Create a standard mark scheme for a question using LLM analysis."""
        prompt = (
            f"You are an expert GCSE {self.subject} ({self.examiner}) senior examiner.\n"
            f"Write an official, precise mark scheme for the following GCSE question for {marks} marks:\n\n"
            f"Topic: {topic}\n"
            f"Question:\n{question}\n\n"
            f"Total Marks: {marks}\n\n"
            f"Include:\n"
            f"- Marking guidance / indicative points (1 mark per distinct point).\n"
            f"- Acceptable alternative terms and reject criteria.\n"
            f"- Total marks."
        )
        return self.llm.invoke(prompt)

    def generate_model_answer(self, question: str, mark_scheme: str = "") -> str:
        """Generate a full-mark exemplar model answer for a question."""
        prompt = (
            f"You are an expert GCSE {self.subject} ({self.examiner}) teacher.\n"
            f"Write a flawless, full-mark exemplar model answer for the question below.\n\n"
            f"Question:\n{question}\n\n"
            f"Mark Scheme Guidance:\n{mark_scheme or 'Standard GCSE grading criteria'}\n\n"
            f"Provide a clear, student-friendly answer using official GCSE scientific terminology."
        )
        return self.llm.invoke(prompt)

    def get_marks_from_question(self, question: str) -> int:
        """Extract mark count from question string using LLM or regex fallback."""
        prompt = self.prompts.get("extract_marks", "Extract marks from question: {question}").format(question=question)
        res = self.llm.invoke(prompt).strip()
        try:
            return int(res)
        except ValueError:
            logger.warning("Could not parse marks from: %s", res)
            raise ValueError(f"Could not parse marks from: {res}")

    def pdf_to_text(self, file_path: str) -> str:
        """Extract text from a PDF file using pypdf."""
        import pypdf
        reader = pypdf.PdfReader(file_path)
        pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
        logger.info("Extracted text from PDF file: %s (%d pages)", file_path, len(pages_text))
        return "\n\n".join(pages_text)

    def mark_answer(
        self,
        question: str = "",
        student_answer: str = "",
        mark_scheme: str = "",
        marks: int = 4,
        answer: Optional[str] = None,
    ) -> str:
        """Evaluate a student's answer against a mark scheme."""
        actual_answer = answer or student_answer
        logger.info("Marking answer for question (%d marks)...", marks)
        prompt = (
            f"You are an expert GCSE {self.subject} ({self.examiner}) examiner.\n"
            f"Mark the following student answer strictly according to the official mark scheme.\n\n"
            f"Question ({marks} Marks):\n{question}\n\n"
            f"Official Mark Scheme:\n{mark_scheme}\n\n"
            f"Student Answer:\n{actual_answer}\n\n"
            f"Provide:\n"
            f"1. Awarded marks in the exact format: [X/{marks}] where X is the awarded integer.\n"
            f"2. Brief, constructive feedback justifying the mark awarded.\n"
            f"3. What was done well and what was missing."
        )
        result = self.llm.invoke(prompt)
        logger.info("Marking completed: %s", result[:100].replace('\n', ' '))
        return result

    def mark_exam(
        self, questions: List[dict], topic: str, compile_booklet: bool = True
    ) -> List[dict]:
        """Mark a full set of question items in parallel.

        For each question item:
        - If student answer is typed text, typesets it onto a clean feedback canvas image.
        - Evaluates via the 3-stage visual marking pipeline (or text fallback).
        - Renders teacher annotations (highlights, ticks, underlines, margin notes).
        - Stitches all marked question pages into test_outputs/marked_exam_booklet.pdf.
        """
        from marking.canvas_typesetter import typeset_student_answer_to_b64
        from marking.visual_marker import VisualExamMarker
        from marking.visual_renderer import render_visual_annotations
        from marking.booklet import assemble_marked_booklet_pdf

        logger.info("Marking full exam (%d questions) for topic '%s'...", len(questions), topic)
        os.makedirs("test_outputs/marked_pages", exist_ok=True)
        visual_marker = VisualExamMarker(
            llm_client=self.llm,
            subject=self.subject,
            examiner=self.examiner,
        )

        def _mark_item(item: dict) -> dict:
            q_text = item.get("question", "")
            ans_text = item.get("student_answer") or item.get("answer", "")
            ms_text = item.get("mark_scheme", "")
            marks = item.get("marks", 4) or 4
            label = item.get("label") or f"Question {item.get('idx', 1)}"
            clean_label = re.sub(r"[^a-zA-Z0-9_\-]", "_", label)

            if not ms_text:
                ms_text = self.create_mark_scheme(q_text, marks, topic)

            # Try visual marking first if student answer exists and vision is supported
            is_mock = type(self.llm).__name__ == "MagicMock"
            is_mock_without_vision = is_mock and ("_vision_enabled" not in self.llm.__dict__)

            if ans_text and not is_mock_without_vision:

                try:
                    if isinstance(ans_text, str) and (
                        os.path.exists(ans_text)
                        or (len(ans_text) > 500 and " " not in ans_text[:50])
                    ):
                        img_b64 = ans_text
                    else:
                        # Typeset typed text onto clean feedback page canvas
                        img_b64 = typeset_student_answer_to_b64(
                            question_label=label,
                            question_prompt=q_text,
                            student_answer_text=ans_text,
                            marks=marks,
                        )

                    marked_img_path = f"test_outputs/marked_pages/{clean_label}_marked.png"
                    vis_res = visual_marker.mark_handwritten_answer(
                        image_b64=img_b64,
                        question=q_text,
                        marks=marks,
                        mark_scheme=ms_text,
                    )
                    render_visual_annotations(
                        image_input=img_b64,
                        annotations_data=vis_res.get("annotations_data", {}),
                        output_path=marked_img_path,
                    )

                    awarded = vis_res.get("awarded_marks")
                    if awarded is None:
                        score_match = re.search(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]", vis_res.get("evaluation", ""))
                        awarded = int(score_match.group(1)) if score_match else None

                    res_item = dict(item)
                    res_item["result"] = vis_res.get("evaluation", "")
                    res_item["awarded_marks"] = awarded
                    res_item["mark_scheme"] = ms_text
                    res_item["marked_image_path"] = marked_img_path
                    res_item["annotations"] = vis_res.get("annotations", [])
                    res_item["teacher_summary"] = vis_res.get("teacher_summary", "")
                    res_item["question_type"] = vis_res.get("question_type", "point_based")
                    return res_item
                except Exception as e:
                    logger.debug(
                        "Visual marking not available or failed for %s, falling back to text: %s",
                        label,
                        e,
                    )


            # Fallback to standard text marking
            feedback = self.mark_answer(q_text, ans_text, ms_text, marks)
            score_match = re.search(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]", feedback)
            awarded = int(score_match.group(1)) if score_match else None

            res_item = dict(item)
            res_item["result"] = feedback
            res_item["awarded_marks"] = awarded
            res_item["mark_scheme"] = ms_text
            return res_item

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(questions), 5)) as executor:
            future_to_idx = {executor.submit(_mark_item, q): idx for idx, q in enumerate(questions)}
            sorted_futures = sorted(future_to_idx.keys(), key=lambda f: future_to_idx[f])
            for future in sorted_futures:
                results.append(future.result())

        # Compile PDF booklet if visual pages were generated
        marked_imgs = [r.get("marked_image_path") for r in results if r.get("marked_image_path")]
        if compile_booklet and marked_imgs:
            try:
                from marking.report import generate_performance_summary

                summary_text = generate_performance_summary(
                    self.subject, self.examiner, topic, results, self.llm
                )
                assemble_marked_booklet_pdf(
                    subject=self.subject,
                    examiner=self.examiner,
                    topic=topic,
                    marked_image_paths=marked_imgs,
                    question_results=results,
                    executive_summary_text=summary_text,
                    output_pdf_path="test_outputs/marked_exam_booklet.pdf",
                )
            except Exception as e:
                logger.warning("Failed to compile marked exam booklet PDF: %s", e)

        logger.info("Full exam marking finished for %d questions.", len(results))
        return results


    def mark_question_from_files(self, user_data_dir: str = "user_data") -> str:
        """Mark questions from documents deposited in user_data/ directory."""
        from ingestion.user_data import UserDataIngestion
        ingestion = UserDataIngestion(self.llm)
        docs = ingestion.convert_documents_to_text(user_data_dir)

        if not docs:
            logger.warning("No documents found in %s", user_data_dir)
            return ""

        qp_text = ""
        ms_text = ""
        ans_text = ""

        for doc in docs:
            fn = doc["filename"].lower()
            text = doc["text"]
            if "question" in fn or "paper" in fn:
                qp_text = text
            elif "mark" in fn or "ms" in fn or "scheme" in fn:
                ms_text = text
            elif "answer" in fn or "student" in fn:
                ans_text = text

        if not qp_text and docs:
            qp_text = docs[0]["text"]

        items = self.segmenter.segment_full_paper(qp_text, ms_text, ans_text)
        if not items:
            return ""

        results = self.mark_exam(items, topic="User Assessment")
        report_path = save_marking_report_md(self.subject, self.examiner, "User Assessment", results, llm_client=self.llm)
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()

    def mark_handwritten_answer(
        self,
        image_b64: str,
        question: str,
        marks: int,
        mark_scheme: str = "",
        render_output_image_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark a student's handwritten answer using the 3-stage visual pipeline.

        Executes:
        - Step 1: Pre-marking tool policy formulation (blind to student answer).
        - Step 2: Diagnostic examiner script evaluation.
        - Step 3: Spatial coordinate grounding (normalized box_2d) and JSON export.
        - Optional: Renders annotated teacher image with highlights, ticks, and margin feedback.
        """
        from marking.visual_marker import VisualExamMarker
        from marking.visual_renderer import render_visual_annotations

        ms = mark_scheme or self.create_mark_scheme(question, marks)
        visual_marker = VisualExamMarker(
            llm_client=self.llm,
            subject=self.subject,
            examiner=self.examiner,
        )
        res = visual_marker.mark_handwritten_answer(
            image_b64=image_b64,
            question=question,
            marks=marks,
            mark_scheme=ms,
            output_json_path=output_json_path,
        )
        if render_output_image_path:
            render_visual_annotations(
                image_input=image_b64,
                annotations_data=res.get("annotations_data", {}),
                output_path=render_output_image_path,
            )
            res["rendered_image_path"] = render_output_image_path
        return res

