"""Exam marking module for GCSE AI.

Handles mark scheme creation, answer marking, and exam grading
using LLM-based analysis against specification content.
"""

import base64
import logging
import os
from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from config import SubjectConfig
from llm_client import LLMClient
from similarity import SimilarityEngine

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
        example_mark_schemes = self.similarity.find_least_similar_objects(
            objects=self.mark_schemes,
            comparison=question,
            n=self.config.example_ms,
            topic_value=topic,
            marks_value=marks,
        )

        structure = self.llm.invoke(
            self.prompts["create_ms_structure"].format(
                subject=self.subject,
                random_mark_schemes=example_mark_schemes,
            )
        )

        # Get relevant specification info for the question
        if self.specification_text:
            prompt = f"Using the following GCSE {self.subject} specification context, extract the relevant syllabus requirements and information needed to answer this question: '{question}'\n\nSPECIFICATION CONTENT:\n{self.specification_text}\n\nRelevant Information:"
            info = self.llm.invoke(prompt)
        else:
            info_query = self.queries["get_question_related_info"].format(
                question=question
            )
            info = self.llm.invoke_qa(self.spec_qa_chain, info_query)
        logger.info("Retrieved spec info for mark scheme creation")
        logger.debug("Spec info: %s", info)

        mark_scheme = self.llm.invoke(
            self.prompts["create_new_markscheme"].format(
                question=question,
                subject=self.subject,
                structure=structure,
                info=info,
                marks=marks,
            )
        )
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
        # Identify the command word for marking guidance
        command_word = self.llm.invoke(
            self.prompts["get_command_word"].format(question=question)
        )

        # Get marking advice for this type of question
        advice_query = (
            f"Give marking advice for {marks} mark '{command_word}' questions"
        )
        advice = self.llm.invoke_qa(self.ms_qa_chain, advice_query)
        logger.debug("Marking advice: %s", advice)

        # Format the mark scheme for consistency
        formatted_ms = self.llm.invoke(
            self.prompts["format_mark_scheme"].format(mark_scheme=mark_scheme)
        )

        # Mark the answer
        result = self.llm.invoke(
            self.prompts["mark_answer"].format(
                subject=self.subject,
                mark_scheme=formatted_ms,
                answer=answer,
                marks=marks,
            )
        )
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

    def mark_question_from_files(self) -> Optional[str]:
        """Mark a question from user-provided files.

        Reads question, answer, and optional mark scheme from the user_data
        directory. Supports .jpg, .png, .jpeg images and .txt, .md text files.

        Returns:
            Marking feedback string, or None if required files are missing.
        """
        question_data: dict[str, str] = {"question": "", "answer": "", "ms": ""}

        for data_type in question_data:
            directory = f"user_data/{data_type}"
            if not os.path.exists(directory):
                logger.warning("Directory not found: %s", directory)
                continue

            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                ext = os.path.splitext(filename)[1].lower()

                if ext in (".jpg", ".png", ".jpeg"):
                    text = self.image_to_text(filepath)
                elif ext in (".txt", ".md"):
                    with open(filepath, "r") as file:
                        text = file.read()
                else:
                    logger.warning("Unsupported %s file format: %s", data_type, ext)
                    continue

                question_data[data_type] += text

        if not question_data["question"]:
            logger.error("No question file found in user_data/question/")
            return None

        if not question_data["answer"]:
            logger.error("No answer file found in user_data/answer/")
            return None

        # Determine exam type and marks
        exam_type = self.identify_exam_type(question_data["question"])
        marks = self.get_marks_from_question(question_data["question"])

        # Generate mark scheme if not provided
        if not question_data["ms"]:
            question_data["ms"] = self.create_mark_scheme(
                question_data["question"], marks, exam_type
            )

        # Generate model answer for reference
        model_answer = self.generate_model_answer(
            question_data["question"], question_data["ms"]
        )

        logger.info("Mark scheme:\n%s", question_data["ms"])
        logger.info("Model answer:\n%s", model_answer)

        return self.mark_answer(
            question_data["answer"],
            question_data["ms"],
            question_data["question"],
            marks,
        )

    def mark_exam(
        self,
        questions: list[dict],
        topic: str,
    ) -> list[dict]:
        """Mark a full exam's worth of student answers.

        Args:
            questions: List of dicts with 'parent_description', 'question',
                      'marks', and 'answer' keys.
            topic: The exam topic.

        Returns:
            List of dicts with marking results.
        """
        results = []

        for question in questions:
            full_question = (
                question.get("parent_description", "") + "\n" + question["question"]
            )
            ms = self.create_mark_scheme(full_question, question["marks"], topic)
            mark = self.mark_answer(
                question["answer"], ms, full_question, question["marks"]
            )

            result = {
                "question": question["question"],
                "mark_scheme": ms,
                "student_answer": question["answer"],
                "result": mark,
            }
            results.append(result)

            logger.info("Mark scheme:\n%s", ms)
            logger.info("Student answer:\n%s", question["answer"])
            logger.info("Result:\n%s", mark)

        return results
