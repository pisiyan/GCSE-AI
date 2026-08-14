"""Optimized Exam generation module for GCSE AI.

Contains ExamStructureBuilder for creating exam structures from past paper patterns,
and QuestionGenerator for generating individual questions and full exams.
"""

import logging
import os
import random
import re
import json
import concurrent.futures
from typing import Any, Optional, Dict, List

import numpy as np
from langchain_classic.chains import RetrievalQA

from config import SubjectConfig
from llm_client import LLMClient
from similarity import SimilarityEngine

logger = logging.getLogger(__name__)

MAX_STRUCTURE_RETRIES = 50


def is_calculation_content(text: str) -> bool:
    """Helper to detect if a text chunk represents a math or calculation task."""
    text_lower = text.lower()
    keywords = ["calculate", "work out", "determine", "formula", "equation", "show that", "value of", "percentage", "rate", "mean", "ratio"]
    return any(kw in text_lower for kw in keywords)


def is_practical_content(text: str) -> bool:
    """Helper to detect if a text chunk represents a practical investigation or experiment."""
    text_lower = text.lower()
    keywords = ["investigate", "experiment", "graph", "table", "figure", "results", "measure", "apparatus", "method", "practical"]
    return any(kw in text_lower for kw in keywords)


def generate_answer_lines(marks: Any, lines_per_mark: int = 2) -> str:
    """Generate clean, properly formatted answer space lines for students to write their responses.

    Args:
        marks: Available marks for the question.
        lines_per_mark: Number of lines per mark.

    Returns:
        Formatted string containing answer lines.
    """
    try:
        m_val = int(marks)
    except (ValueError, TypeError):
        m_val = 1
    if m_val <= 0:
        m_val = 1

    num_lines = max(2, min(m_val * lines_per_mark, 16))
    line_str = "_" * 68
    return "\n".join(["  " + line_str for _ in range(num_lines)])


def fix_ocr_formatting(text: str) -> str:
    """Reconstruct split words and clean up OCR spacing artifacts from raw PDF text."""
    if not text:
        return ""

    cleaned = text

    # Remove non-standard OCR symbol artifacts and bullet boxes
    cleaned = re.sub(
        r"[\u2022\u25a0\u25a1\u2571\u2572\u274c\u2713\ufffd\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\u2069\u206a-\u206f\u2027\u2219\u25aa\u25ab\u25c6\u25c7\u25a2\u25a3\u25a4\u25a5\u25a6\u25a7\u25a8\u25a9\u25aa\u25ab\u25ac\u25ad\u25ae\u25af\u25b0\u25b1\u25b2\u25b3\u25b4\u25b5\u25b6\u25b7\u25b8\u25b9\u25ba\u25bb\u25bc\u25bd\u25be\u25bf\u25c0\u25c1\u25c2\u25c3\u25c4\u25c5\u25c6\u25c7\u25c8\u25c9\u25ca\u25cb\u25cc\u25cd\u25ce\u25cf\u25d0-\u25ff\uf000-\uf8ff]",
        "",
        cleaned,
    )

    # Reconstruct words split across line breaks (e.g., 'Bloodwor\nms' -> 'Bloodworms')
    cleaned = re.sub(r"(\b[a-zA-Z]{3,})\s*[\r\n]+\s*([a-zA-Z]{1,3}\b)", r"\1\2", cleaned)
    cleaned = re.sub(r"(\b[a-zA-Z]+)\s*-\s*[\r\n]+\s*([a-zA-Z]+)\b", r"\1-\2", cleaned)

    # Reconstruct hyphenated words split by spaces (e.g., 'white - claw' -> 'white-claw')
    cleaned = re.sub(r"(\b[a-zA-Z]+)\s*-\s*([a-zA-Z]+)\b", r"\1-\2", cleaned)

    # Reconstruct single/double letter split words inside lines (e.g., 'sludgewor m' -> 'sludgeworm', 'Figur e' -> 'Figure')
    cleaned = re.sub(r"\b([a-zA-Z]{3,})\s+([b-zB-Z]{1,2})\b", r"\1\2", cleaned)
    cleaned = re.sub(r"\b([b-hj-zB-HJ-Z]{1})\s+([a-zA-Z]{3,})\b", r"\1\2", cleaned)

    # Specific known GCSE terms and glued words that suffer OCR splitting
    glued_fixes = {
        "Bloodwor ms": "Bloodworms", "Bloodwor m": "Bloodworm", "bloodwor ms": "bloodworms", "bloodwor m": "bloodworm",
        "sludgewor ms": "sludgeworms", "sludgewor m": "sludgeworm", "Figur e": "Figure", "figur e": "figure",
        "in vasive": "invasive", "indica tes": "indicates", "indica te": "indicate", "fertil iser": "fertiliser",
        "reac tion": "reaction", "concen tration": "concentration", "tempe rature": "temperature",
        "expe riment": "experiment", "investiga tion": "investigation", "Car bon": "Carbon", "car bon": "carbon",
        r"\bcolourof\b": "colour of", r"\bshownin\b": "shown in", r"\btestingfor\b": "testing for",
        r"\bcanbe\b": "can be", r"\bsolutionof\b": "solution of", r"\bsolutionsof\b": "solutions of",
        r"\bsolutionw as\b": "solution was", r"\bsolutionw\b": "solution", r"\bwateris\b": "water is",
        r"\bsurvivein\b": "survive in", r"\bhaemoglobinin\b": "haemoglobin in", r"\blevelof\b": "level of"
    }
    for wrong, right in glued_fixes.items():
        cleaned = re.sub(wrong, right, cleaned, flags=re.IGNORECASE)

    # Format Multiple Choice options cleanly into bullets if detected (e.g. A opt B opt C opt D opt)
    mc_pattern = r"\bA\s+([^\nB]+?)\s+B\s+([^\nC]+?)\s+C\s+([^\nD]+?)\s+D\s+([^\n]+)"
    match = re.search(mc_pattern, cleaned)
    if match:
        a_opt = match.group(1).strip()
        b_opt = match.group(2).strip()
        c_opt = match.group(3).strip()
        d_opt = match.group(4).strip()
        mc_formatted = f"\n\n  * **A** {a_opt}\n  * **B** {b_opt}\n  * **C** {c_opt}\n  * **D** {d_opt}"
        cleaned = cleaned[:match.start()] + mc_formatted + cleaned[match.end():]

    return cleaned


def clean_question_text(text: str) -> str:
    """Clean raw question content by removing exam board boilerplate, codes, and junk text."""
    if not text:
        return ""

    cleaned = text

    # Remove margin instructions with OCR spacing glitches
    cleaned = re.sub(r"(?i)DO\s*NO?\s*T?\s*WRITE?\s*IN?\s*THIS?\s*(?:AREA|MARGIN|PAGE).*", "", cleaned)
    cleaned = re.sub(r"(?i)\(?\s*Total\s+for\s+Question.*", "", cleaned)
    cleaned = re.sub(r"(?i)\(?\s*Total\s+\d+\s+marks?\s*\)?", "", cleaned)

    # Remove exam paper instructions & boilerplate
    cleaned = re.sub(r"(?i)Answer\s+ALL\s+questions.*", "", cleaned)
    cleaned = re.sub(r"(?i)Write\s+your\s+answers\s+in\s+the\s+spaces\s+provided.*", "", cleaned)
    cleaned = re.sub(r"(?i)Some\s+questions\s+must\s+be\s+answered\s+with\s+a\s+cross.*", "", cleaned)
    cleaned = re.sub(r"(?i)If\s+you\s+change\s+your\s+mind.*", "", cleaned)
    cleaned = re.sub(r"(?i)mark\s+your\s+new\s+answer.*", "", cleaned)
    cleaned = re.sub(r"(?i)\bTurn\s+over\b", "", cleaned)
    cleaned = re.sub(r"(?i)Pearson\s+Edexcel.*", "", cleaned)
    cleaned = re.sub(r"\b[A-Z]\d{4,}[A-Z0-9]*\b", "", cleaned)
    cleaned = re.sub(r"\*\s*[A-Z0-9]{5,}\s*\*", "", cleaned)

    # Remove long dotted or underlined prompt lines
    cleaned = re.sub(r"\.{3,}", "", cleaned)
    cleaned = re.sub(r"_{3,}", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\(\d+\)\s*$", "", cleaned)

    # Reconstruct OCR words & split text
    cleaned = fix_ocr_formatting(cleaned)

    # Collapse multiple blank lines & spaces
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    # Filter out standalone page number lines or single digits
    filtered_lines = []
    for line in lines:
        if line.isdigit() and len(line) <= 2:
            continue
        filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines)
    cleaned = re.sub(r"^(?:Question\s*\d+\s*\(?[a-z]?\)?|\d+\s*\([a-z]\)|\d+\.)\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    return cleaned.strip()


class ExamStructureBuilder:
    """Builds exam structures based on patterns from past papers.

    Analyses historical question papers to determine typical mark distributions
    and creates new exam structures that follow similar patterns.
    """

    def __init__(
        self,
        config: SubjectConfig,
        questions: list[dict],
    ):
        self.config = config
        self.questions = questions

    @staticmethod
    def flatten_marks(marks: Any) -> int:
        """Flatten a potentially nested mark structure into a total mark count.

        Args:
            marks: An int, or a list of ints/lists representing sub-question marks.

        Returns:
            Total marks as an integer, or 0 if marks is None or an unrecognised type.
        """
        if marks is None:
            return 0
        if isinstance(marks, int):
            return marks
        if not isinstance(marks, list):
            return 0
        total = 0
        for item in marks:
            if isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, int):
                        total += sub_item
            elif isinstance(item, int):
                total += item
        return total

    def get_valid_question_options(self, topic: Optional[str] = None) -> list[tuple[int, str]]:
        """Parse all questions at the highest level (individual/parent questions) and record valid mark options.

        Sub-questions / basic questions that are part of a parent question are NOT considered as standalone basic mark options.
        Only basic questions that are not part of a parent question are recorded as basic mark options.

        Returns:
            List of (marks, question_type) tuples present in self.questions.
        """
        options = set()
        for q in self.questions:
            q_type = q.get("type")
            is_parent = (
                q_type == "parent_question"
                or bool(q.get("parent_question_structure"))
                or bool(q.get("sub_questions"))
            )

            if is_parent:
                pqs = q.get("parent_question_structure")
                m = self.flatten_marks(pqs)
                if m > 0:
                    options.add((m, "parent"))
            else:
                # Only basic questions NOT part of a parent question
                m = q.get("marks")
                if m and isinstance(m, int) and m > 0:
                    options.add((m, "basic"))

        # Fallback scan over all questions if type field is missing or unpopulated
        if not options:
            for q in self.questions:
                pqs = q.get("parent_question_structure")
                m = self.flatten_marks(pqs) if pqs else q.get("marks")
                if m and isinstance(m, int) and m > 0:
                    q_t = "parent" if (pqs or q.get("sub_questions")) else "basic"
                    options.add((m, q_t))

        return sorted(list(options), key=lambda x: (x[0], x[1]))

    def get_past_exam_structures(self, topic: Optional[str] = None) -> list[list[list]]:
        """Extract mark structures from past exams.

        Args:
            topic: Optional exam topic / exam_type / subject to filter questions by.

        Returns:
            A list of exams, where each exam is a list of [marks, type] pairs.
        """
        def _extract(filtered_qs: list[dict]) -> list[list[list]]:
            current_exam = ""
            exam_marks: list[list] = []
            all_exams: list[list[list]] = []

            for question in filtered_qs:
                q_t = question.get("type", "")
                if q_t not in ("parent_question", "basic_question"):
                    continue

                q_exam = question.get("exam") or question.get("exam_id") or "default_exam"
                if q_exam != current_exam:
                    if exam_marks:
                        all_exams.append(exam_marks)
                    exam_marks = []
                    current_exam = q_exam

                if q_t == "parent_question":
                    pqs = question.get("parent_question_structure")
                    if pqs is None:
                        continue
                    marks = [self.flatten_marks(pqs), "parent"]
                else:
                    q_marks = question.get("marks")
                    if q_marks is None:
                        continue
                    marks = [q_marks, "basic"]
                exam_marks.append(marks)

            if exam_marks:
                all_exams.append(exam_marks)

            return all_exams

        if topic:
            topic_clean = str(topic).strip().lower()
            relevant = [
                q for q in self.questions
                if str(q.get("topic", "")).strip().lower() == topic_clean
                or str(q.get("exam_type", "")).strip().lower() == topic_clean
                or str(q.get("subject", "")).strip().lower() == topic_clean
            ]
            res = _extract(relevant)
            if res:
                return res

        return _extract(self.questions)

    def get_position_options(self, topic: Optional[str] = None) -> dict[int, set[tuple[int, str]]]:
        """Extract valid question mark/type options per position index (0-indexed) from past exam structures."""
        past_structures = self.get_past_exam_structures(topic)
        pos_options: dict[int, set[tuple[int, str]]] = {}
        for struct in past_structures:
            for idx, (m, t) in enumerate(struct):
                if idx not in pos_options:
                    pos_options[idx] = set()
                pos_options[idx].add((m, t))
        return pos_options

    def extract_block_patterns(self, past_structures: list[list[list]]) -> list[list[list]]:
        """Extract repeating section/block patterns from past paper structures."""
        if not past_structures:
            return []

        counts: dict[tuple, int] = {}
        for struct in past_structures:
            n = len(struct)
            for k in range(2, min(n + 1, 11)):
                for start in range(0, n - k + 1):
                    sub = tuple((m, t) for m, t in struct[start : start + k])
                    counts[sub] = counts.get(sub, 0) + 1

        if not counts:
            return []

        sorted_blocks = sorted(
            counts.keys(),
            key=lambda b: (len(b), counts[b], sum(x[0] for x in b)),
            reverse=True,
        )
        return [[list(item) for item in b] for b in sorted_blocks]

    def _find_candidate_combinations(
        self,
        total_marks: int,
        valid_options: list[tuple[int, str]],
        target_num_questions: Optional[int] = None,
        limit: int = 50,
    ) -> list[list[list]]:
        """Find multiple valid candidate sequences of [marks, type] that sum to total_marks."""
        if total_marks <= 0 or not valid_options:
            return []

        options = list(valid_options)
        min_m = min(o[0] for o in options)
        max_m = max(o[0] for o in options)

        results = []

        def dfs(remaining: int, current_path: list, depth: int):
            if len(results) >= limit:
                return
            if remaining == 0:
                if target_num_questions is None or len(current_path) == target_num_questions:
                    results.append(current_path)
                return
            if depth > 60:
                return

            curr_len = len(current_path)
            if target_num_questions is not None:
                if curr_len >= target_num_questions:
                    return
                rem_qs = target_num_questions - curr_len
                if remaining < rem_qs * min_m or remaining > rem_qs * max_m:
                    return

            fitting = [opt for opt in options if opt[0] <= remaining]
            if not fitting:
                return

            for m, t in fitting:
                dfs(remaining - m, current_path + [[m, t]], depth + 1)
                if len(results) >= limit:
                    break

        dfs(total_marks, [], 0)
        return results

    def _score_and_select_candidate(
        self,
        candidates: list[list[list]],
        past_structures: list[list[list]],
    ) -> Optional[list[list]]:
        """Score candidate structures against past paper patterns and return the highest-scoring candidate."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        transitions: set[tuple[tuple[int, str], tuple[int, str]]] = set()
        pos_counts: dict[int, dict[tuple[int, str], int]] = {}

        for struct in past_structures:
            for idx, (m, t) in enumerate(struct):
                item = (m, t)
                if idx not in pos_counts:
                    pos_counts[idx] = {}
                pos_counts[idx][item] = pos_counts[idx].get(item, 0) + 1

                if idx < len(struct) - 1:
                    next_item = (struct[idx + 1][0], struct[idx + 1][1])
                    transitions.add((item, next_item))

        best_score = -1.0
        best_candidate = candidates[0]

        for cand in candidates:
            score = 0.0
            n = len(cand)
            for i in range(n - 1):
                t_pair = ((cand[i][0], cand[i][1]), (cand[i + 1][0], cand[i + 1][1]))
                if t_pair in transitions:
                    score += 2.0

            for i in range(n):
                item = (cand[i][0], cand[i][1])
                if i in pos_counts and item in pos_counts[i]:
                    score += 1.0 + pos_counts[i][item]

            if score > best_score:
                best_score = score
                best_candidate = cand

        return best_candidate

    def _find_valid_combination(
        self,
        total_marks: int,
        valid_options: list[tuple[int, str]],
        target_num_questions: Optional[int] = None,
        position_options: Optional[dict[int, set[tuple[int, str]]]] = None,
    ) -> Optional[list[list]]:
        """Find a valid sequence of [marks, type] from valid_options that sum exactly to total_marks.

        If target_num_questions is specified, only combinations with that exact question count are returned.
        If position_options is specified, options at step i are restricted to valid (marks, type) at position i.
        """
        if total_marks <= 0 or not valid_options:
            return None

        options = list(valid_options)
        min_m = min(o[0] for o in options)
        max_m = max(o[0] for o in options)

        def dfs(remaining: int, current_path: list, depth: int) -> Optional[list]:
            if remaining == 0:
                if target_num_questions is None or len(current_path) == target_num_questions:
                    return current_path
                return None
            if depth > 60:
                return None

            curr_len = len(current_path)
            if target_num_questions is not None:
                if curr_len >= target_num_questions:
                    return None
                rem_qs = target_num_questions - curr_len
                if remaining < rem_qs * min_m or remaining > rem_qs * max_m:
                    return None

            available = options
            if position_options and curr_len in position_options and position_options[curr_len]:
                pos_allowed = position_options[curr_len]
                pos_filtered = [opt for opt in options if opt in pos_allowed]
                if pos_filtered:
                    available = pos_filtered

            fitting = [opt for opt in available if opt[0] <= remaining]
            if not fitting:
                return None

            random.shuffle(fitting)
            for m, t in fitting:
                res = dfs(remaining - m, current_path + [[m, t]], depth + 1)
                if res is not None:
                    return res
            return None

        return dfs(total_marks, [], 0)

    def build_structure(
        self,
        total_marks: int,
        topic: str,
        num_questions: Optional[int] = None,
    ) -> list:
        """Build an exam structure using past paper pattern matching and 3-tiered fallback.

        Args:
            total_marks: Target total marks for the exam.
            topic: The exam topic or exam_type.
            num_questions: Optional requested number of questions in the exam.

        Returns:
            A list of mark structures [marks, question_type] for each question.
        """
        valid_options = self.get_valid_question_options(topic)
        if not valid_options:
            valid_options = self.get_valid_question_options(None)

        if not valid_options:
            raise ValueError(f"No valid questions found in database for topic '{topic}'.")

        logger.info("Recorded %d valid question mark options: %s", len(valid_options), valid_options)

        past_structures = self.get_past_exam_structures(topic)
        require_position_matching = bool(getattr(self.config, "question_no_importance", False))
        position_options = self.get_position_options(topic) if require_position_matching else None

        # TIER 1: High-Confidence Exact / Scaled Block Tiling
        if past_structures:
            blocks = self.extract_block_patterns(past_structures)
            for struct in past_structures:
                if struct and struct not in blocks:
                    blocks.append(struct)

            for block in blocks:
                b_marks = sum(x[0] for x in block)
                b_len = len(block)
                if b_marks <= 0 or b_len == 0:
                    continue

                if not all((m, t) in valid_options for m, t in block):
                    continue

                if num_questions is not None and isinstance(num_questions, int) and num_questions > 0:
                    if num_questions % b_len == 0:
                        multiplier = num_questions // b_len
                        if multiplier * b_marks == total_marks:
                            tiled = block * multiplier
                            logger.info("Tier 1: Matched block pattern tiling (%d questions, %d marks): %s", num_questions, total_marks, tiled)
                            return tiled
                else:
                    if total_marks % b_marks == 0:
                        multiplier = total_marks // b_marks
                        tiled = block * multiplier
                        logger.info("Tier 1: Matched block pattern tiling (%d marks): %s", total_marks, tiled)
                        return tiled

        # TIER 2: Pattern-Scored Candidate Selection
        if past_structures:
            candidates = self._find_candidate_combinations(
                total_marks, valid_options, target_num_questions=num_questions, limit=50
            )
            if candidates:
                best_cand = self._score_and_select_candidate(candidates, past_structures)
                if best_cand:
                    logger.info("Tier 2: Selected pattern-scored structure: %s", best_cand)
                    return best_cand

        # TIER 3: Unconstrained Fallback (Pattern-Agnostic Safety)
        if num_questions is not None and isinstance(num_questions, int) and num_questions > 0:
            logger.info("Attempting Tier 3 fallback with requested %d questions for %d marks...", num_questions, total_marks)
            comb = self._find_valid_combination(
                total_marks, valid_options, target_num_questions=num_questions, position_options=position_options
            )
            if comb is not None and sum(x[0] for x in comb) == total_marks and len(comb) == num_questions:
                logger.info("Tier 3: Built structure with requested %d questions: %s", num_questions, comb)
                return comb
            elif position_options:
                comb = self._find_valid_combination(
                    total_marks, valid_options, target_num_questions=num_questions
                )
                if comb is not None and sum(x[0] for x in comb) == total_marks and len(comb) == num_questions:
                    logger.info("Tier 3: Built structure with requested %d questions (position constraint relaxed): %s", num_questions, comb)
                    return comb

            logger.warning(
                "Impossible to generate an exam with exactly %d questions for %d marks using available question options. "
                "Falling back to automatic question count selection.",
                num_questions, total_marks
            )

        past_exam_exact = past_structures
        if past_exam_exact:
            for exam_struct in past_exam_exact:
                current_sum = 0
                candidate_struct = []
                for q_m, q_t in exam_struct:
                    if (q_m, q_t) in valid_options and current_sum + q_m <= total_marks:
                        candidate_struct.append([q_m, q_t])
                        current_sum += q_m
                        if current_sum == total_marks:
                            logger.info("Tier 3: Found past exam pattern matching exact %d marks", total_marks)
                            return candidate_struct

        max_retries = getattr(self.config, "max_structure_retries", MAX_STRUCTURE_RETRIES)
        for retry in range(max_retries):
            comb = self._find_valid_combination(
                total_marks, valid_options, target_num_questions=None, position_options=position_options
            )
            if comb is not None and sum(x[0] for x in comb) == total_marks:
                logger.info("Tier 3: Structure built automatically on attempt %d (position-matched=%s): %s", retry + 1, bool(position_options), comb)
                return comb

        if position_options:
            logger.warning("Position-matched structure building failed. Falling back to non-position matched generation.")
            for retry in range(max_retries):
                comb = self._find_valid_combination(
                    total_marks, valid_options, target_num_questions=None
                )
                if comb is not None and sum(x[0] for x in comb) == total_marks:
                    logger.info("Tier 3: Structure built automatically on attempt %d (fallback): %s", retry + 1, comb)
                    return comb

        raise RuntimeError(
            f"Could not build valid exam structure using available question marks after {max_retries} attempts "
            f"for {total_marks} marks on topic '{topic}'"
        )

    @staticmethod
    def distribute_to_topics(
        base_structure: list, topics: list[str]
    ) -> dict[str, list]:
        """Distribute an exam structure evenly across topics.

        Args:
            base_structure: The overall exam structure (list of mark allocations).
            topics: List of topic names to distribute across.

        Returns:
            Dict mapping topic names to their allocated structure portions.
        """
        if not base_structure:
            return {t: [] for t in topics}
        n = len(base_structure)
        k = len(topics)
        if k == 0:
            raise ValueError("At least one topic is required")

        chunk_size = n // k
        remainder = n % k

        result = {}
        start = 0
        for i, topic in enumerate(topics):
            end = start + chunk_size + (1 if i < remainder else 0)
            result[topic] = base_structure[start:end]
            start = end

        return result


class LocalSimilarityEngine:
    """Fast, local semantic similarity calculator that caches embeddings.

    Uses the assistant's local HuggingFace embedding model for free and instant computations.
    """

    def __init__(self, embedding_model: Any = None, llm_client: Optional[LLMClient] = None, fallback_dim: int = 384):
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.cache: Dict[str, list[float]] = {}
        self.fallback_dim = fallback_dim

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts using local model or fallback."""
        if not texts:
            return []

        # Remove duplicate texts for efficiency
        unique_texts = list(set(texts))
        uncached_texts = [t for t in unique_texts if t not in self.cache]

        if uncached_texts:
            if self.embedding_model:
                try:
                    embeddings = self.embedding_model.embed_documents(uncached_texts)
                    for text, emb in zip(uncached_texts, embeddings):
                        self.cache[text] = emb
                except Exception as e:
                    logger.warning("Local embeddings failed, falling back to API: %s", e)
                    if self.llm_client:
                        embeddings = self.llm_client.get_embeddings(uncached_texts)
                        for text, emb in zip(uncached_texts, embeddings):
                            self.cache[text] = emb
            elif self.llm_client:
                embeddings = self.llm_client.get_embeddings(uncached_texts)
                for text, emb in zip(uncached_texts, embeddings):
                    self.cache[text] = emb
            else:
                # Random fallback if no embedding model is configured (e.g. in minimal tests)
                for text in uncached_texts:
                    self.cache[text] = [random.random() for _ in range(self.fallback_dim)]

        return [self.cache[t] for t in texts]

    def find_most_similar(self, query: str, candidates: list[str]) -> str:
        """Find the candidate that is semantically closest to the query."""
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        all_texts = [query] + candidates
        embeddings = self.get_embeddings(all_texts)

        q_emb = np.array(embeddings[0])
        c_embs = np.array(embeddings[1:])

        norms = np.linalg.norm(c_embs, axis=1)
        q_norm = np.linalg.norm(q_emb)

        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        q_norm = 1.0 if q_norm == 0 else q_norm

        similarities = c_embs @ q_emb / (norms * q_norm)
        best_idx = int(np.argmax(similarities))
        return candidates[best_idx]

    def pick_diverse_subset(self, candidates: list[str], k: int) -> list[str]:
        """Pick k candidates from a list that are least similar to each other, using a greedy MaxMin algorithm.

        The first item chosen is the first one in the candidates list (assumed to be the most relevant).
        """
        if not candidates or k <= 0:
            return []
        if len(candidates) <= k:
            return candidates.copy()

        selected = [candidates[0]]
        remaining = candidates[1:]

        # Get embeddings for all candidates
        embeddings = self.get_embeddings(candidates)
        emb_map = {text: np.array(emb) for text, emb in zip(candidates, embeddings)}

        while len(selected) < k and remaining:
            best_cand = None
            min_max_sim = float("inf")

            # Pre-compute normalized embeddings for selected items
            sel_embs = np.array([emb_map[s] for s in selected])
            sel_norms = np.linalg.norm(sel_embs, axis=1, keepdims=True)
            sel_norms = np.where(sel_norms == 0, 1.0, sel_norms)
            norm_sel_embs = sel_embs / sel_norms

            for cand in remaining:
                cand_emb = emb_map[cand]
                cand_norm = np.linalg.norm(cand_emb)
                cand_norm = 1.0 if cand_norm == 0 else cand_norm
                norm_cand_emb = cand_emb / cand_norm

                # Compute cosine similarities to all selected items
                sims = norm_sel_embs @ norm_cand_emb
                max_sim = float(np.max(sims))

                if max_sim < min_max_sim:
                    min_max_sim = max_sim
                    best_cand = cand

            if best_cand:
                selected.append(best_cand)
                remaining.remove(best_cand)
            else:
                break

        return selected

    def pick_least_similar(self, candidates: list[str], used: list[str]) -> str:
        """Pick a candidate from the least similar half vs used items to encourage diversity."""
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        if not used:
            return random.choice(candidates)

        embeddings_c = np.array(self.get_embeddings(candidates))
        embeddings_u = np.array(self.get_embeddings(used))

        norms_c = np.linalg.norm(embeddings_c, axis=1, keepdims=True)
        norms_u = np.linalg.norm(embeddings_u, axis=1, keepdims=True)
        norms_c = np.where(norms_c == 0, 1.0, norms_c)
        norms_u = np.where(norms_u == 0, 1.0, norms_u)

        sim_matrix = (embeddings_c / norms_c) @ (embeddings_u / norms_u).T
        max_sims = np.max(sim_matrix, axis=1)

        scored = list(zip(candidates, max_sims))
        scored.sort(key=lambda x: x[1])

        half_idx = len(scored) // 2
        lower_half = scored[half_idx:]
        if not lower_half:
            return scored[-1][0]

        return random.choice(lower_half)[0]


def _filter_by_exam_type(questions: list[dict], target_exam_type: str) -> list[dict]:
    """Filter candidate questions to match the target exam_type.
    
    Excludes questions that explicitly belong to a different exam_type.
    Falls back to untagged questions or all questions if no target match exists.
    """
    if not target_exam_type or not questions:
        return list(questions)
    
    target_clean = str(target_exam_type).strip().lower()
    
    # 1. Questions that explicitly match target exam_type
    exact_matches = [
        q for q in questions
        if q.get("exam_type") and str(q.get("exam_type")).strip().lower() == target_clean
    ]
    if exact_matches:
        return exact_matches
        
    # 2. If no exact matches, fallback to questions without an explicit exam_type tag
    unspecified = [
        q for q in questions
        if not q.get("exam_type")
    ]
    if unspecified:
        return unspecified

    # 3. Final safety fallback to all questions
    return list(questions)


class QuestionGenerator:
    """Generates GCSE exam questions using parallel, single-pass LLM prompts."""

    def __init__(
        self,
        config: SubjectConfig,
        llm_client: LLMClient,
        similarity_engine: SimilarityEngine,
        questions: list[dict],
        prompts: dict[str, str],
        queries: dict[str, str],
        spec_qa_chain: RetrievalQA,
        ms_retriever: Any = None,
        embedding_model: Any = None,
        specification_text: str = "",
    ):
        self.config = config
        self.llm = llm_client
        self.similarity = similarity_engine
        self.questions = questions
        self.prompts = prompts
        self.queries = queries
        self.spec_qa_chain = spec_qa_chain
        self.specification_text = specification_text

        # Initialize local fast similarity engine
        fallback_dim = getattr(config, "fallback_embedding_dim", 384)
        self.local_similarity = LocalSimilarityEngine(embedding_model, llm_client, fallback_dim=fallback_dim)

        # Cache for specification trees
        self.spec_trees_cache: Dict[str, dict] = {}

    def _get_spec_tree_cached(self, topic: str, exam_type: str) -> dict:
        """Fetch specification tree by looking up specification_summary.json or falling back to LLM parsing."""
        cache_key = f"{exam_type}-{topic}"
        if cache_key in self.spec_trees_cache:
            return self.spec_trees_cache[cache_key]

        # Fast lookup from specification_summary.json
        subject = getattr(self.config, "subject", "")
        examiner = getattr(self.config, "examiner", "")
        if subject and examiner:
            sum_path = f"data/{subject}/{examiner}/{subject}-{examiner}-specification_summary.json"
            if os.path.exists(sum_path):
                try:
                    with open(sum_path, "r", encoding="utf-8") as f:
                        summary_data = json.load(f)

                    et_target = exam_type.strip().lower() if exam_type else ""
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
                                logger.info("Found specification summary lookup for %s: %s (%d subtopics)", cache_key, spec_code, len(subtopics))
                                return tree
                except Exception as e:
                    logger.warning("Failed summary lookup for %s: %s", cache_key, e)

        logger.info("Extracting and compiling specification hierarchy via LLM for: %s", cache_key)

        if self.specification_text:
            spec_content = self.specification_text
        else:
            query = f"{exam_type} {topic}"
            docs = []
            if self.spec_qa_chain and hasattr(self.spec_qa_chain, "retriever"):
                try:
                    docs = self.spec_qa_chain.retriever.invoke(query)
                except Exception as e:
                    logger.error("Failed to retrieve docs from spec retriever: %s", e)

            spec_content = "\n\n".join(doc.page_content for doc in docs) if docs else "No specification details found."

        if exam_type and exam_type.strip().lower() not in topic.strip().lower():
            full_topic_ref = f"{exam_type} - {topic}"
        else:
            full_topic_ref = topic

        prompt = f"""You are an expert GCSE spec parser. Analyze the specification content below for the topic "{full_topic_ref}".
First, locate "{full_topic_ref}" within the specification and find its specific specification number/code (e.g. 'Topic 1 (1.1 - 1.17)', 'Section 2.1 - 2.15', 'Specification Code 3.1 - 3.20', '6.1 - 6.6').
Second, extract all key subtopics for "{full_topic_ref}" strictly and ONLY from the content belonging to that specific specification number/code for {exam_type if exam_type else 'this exam'}.

Specification content:
{spec_content}

Return the output strictly as a JSON object matching this schema:
{{
  "spec_code": "The specific specification number/code found for {full_topic_ref} (e.g. Topic 1 / 1.1 - 1.17)",
  "subtopics": [
    "Subtopic Name 1",
    "Subtopic Name 2"
  ]
}}
Ensure that:
1. "spec_code" contains the exact specification number or code range identified in the specification for "{full_topic_ref}".
2. All items in "subtopics" are extracted strictly and ONLY from the content under that specific specification number/code for {exam_type if exam_type else 'the requested exam'}.
3. Subtopics are possible subtopics to write questions of, so ensure each subtopic is appropriately scoped (avoiding items that are overly generic or hyper-specific).
Do not return any explanations or markdown wrapper, return only raw JSON."""

        try:
            raw_tree = self.llm.invoke_json(prompt)
        except Exception as e:
            logger.error("Failed parsing specification tree via LLM: %s. Using fallback.", e)
            raw_tree = {"spec_code": topic, "subtopics": [topic]}

        spec_code = str(raw_tree.get("spec_code", topic)).strip()
        raw_subs = raw_tree.get("subtopics", [])
        subtopics_list = []
        if isinstance(raw_subs, list):
            for sub in raw_subs:
                if isinstance(sub, str) and sub.strip():
                    subtopics_list.append(sub.strip())
                elif isinstance(sub, dict) and sub.get("name"):
                    subtopics_list.append(sub.get("name").strip())

        if not subtopics_list:
            subtopics_list = [topic]

        tree = {
            "spec_code": spec_code,
            "subtopics": subtopics_list
        }

        self.spec_trees_cache[cache_key] = tree
        return tree

    def _find_most_similar_question(self, query: str, candidate_qs: list[dict]) -> tuple[Optional[dict], float]:
        """Find candidate question matching desired subtopic using vector similarity.

        Compares strictly the subtopic titles (query vs candidate subtopic title).
        Selects a random candidate from any candidates with similarity >= threshold (default 70%).
        If no candidate meets the threshold, selects a random candidate from all available candidates.

        Returns:
            Tuple of (chosen_candidate_dict, similarity_score).
        """
        if not candidate_qs:
            return None, 0.0
        if len(candidate_qs) == 1:
            q = candidate_qs[0]
            sub_meta = (q.get("subtopic") or q.get("topic") or "").strip()
            text = sub_meta if sub_meta else "General"
            embeddings = self.local_similarity.get_embeddings([query, text])
            q_emb = np.array(embeddings[0])
            c_emb = np.array(embeddings[1])
            q_norm = np.linalg.norm(q_emb) or 1.0
            c_norm = np.linalg.norm(c_emb) or 1.0
            sim = float(c_emb @ q_emb / (c_norm * q_norm))
            return q, max(0.0, sim)

        threshold = float(getattr(self.config, "similarity_threshold", 0.70))

        texts = []
        for q in candidate_qs:
            sub_meta = (q.get("subtopic") or q.get("topic") or "").strip()
            text = sub_meta if sub_meta else "General"
            texts.append(text)

        all_texts = [query] + texts
        embeddings = self.local_similarity.get_embeddings(all_texts)

        q_emb = np.array(embeddings[0])
        c_embs = np.array(embeddings[1:])

        norms = np.linalg.norm(c_embs, axis=1)
        q_norm = np.linalg.norm(q_emb)
        norms = np.where(norms == 0, 1.0, norms)
        q_norm = 1.0 if q_norm == 0 else q_norm

        similarities = c_embs @ q_emb / (norms * q_norm)
        above_threshold_indices = [idx for idx, sim in enumerate(similarities) if sim >= threshold]

        if above_threshold_indices:
            chosen_idx = int(random.choice(above_threshold_indices))
        else:
            chosen_idx = int(np.argmax(similarities))

        chosen_sim = float(similarities[chosen_idx])
        return candidate_qs[chosen_idx], max(0.0, chosen_sim)

    def _clean_and_format_sub_questions(self, sub_questions: list[dict]) -> list[dict]:
        """Clean redundant text from sub-questions and format nicely."""
        cleaned = []
        for sq in sub_questions:
            item = {}
            if "label" in sq:
                item["label"] = sq["label"]
            if "context" in sq and sq["context"]:
                item["context"] = clean_question_text(sq["context"])
            if "text" in sq and sq["text"]:
                item["text"] = clean_question_text(sq["text"])
            if "marks" in sq:
                item["marks"] = sq["marks"]
            if "subtopic" in sq:
                item["subtopic"] = sq["subtopic"]
            if "sub_parts" in sq:
                gqs = []
                for gq in sq["sub_parts"]:
                    gqs.append({
                        "label": gq.get("label", ""),
                        "text": clean_question_text(gq.get("text", "")),
                        "marks": gq.get("marks", 1),
                    })
                item["sub_parts"] = gqs
            cleaned.append(item)
        return cleaned

    def _execute_generation_task(self, task: dict) -> dict:
        """Executes a single question generation task (called in parallel thread)."""
        mark_structure = task["mark_structure"]
        exam_type = task["exam_type"]
        subtopic = task.get("subtopic", "")
        subject = task["subject"]
        q_num = task["number"]

        # Parse mark structure & question type
        if isinstance(mark_structure, list) and len(mark_structure) == 2 and isinstance(mark_structure[1], str):
            target_marks, q_type = mark_structure[0], mark_structure[1]
        elif isinstance(mark_structure, int):
            target_marks, q_type = mark_structure, "basic"
        else:
            target_marks = ExamStructureBuilder.flatten_marks(mark_structure)
            q_type = "parent" if isinstance(mark_structure, list) else "basic"

        # Pre-filter candidate questions pool to match target exam_type
        pool = _filter_by_exam_type(self.questions, exam_type)

        if q_type == "parent":
            logger.info("Retrieving parent question Q%d (%d marks) [%s] matching subtopic '%s'...", q_num, target_marks, exam_type, subtopic)

            # Priority 1: parent_question with exact total marks within filtered pool
            candidates = [
                q for q in pool
                if (q.get("type") == "parent_question" or q.get("parent_question_structure") or q.get("sub_questions"))
                and ExamStructureBuilder.flatten_marks(q.get("parent_question_structure")) == target_marks
            ]
            # Priority 2: any parent question within filtered pool
            if not candidates:
                candidates = [
                    q for q in pool
                    if q.get("type") == "parent_question" or q.get("parent_question_structure") or q.get("sub_questions")
                ]
            # Priority 3: fallback all questions within filtered pool
            if not candidates:
                candidates = list(pool)
            # Final safety fallback to all questions if pool was empty
            if not candidates:
                candidates = list(self.questions)

            q_best, sim_score = self._find_most_similar_question(subtopic, candidates)
            parent_desc = clean_question_text(q_best.get("parent_description", "") or q_best.get("question_content", ""))
            matched_subtopic = q_best.get("subtopic", subtopic) if q_best else subtopic
            
            orig_sub_qs = q_best.get("sub_questions", []) if q_best else []
            cleaned_sub_qs = self._clean_and_format_sub_questions(orig_sub_qs)

            return {
                "number": f"{q_num})",
                "parent_description": parent_desc,
                "sub_questions": cleaned_sub_qs,
                "subtopic": subtopic,
                "matched_subtopic": matched_subtopic,
                "similarity_score": round(sim_score, 4),
                "subtopic_data": task.get("subtopic_data", {}),
                "marks": target_marks,
                "q_num": q_num
            }

        else:
            logger.info("Retrieving standalone question Q%d (%d marks) [%s] matching subtopic '%s'...", q_num, target_marks, exam_type, subtopic)

            # Priority 1: standalone basic_question (NOT part of a parent question) with exact marks within filtered pool
            candidates = [
                q for q in pool
                if (q.get("type") == "basic_question" and not q.get("parent_question_structure") and not q.get("sub_questions"))
                and q.get("marks") == target_marks
            ]
            # Priority 2: any standalone basic question within filtered pool
            if not candidates:
                candidates = [
                    q for q in pool
                    if q.get("type") == "basic_question" and not q.get("parent_question_structure") and not q.get("sub_questions")
                ]
            # Priority 3: fallback all questions within filtered pool
            if not candidates:
                candidates = list(pool)
            # Final safety fallback to all questions if pool was empty
            if not candidates:
                candidates = list(self.questions)

            q_best, sim_score = self._find_most_similar_question(subtopic, candidates)
            question_text = clean_question_text(q_best.get("question_content") or q_best.get("text") or "") if q_best else ""
            matched_subtopic = q_best.get("subtopic", subtopic) if q_best else subtopic

            return {
                "number": f"{q_num})",
                "text": question_text,
                "marks": target_marks,
                "subtopic": subtopic,
                "matched_subtopic": matched_subtopic,
                "similarity_score": round(sim_score, 4),
                "subtopic_data": task.get("subtopic_data", {}),
                "q_num": q_num
            }

    def generate_exam(
        self,
        exam_type: str,
        total_marks: int,
        topics: list[str],
        subject: str,
        structure_builder: ExamStructureBuilder,
        user_preferences: Optional[Any] = None,
        num_questions: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate a complete exam using parallel generation.

        Args:
            exam_type: The exam type/tier (e.g., "Higher", "Christianity").
            total_marks: Total marks for the exam.
            topics: List of topic areas to cover.
            subject: The subject name.
            structure_builder: An ExamStructureBuilder instance.
            user_preferences: Optional custom preferences.
            num_questions: Optional requested number of questions in the exam.

        Returns:
            Dict containing the exam structure, specification trees, and generated questions.
        """
        raw_structure = structure_builder.build_structure(total_marks, exam_type, num_questions=num_questions)
        exam_structure = structure_builder.distribute_to_topics(raw_structure, topics)
    def _optimize_topic_subtopic_assignments(
        self,
        exam_structure: dict[str, list],
        exam_type: str,
        spec_trees: dict[str, Any]
    ) -> dict[str, list[str]]:
        """Optimize subtopic assignments for each topic's question slots.

        Ensures question slots are assigned subtopics that have candidate questions
        matching >= 50% subtopic similarity whenever possible.
        If no candidate meets 50% similarity for a slot, assigns the subtopic
        with the highest achievable similarity.
        """
        def _get_subtopics(tree_obj: Any) -> list[str]:
            if isinstance(tree_obj, dict):
                if "subtopics" in tree_obj and isinstance(tree_obj["subtopics"], list):
                    return [s for s in tree_obj["subtopics"] if isinstance(s, str)]
                return [k for k in tree_obj.keys() if not k.startswith("_")]
            elif isinstance(tree_obj, list):
                return [s for s in tree_obj if isinstance(s, str)]
            return []

        min_threshold = float(getattr(self.config, "min_subtopic_similarity", 0.50))
        soft_threshold = float(getattr(self.config, "min_soft_subtopic_similarity", 0.35))
        diversity_weight = float(getattr(self.config, "subtopic_diversity_weight", 500.0))
        pool = _filter_by_exam_type(self.questions, exam_type)
        assigned_subtopics_by_topic: dict[str, list[str]] = {}

        for topic, mark_infos in exam_structure.items():
            spec_tree = spec_trees.get(topic, {})
            subtopics_pool = _get_subtopics(spec_tree)
            if not subtopics_pool:
                subtopics_pool = [topic]

            n_slots = len(mark_infos)
            n_subtopics = len(subtopics_pool)

            max_sim = np.zeros((n_slots, n_subtopics), dtype=float)

            subtopic_embs = np.array(self.local_similarity.get_embeddings(subtopics_pool))
            subtopic_norms = np.linalg.norm(subtopic_embs, axis=1)
            subtopic_norms = np.where(subtopic_norms == 0, 1.0, subtopic_norms)

            norm_sub_embs = subtopic_embs / subtopic_norms[:, None]
            subtopic_inter_sim = norm_sub_embs @ norm_sub_embs.T

            for i, mark_info in enumerate(mark_infos):
                if isinstance(mark_info, list) and len(mark_info) == 2 and isinstance(mark_info[1], str):
                    target_marks, q_type = mark_info[0], mark_info[1]
                elif isinstance(mark_info, int):
                    target_marks, q_type = mark_info, "basic"
                else:
                    target_marks = ExamStructureBuilder.flatten_marks(mark_info)
                    q_type = "parent" if isinstance(mark_info, list) else "basic"

                if q_type == "parent":
                    candidates = [
                        q for q in pool
                        if (q.get("type") == "parent_question" or q.get("parent_question_structure") or q.get("sub_questions"))
                        and ExamStructureBuilder.flatten_marks(q.get("parent_question_structure")) == target_marks
                    ]
                    if not candidates:
                        candidates = [
                            q for q in pool
                            if q.get("type") == "parent_question" or q.get("parent_question_structure") or q.get("sub_questions")
                        ]
                else:
                    candidates = [
                        q for q in pool
                        if (q.get("type") == "basic_question" and not q.get("parent_question_structure") and not q.get("sub_questions"))
                        and q.get("marks") == target_marks
                    ]
                    if not candidates:
                        candidates = [
                            q for q in pool
                            if q.get("type") == "basic_question" and not q.get("parent_question_structure") and not q.get("sub_questions")
                        ]
                if not candidates:
                    candidates = list(pool) or list(self.questions)

                cand_texts = [
                    (q.get("subtopic") or q.get("topic") or "").strip() or "General"
                    for q in candidates
                ]
                cand_embs = np.array(self.local_similarity.get_embeddings(cand_texts))
                cand_norms = np.linalg.norm(cand_embs, axis=1)
                cand_norms = np.where(cand_norms == 0, 1.0, cand_norms)

                sim_matrix = (cand_embs @ subtopic_embs.T) / (cand_norms[:, None] * subtopic_norms[None, :])
                sim_matrix = np.maximum(0.0, sim_matrix)

                max_sim[i] = np.max(sim_matrix, axis=0)

            def _eval_score(assign: list[int]) -> float:
                above_cnt = sum(1 for idx, j_idx in enumerate(assign) if max_sim[idx][j_idx] >= min_threshold)
                soft_above_cnt = sum(1 for idx, j_idx in enumerate(assign) if max_sim[idx][j_idx] >= soft_threshold)
                unique_cnt = len(set(assign))
                repetition_penalty = (n_slots - unique_cnt) * 300.0

                distinct_indices = list(set(assign))
                if len(distinct_indices) > 1:
                    inter_sims = [
                        subtopic_inter_sim[distinct_indices[a]][distinct_indices[b]]
                        for a in range(len(distinct_indices))
                        for b in range(a + 1, len(distinct_indices))
                    ]
                    mean_inter_sim = float(np.mean(inter_sims))
                    semantic_diversity_reward = (1.0 - mean_inter_sim) * 100.0
                else:
                    semantic_diversity_reward = 0.0

                tot_sim = sum(max_sim[idx][j_idx] for idx, j_idx in enumerate(assign))
                return (above_cnt * 1000.0) + (soft_above_cnt * 200.0) + (unique_cnt * diversity_weight) - repetition_penalty + semantic_diversity_reward + tot_sim

            used_indices = set()
            assignment = []
            for i in range(n_slots):
                valid_unused = [j for j in range(n_subtopics) if j not in used_indices and max_sim[i][j] >= min_threshold]
                if valid_unused:
                    best_j = max(valid_unused, key=lambda j: max_sim[i][j])
                else:
                    soft_unused = [j for j in range(n_subtopics) if j not in used_indices and max_sim[i][j] >= soft_threshold]
                    if soft_unused:
                        best_j = max(soft_unused, key=lambda j: max_sim[i][j])
                    else:
                        unused_any = [j for j in range(n_subtopics) if j not in used_indices]
                        if unused_any:
                            best_j = max(unused_any, key=lambda j: max_sim[i][j])
                        else:
                            best_j = int(np.argmax(max_sim[i]))
                assignment.append(best_j)
                used_indices.add(best_j)

            best_assignment = list(assignment)
            best_score = _eval_score(best_assignment)

            for i in range(n_slots):
                for j in range(n_subtopics):
                    curr = list(best_assignment)
                    curr[i] = j
                    sc = _eval_score(curr)
                    if sc > best_score:
                        best_score = sc
                        best_assignment = curr

            for i1 in range(n_slots):
                for i2 in range(i1 + 1, n_slots):
                    curr = list(best_assignment)
                    curr[i1], curr[i2] = curr[i2], curr[i1]
                    sc = _eval_score(curr)
                    if sc > best_score:
                        best_score = sc
                        best_assignment = curr

            assigned_subtopics_by_topic[topic] = [subtopics_pool[j] for j in best_assignment]

        return assigned_subtopics_by_topic

    def generate_exam(
        self,
        exam_type: str,
        total_marks: int,
        topics: list[str],
        subject: str,
        structure_builder: ExamStructureBuilder,
        user_preferences: Optional[Any] = None,
        num_questions: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate a complete exam using parallel generation.

        Args:
            exam_type: The exam type/tier (e.g., "Higher", "Christianity").
            total_marks: Total marks for the exam.
            topics: List of topic areas to cover.
            subject: The subject name.
            structure_builder: An ExamStructureBuilder instance.
            user_preferences: Optional custom preferences.
            num_questions: Optional requested number of questions in the exam.

        Returns:
            Dict containing the exam structure, specification trees, and generated questions.
        """
        raw_structure = structure_builder.build_structure(total_marks, exam_type, num_questions=num_questions)
        exam_structure = structure_builder.distribute_to_topics(raw_structure, topics)
        logger.info("Exam structure distributed: %s", exam_structure)

        # Prepare tasks for all questions across all topics
        tasks = []
        question_number = 0
        spec_trees = {}

        for topic in exam_structure:
            spec_tree = self._get_spec_tree_cached(topic, exam_type)
            spec_trees[topic] = spec_tree

        # Optimize subtopic assignments across question slots to maximize >= 50% subtopic similarity
        optimized_subtopics = self._optimize_topic_subtopic_assignments(
            exam_structure, exam_type, spec_trees
        )

        for topic in exam_structure:
            spec_tree = spec_trees[topic]
            assigned_list = optimized_subtopics.get(topic, [])

            for slot_idx, mark_info in enumerate(exam_structure[topic]):
                question_number += 1
                subtopic = assigned_list[slot_idx] if slot_idx < len(assigned_list) else topic

                tasks.append({
                    "number": question_number,
                    "topic": topic,
                    "subtopic": subtopic,
                    "subtopic_data": spec_tree.get(subtopic, {}),
                    "mark_structure": mark_info,
                    "exam_type": exam_type,
                    "subject": subject,
                    "user_preferences": user_preferences,
                })

        # Execute all tasks in parallel using a thread pool
        exam_output: dict[str, Any] = {
            "structure": exam_structure,
            "spec_trees": spec_trees,
            "questions": {}
        }
        results = []

        logger.info("Submitting %d question generation tasks to ThreadPoolExecutor...", len(tasks))
        max_workers = getattr(self.config, "max_parallel_workers", 10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(tasks) or 1)) as executor:
            future_to_task = {executor.submit(self._execute_generation_task, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logger.error("Question task generation failed for Q%d: %s", task["number"], e, exc_info=True)
                    raise

        # Sort the results by question number to restore logical order
        results.sort(key=lambda x: x["q_num"])

        # Group sorted results by topic
        for task in tasks:
            topic = task["topic"]
            exam_output["questions"].setdefault(topic, [])

        for res in results:
            q_num = res["q_num"]
            orig_task = next(t for t in tasks if t["number"] == q_num)
            topic = orig_task["topic"]

            res.pop("q_num", None)
            exam_output["questions"][topic].append(res)

        return exam_output

    def get_topics_from_user_input(self, user_input: str) -> str:
        """Extract topic names from free-text user input."""
        prompt = self.prompts["get_topics_from_user"].format(
            user_input=user_input
        )
        return self.llm.invoke(prompt)

    def get_revision_materials(
        self, topic: str, subject: str, spec_qa_chain: RetrievalQA
    ) -> str:
        """Generate revision materials for a topic."""
        if self.specification_text:
            prompt = f"Using the following GCSE {subject} specification context, generate detailed, well-structured revision notes/materials for the topic '{topic}':\n\nSPECIFICATION CONTEXT:\n{self.specification_text}\n\nRevision Notes:"
            return self.llm.invoke(prompt)
        else:
            query = self.queries["revision_materials"].format(
                topic=topic, subject=subject
            )
            return self.llm.invoke_qa(spec_qa_chain, query)
