"""Optimized Exam generation module for GCSE AI.

Contains ExamStructureBuilder for creating exam structures from past paper patterns,
and QuestionGenerator for generating individual questions and full exams.
"""

import logging
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

    def get_past_exam_structures(self, topic: str) -> list[list[list]]:
        """Extract mark structures from past exams for a given topic.

        Args:
            topic: The exam topic to filter questions by.

        Returns:
            A list of exams, where each exam is a list of [marks, type] pairs.
        """
        current_exam = ""
        exam_marks: list[list] = []
        all_exams: list[list[list]] = []

        for question in self.questions:
            is_relevant = (
                question["topic"] == topic
                and question["type"] in ("parent_question", "basic_question")
            )
            if not is_relevant:
                continue

            if question["exam"] != current_exam:
                if exam_marks:
                    all_exams.append(exam_marks)
                exam_marks = []
                current_exam = question["exam"]

            if question["type"] == "parent_question":
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

    def _find_valid_combination(
        self,
        total_marks: int,
        valid_options: list[tuple[int, str]],
        target_num_questions: Optional[int] = None,
    ) -> Optional[list[list]]:
        """Find a valid sequence of [marks, type] from valid_options that sum exactly to total_marks.

        If target_num_questions is specified, only combinations with that exact question count are returned.
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

            fitting = [opt for opt in options if opt[0] <= remaining]
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
        """Build an exam structure that totals the given marks using ONLY recorded valid question marks.

        Args:
            total_marks: Target total marks for the exam.
            topic: The exam topic.
            num_questions: Optional requested number of questions in the exam.

        Returns:
            A list of mark structures [marks, question_type] for each question.
        """
        # 1. Parse all questions at highest level to record what marked questions are available
        valid_options = self.get_valid_question_options(topic)
        if not valid_options:
            valid_options = self.get_valid_question_options(None)

        if not valid_options:
            raise ValueError(f"No valid questions found in database for topic '{topic}'.")

        logger.info("Recorded %d valid question mark options: %s", len(valid_options), valid_options)

        # 2. If explicit num_questions requested, try to generate a combination with exactly num_questions
        if num_questions is not None and isinstance(num_questions, int) and num_questions > 0:
            logger.info("Attempting to build structure with requested %d questions for %d marks...", num_questions, total_marks)
            comb = self._find_valid_combination(total_marks, valid_options, target_num_questions=num_questions)
            if comb is not None and sum(x[0] for x in comb) == total_marks and len(comb) == num_questions:
                logger.info("Successfully built structure with requested %d questions: %s", num_questions, comb)
                return comb
            else:
                logger.warning(
                    "Impossible to generate an exam with exactly %d questions for %d marks using available question options. "
                    "Falling back to automatic question count selection.",
                    num_questions, total_marks
                )

        # 3. Check if a past exam structure sums to exact total_marks using ONLY valid options
        past_structures = self.get_past_exam_structures(topic)
        if past_structures:
            for exam_struct in past_structures:
                current_sum = 0
                candidate_struct = []
                for q_m, q_t in exam_struct:
                    if (q_m, q_t) in valid_options and current_sum + q_m <= total_marks:
                        candidate_struct.append([q_m, q_t])
                        current_sum += q_m
                        if current_sum == total_marks:
                            logger.info("Found past exam pattern matching exact %d marks", total_marks)
                            return candidate_struct

        # 4. Generate a combination automatically using ONLY valid question mark options that sum to total_marks
        max_retries = getattr(self.config, "max_structure_retries", MAX_STRUCTURE_RETRIES)
        for retry in range(max_retries):
            comb = self._find_valid_combination(total_marks, valid_options, target_num_questions=None)
            if comb is not None and sum(x[0] for x in comb) == total_marks:
                logger.info("Exam structure built automatically on attempt %d: %s", retry + 1, comb)
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
        """Fetch specification chunks and compile them into a structured tree using a single LLM call."""
        cache_key = f"{exam_type}-{topic}"
        if cache_key in self.spec_trees_cache:
            return self.spec_trees_cache[cache_key]

        logger.info("Extracting and compiling specification hierarchy for: %s", cache_key)

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

    def _find_most_similar_question(self, query: str, candidate_qs: list[dict]) -> dict:
        if not candidate_qs:
            return None
        if len(candidate_qs) == 1:
            return candidate_qs[0]

        texts = []
        for q in candidate_qs:
            text = q.get("question_content") or q.get("parent_question_description") or q.get("topic") or ""
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
        best_idx = int(np.argmax(similarities))
        return candidate_qs[best_idx]

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

            # Prefer exact subtopic and topic metadata matches if present
            sub_matches = [q for q in candidates if q.get("subtopic") and q.get("subtopic").lower() == subtopic.lower()]
            if sub_matches:
                candidates = sub_matches
            else:
                top_matches = [q for q in candidates if q.get("topic") and task.get("topic") and q.get("topic").lower() == task.get("topic").lower()]
                if top_matches:
                    candidates = top_matches

            q_best = self._find_most_similar_question(subtopic, candidates)
            parent_desc = clean_question_text(q_best.get("parent_description", "") or q_best.get("question_content", ""))
            
            orig_sub_qs = q_best.get("sub_questions", [])
            cleaned_sub_qs = self._clean_and_format_sub_questions(orig_sub_qs)

            return {
                "number": f"{q_num})",
                "parent_description": parent_desc,
                "sub_questions": cleaned_sub_qs,
                "subtopic": subtopic,
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

            # Prefer exact subtopic and topic metadata matches if present
            sub_matches = [q for q in candidates if q.get("subtopic") and q.get("subtopic").lower() == subtopic.lower()]
            if sub_matches:
                candidates = sub_matches
            else:
                top_matches = [q for q in candidates if q.get("topic") and task.get("topic") and q.get("topic").lower() == task.get("topic").lower()]
                if top_matches:
                    candidates = top_matches

            q_best = self._find_most_similar_question(subtopic, candidates)
            question_text = clean_question_text(q_best.get("question_content") or q_best.get("text") or "")

            return {
                "number": f"{q_num})",
                "text": question_text,
                "marks": target_marks,
                "subtopic": subtopic,
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
        logger.info("Exam structure distributed: %s", exam_structure)

        # Prepare tasks for all questions across all topics
        tasks = []
        question_number = 0
        spec_trees = {}

        def _get_subtopics(tree_obj: Any) -> list[str]:
            if isinstance(tree_obj, dict):
                if "subtopics" in tree_obj and isinstance(tree_obj["subtopics"], list):
                    return [s for s in tree_obj["subtopics"] if isinstance(s, str)]
                return [k for k in tree_obj.keys() if not k.startswith("_")]
            elif isinstance(tree_obj, list):
                return [s for s in tree_obj if isinstance(s, str)]
            return []

        for topic in exam_structure:
            spec_tree = self._get_spec_tree_cached(topic, exam_type)
            spec_trees[topic] = spec_tree
            subtopics_pool = _get_subtopics(spec_tree)
            used_subtopics: list[str] = []

            for mark_info in exam_structure[topic]:
                question_number += 1
                
                # Single broad subtopic for the question
                if not subtopics_pool:
                    subtopics_pool = _get_subtopics(spec_tree)
                subtopic = self.local_similarity.pick_least_similar(subtopics_pool, used_subtopics)
                if subtopic in subtopics_pool:
                    subtopics_pool.remove(subtopic)
                used_subtopics.append(subtopic)

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
