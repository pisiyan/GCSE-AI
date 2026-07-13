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
from langchain.chains import RetrievalQA

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
            Total marks as an integer.
        """
        if isinstance(marks, int):
            return marks
        if not isinstance(marks, list):
            return marks
        total = 0
        for item in marks:
            if isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, int):
                        total += sub_item
            elif isinstance(item, int):
                total += item
        return total

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
                marks = [self.flatten_marks(question["parent_question_structure"]), "parent"]
            else:
                marks = [question["marks"], "basic"]
            exam_marks.append(marks)

        if exam_marks:
            all_exams.append(exam_marks)

        return all_exams

    def build_structure(self, total_marks: int, topic: str) -> list:
        """Build an exam structure that totals the given marks.

        Uses patterns from past papers to create a realistic exam structure.
        Retries up to MAX_STRUCTURE_RETRIES times if the structure cannot be resolved.

        Args:
            total_marks: Target total marks for the exam.
            topic: The exam topic for retrieving past structures.

        Returns:
            A list of mark structures (ints or nested lists for parent questions).

        Raises:
            RuntimeError: If a valid structure cannot be built within the retry limit.
        """
        past_structures = self.get_past_exam_structures(topic)
        if not past_structures:
            raise ValueError(f"No past exam structures found for topic: {topic}")

        avg_marks = sum(m[0] for m in past_structures[0]) / len(past_structures[0])
        n_questions = round(total_marks / avg_marks)

        max_retries = getattr(self.config, "max_structure_retries", MAX_STRUCTURE_RETRIES)
        for retry in range(max_retries):
            try:
                structure = self._attempt_build(
                    total_marks, n_questions, past_structures, topic
                )
                if structure is not None:
                    logger.info("Exam structure built successfully on attempt %d", retry + 1)
                    return structure
            except (IndexError, ValueError) as e:
                logger.debug("Structure attempt %d failed: %s", retry + 1, e)

        raise RuntimeError(
            f"Could not build valid exam structure after {max_retries} attempts "
            f"for {total_marks} marks on topic '{topic}'"
        )

    def _attempt_build(
        self,
        total_marks: int,
        n_questions: int,
        past_structures: list[list[list]],
        topic: str,
    ) -> Optional[list]:
        """Single attempt to build an exam structure.

        Returns:
            The completed structure, or None if this attempt failed.
        """
        exam_structure = []
        last_type = ""
        marks_left = total_marks

        for i in range(n_questions):
            possible_marks = []
            for exam in past_structures:
                if i < len(exam):
                    possible_marks.append(exam[i])

            valid_marks = [m for m in possible_marks if m[0] <= marks_left]
            if valid_marks:
                chosen = random.choice(valid_marks)
            else:
                chosen = [0, last_type]

            exam_structure.append(chosen)
            marks_left -= chosen[0]
            last_type = chosen[1]

        # Distribute remaining marks to the last question
        if marks_left != 0 and exam_structure:
            exam_structure[-1][0] += marks_left

        # Resolve parent question structures
        return self._resolve_structures(exam_structure, past_structures, topic)

    def _resolve_structures(
        self,
        exam_structure: list[list],
        past_structures: list[list[list]],
        topic: str,
    ) -> Optional[list]:
        """Resolve abstract mark allocations into concrete question structures.

        For parent questions, finds matching structures from past papers.
        """
        final_structure = []
        question_number = 0
        last_exam = ""

        for question in exam_structure:
            marks = question[0]
            question_type = question[1]

            if question_type == "parent":
                possible_structures = []
                q_counter = 0

                for q in self.questions:
                    if q["type"] not in ("parent_question", "basic_question"):
                        continue
                    q_counter += 1
                    if q["exam"] != last_exam:
                        last_exam = q["exam"]
                        q_counter = 0

                    matches_marks = (
                        q["type"] == "parent_question" 
                        and q.get("parent_question_structure") is not None
                        and self.flatten_marks(q["parent_question_structure"]) == marks
                    )
                    matches_topic = q["topic"] == topic
                    matches_position = (
                        q_counter == exam_structure.index(question)
                        or not self.config.question_no_importance
                    )

                    if matches_marks and matches_topic and matches_position:
                        possible_structures.append(q["parent_question_structure"])

                if not possible_structures:
                    return None  # Signal retry needed

                marks = random.choice(possible_structures)
                question_number += 1

            final_structure.append(marks)

        return final_structure

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


    def find_least_similar_objects(
        self,
        objects: list[dict],
        comparison: str,
        n: int,
        topic_value: str = "",
        marks_value: Optional[int] = None,
    ) -> list[str]:
        """Filter past questions, score by similarity, return n LEAST similar to prevent duplicates."""
        target_calc = is_calculation_content(comparison)
        target_pract = is_practical_content(comparison)

        filtered = []
        shuffled = objects.copy()
        random.shuffle(shuffled)

        for obj in shuffled:
            matches_topic = (not topic_value) or obj.get("topic") == topic_value
            matches_marks = (marks_value is None) or obj.get("marks") == marks_value
            if not (matches_topic and matches_marks):
                continue

            obj_content = obj.get("question_content", "") or ""
            obj_calc = is_calculation_content(obj_content)
            obj_pract = is_practical_content(obj_content)

            # Filter/Prioritize based on cognitive type matching
            if target_calc and not obj_calc:
                continue
            if target_pract and not obj_pract:
                continue
            if not target_calc and not target_pract and (obj_calc or obj_pract):
                continue

            filtered.append(obj)
            if len(filtered) >= n * 2:  # Subset limit for speed
                break

        # Fallback: if we didn't find enough matches, relax the cognitive constraint
        if len(filtered) < n:
            filtered = []
            for obj in shuffled:
                matches_topic = (not topic_value) or obj.get("topic") == topic_value
                matches_marks = (marks_value is None) or obj.get("marks") == marks_value
                if matches_topic and matches_marks:
                    filtered.append(obj)
                if len(filtered) >= n * 2:
                    break

        if not filtered:
            return []

        contents = [obj["question_content"] for obj in filtered]
        all_texts = [comparison] + contents
        embeddings = self.get_embeddings(all_texts)

        q_emb = np.array(embeddings[0])
        c_embs = np.array(embeddings[1:])

        norms = np.linalg.norm(c_embs, axis=1)
        q_norm = np.linalg.norm(q_emb)
        norms = np.where(norms == 0, 1.0, norms)
        q_norm = 1.0 if q_norm == 0 else q_norm

        similarities = c_embs @ q_emb / (norms * q_norm)
        scored = list(zip(contents, similarities))
        scored.sort(key=lambda x: x[1])  # Ascending order (least similar first)

        return [item[0] for item in scored[:n]]

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

    def _get_mark_calibration_examples(self) -> str:
        """Find real exemplar questions from the database for each mark weight to serve as calibration guidelines."""
        import random
        marks_needed = [1, 2, 3, 4, 6, 12]
        examples = {}
        
        # Shuffle questions to get a random mix each time
        shuffled_q = list(self.questions)
        random.shuffle(shuffled_q)
        
        for q in shuffled_q:
            q_marks = q.get("marks")
            content = q.get("question_content") or q.get("text")
            if q_marks in marks_needed and content and len(content) < 300:
                examples[q_marks] = content.strip()
                if len(examples) == len(marks_needed):
                    break
        
        lines = ["GCSE Mark Calibration Guidelines (depth expected per mark weight):"]
        for m in sorted(marks_needed):
            ex = examples.get(m)
            if ex:
                ex_clean = " ".join(ex.split())
                lines.append(f"- {m} Mark Example (DO this level of depth): \"{ex_clean}\"")
            else:
                if m == 1:
                    lines.append("- 1 Mark Guideline: Single direct recall, simple identification, or multiple-choice question.")
                elif m == 2:
                    lines.append("- 2 Marks Guideline: State a fact/method and give one brief reason or detail.")
                elif m == 3:
                    lines.append("- 3 Marks Guideline: Multi-step simple calculation, or a concept explanation with two logical steps.")
                elif m == 4:
                    lines.append("- 4 Marks Guideline: Detailed explanation showing cause and effect, or calculation with conversions/formula showing.")
                elif m == 6:
                    lines.append("- 6 Marks Guideline: Structured, detailed scientific explanation or multi-faceted evaluation (requires dotted lines).")
                elif m == 12:
                    lines.append("- 12 Marks Guideline: AQA Religious Studies essay question. Must present arguments for and against a statement, plus a justified conclusion.")
                    
        return "\n".join(lines)

    def _get_spec_tree_cached(self, topic: str, exam_topic: str) -> dict:
        """Fetch specification chunks and compile them into a structured tree using a single LLM call."""
        cache_key = f"{exam_topic}-{topic}"
        if cache_key in self.spec_trees_cache:
            return self.spec_trees_cache[cache_key]

        logger.info("Extracting and compiling specification hierarchy for: %s", cache_key)
        
        # 1. Retrieve raw spec documents directly from FAISS without LLM (or use full spec text)
        if self.specification_text:
            spec_content = self.specification_text
        else:
            query = f"{exam_topic} {topic}"
            docs = []
            if self.spec_qa_chain and hasattr(self.spec_qa_chain, "retriever"):
                try:
                    docs = self.spec_qa_chain.retriever.invoke(query)
                except Exception as e:
                    logger.error("Failed to retrieve docs from spec retriever: %s", e)
            
            spec_content = "\n\n".join(doc.page_content for doc in docs) if docs else "No specification details found."

        # 2. Call LLM once to compile it into a structured subtopic/sub-subtopic tree
        prompt = f"""You are an expert GCSE spec parser. Analyze the specification content for the topic "{topic}" below.
Extract all key subtopics. For each subtopic, provide:
1. "name": The subtopic name (e.g. 'Photosynthesis', 'Core Practical').
2. "description": A summary of the core concepts or facts.
3. "sub_subtopics": A list of specific concepts, requirements, or points, each with a 'name' and 'description'.

Specification content:
{spec_content}

Return the output strictly as a JSON object matching this schema:
{{
  "subtopics": [
    {{
      "name": "Subtopic Name",
      "description": "Short description of what is studied",
      "sub_subtopics": [
        {{
          "name": "Sub-subtopic Name/Concept",
          "description": "Specific detail or adaptation requirement"
        }}
      ]
    }}
  ]
}}
Do not return any explanations, return only the JSON block (no ```json markdown wrapper, just raw JSON)."""

        try:
            raw_tree = self.llm.invoke_json(prompt)
        except Exception as e:
            logger.error("Failed parsing specification tree via LLM: %s. Using fallback.", e)
            raw_tree = {"subtopics": [{"name": topic, "description": topic, "sub_subtopics": []}]}

        # Re-structure for clean lookups
        tree = {}
        for sub in raw_tree.get("subtopics", []):
            name = sub.get("name", "")
            if name:
                tree[name] = {
                    "description": sub.get("description", ""),
                    "sub_subtopics": sub.get("sub_subtopics", [])
                }

        if not tree:
            # Fallback
            tree = {topic: {"description": topic, "sub_subtopics": []}}

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

    def _build_copied_sub_questions(self, structure: list, leaf_qs: list[dict], subtopics: list[str]) -> list[dict]:
        letters = "abcdefghijklmn"
        romans = ["i", "ii", "iii", "iv", "v", "vi"]
        
        parsed = []
        leaf_iterator = iter(leaf_qs)
        
        for idx, item in enumerate(structure):
            label = f"{letters[idx]})"
            subtopic_name = subtopics[idx] if idx < len(subtopics) else (subtopics[-1] if subtopics else "")
            
            if isinstance(item, list):
                # Sub-parent: contains grandchild questions
                roman_list = []
                for ridx, sub_item in enumerate(item):
                    r_label = f"{romans[ridx]})"
                    try:
                        leaf_q = next(leaf_iterator)
                        text = (leaf_q.get("question_content") or "").strip()
                        marks = leaf_q.get("marks", 1)
                    except StopIteration:
                        text = ""
                        marks = sub_item
                    
                    roman_list.append({
                        "label": r_label,
                        "text": text,
                        "marks": marks,
                        "subtopic": subtopic_name
                    })
                parsed.append({
                    "label": label,
                    "context": "",
                    "sub_parts": roman_list,
                    "subtopic": subtopic_name
                })
            else:
                # Simple child question
                try:
                    leaf_q = next(leaf_iterator)
                    text = (leaf_q.get("question_content") or "").strip()
                    marks = leaf_q.get("marks", 1)
                except StopIteration:
                    text = ""
                    marks = item
                    
                parsed.append({
                    "label": label,
                    "text": text,
                    "marks": marks,
                    "subtopic": subtopic_name
                })
        return parsed

    def _execute_generation_task(self, task: dict) -> dict:
        """Executes a single question generation task (called in parallel thread)."""
        mark_structure = task["mark_structure"]
        exam_topic = task["exam_topic"]
        subtopic = task.get("subtopic")
        subtopic_data = task.get("subtopic_data")
        subject = task["subject"]
        q_num = task["number"]

        if isinstance(mark_structure, int):
            # Basic standalone question
            logger.info("Retrieving standalone question Q%d (%d marks) matching subtopic '%s'...", q_num, mark_structure, subtopic)
            
            # Filter candidates from past papers
            # Priority 1: type basic_question and exact marks
            candidates = [
                q for q in self.questions 
                if q.get("type") == "basic_question" and q.get("marks") == mark_structure
            ]
            if not candidates:
                # Priority 2: any question type with the exact marks
                candidates = [q for q in self.questions if q.get("marks") == mark_structure]
            if not candidates:
                # Priority 3: any basic question
                candidates = [q for q in self.questions if q.get("type") == "basic_question"]
            if not candidates:
                # Priority 4: any question
                candidates = list(self.questions)

            # Find the most semantically similar question
            q_best = self._find_most_similar_question(subtopic, candidates)
            question_text = (q_best.get("question_content") or q_best.get("text") or "").strip()

            return {
                "number": f"{q_num})",
                "text": question_text,
                "marks": mark_structure,
                "subtopic": subtopic,
                "q_num": q_num
            }

        else:
            # Parent question with sub-parts
            logger.info("Retrieving parent question Q%d (structure: %s) matching subtopics '%s'...", q_num, mark_structure, subtopic)
            
            # Filter candidates from past papers
            # Priority 1: type parent_question and exact parent_question_structure match
            candidates = [
                q for q in self.questions 
                if q.get("type") == "parent_question" and q.get("parent_question_structure") == mark_structure
            ]
            if not candidates:
                # Priority 2: type parent_question and same flattened marks
                flattened_marks = ExamStructureBuilder.flatten_marks(mark_structure)
                candidates = [
                    q for q in self.questions 
                    if q.get("type") == "parent_question" and ExamStructureBuilder.flatten_marks(q.get("parent_question_structure")) == flattened_marks
                ]
            if not candidates:
                # Priority 3: any parent question
                candidates = [q for q in self.questions if q.get("type") == "parent_question"]
            if not candidates:
                # Priority 4: any question
                candidates = list(self.questions)

            # Find the most semantically similar question using the subtopics summary
            q_best = self._find_most_similar_question(subtopic, candidates)
            
            # Retrieve all child and grandchild questions for the chosen parent question
            p_desc = q_best.get("parent_question_description")
            p_exam = q_best.get("exam")
            related = [
                q for q in self.questions
                if q.get("parent_question_description") == p_desc and q.get("exam") == p_exam
            ]
            leaf_qs = [
                q for q in related 
                if q.get("type") in ("child_question", "grandchild_question")
            ]
            
            # Reconstruct sub_questions matching the exact parent question structure
            top_level_subtopics = task.get("subtopics", [subtopic])
            sub_questions = self._build_copied_sub_questions(
                q_best.get("parent_question_structure") or mark_structure, 
                leaf_qs, 
                top_level_subtopics
            )

            return {
                "number": f"{q_num})",
                "parent_description": q_best.get("parent_question_description", "").strip(),
                "sub_questions": sub_questions,
                "subtopic": subtopic,
                "q_num": q_num
            }

    def generate_exam(
        self,
        exam_topic: str,
        total_marks: int,
        topics: list[str],
        subject: str,
        structure_builder: ExamStructureBuilder,
    ) -> dict[str, Any]:
        """Generate a complete exam using parallel generation.

        Args:
            exam_topic: The broad exam category (e.g., "Higher", "Christianity").
            total_marks: Total marks for the exam.
            topics: List of topic areas to cover.
            subject: The subject name.
            structure_builder: An ExamStructureBuilder instance.

        Returns:
            Dict containing the exam structure and generated questions.
        """
        raw_structure = structure_builder.build_structure(total_marks, exam_topic)
        exam_structure = structure_builder.distribute_to_topics(raw_structure, topics)
        logger.info("Exam structure distributed: %s", exam_structure)

        # Prepare tasks for all questions across all topics
        tasks = []
        question_number = 0

        for topic in exam_structure:
            # 1. Compile specification hierarchy for this topic (uses 1 single structured LLM call)
            spec_tree = self._get_spec_tree_cached(topic, exam_topic)
            subtopics_pool = list(spec_tree.keys())
            used_subtopics: list[str] = []

            for mark in exam_structure[topic]:
                question_number += 1
                
                if isinstance(mark, list):
                    # Parent question: allocate a distinct subtopic for each top-level part
                    selected_subtopics = []
                    selected_subtopics_data = []
                    for _ in range(len(mark)):
                        if not subtopics_pool:
                            subtopics_pool = list(spec_tree.keys())
                        subtopic = self.local_similarity.pick_least_similar(subtopics_pool, used_subtopics)
                        if subtopic in subtopics_pool:
                            subtopics_pool.remove(subtopic)
                        used_subtopics.append(subtopic)
                        selected_subtopics.append(subtopic)
                        selected_subtopics_data.append(spec_tree[subtopic])
                    
                    tasks.append({
                        "number": question_number,
                        "topic": topic,
                        "subtopics": selected_subtopics,
                        "subtopics_data": selected_subtopics_data,
                        "subtopic": ", ".join(selected_subtopics),  # backwards compatibility & summary display
                        "mark_structure": mark,
                        "exam_topic": exam_topic,
                        "subject": subject
                    })
                else:
                    # Basic standalone question
                    if not subtopics_pool:
                        subtopics_pool = list(spec_tree.keys())
                    subtopic = self.local_similarity.pick_least_similar(subtopics_pool, used_subtopics)
                    if subtopic in subtopics_pool:
                        subtopics_pool.remove(subtopic)
                    used_subtopics.append(subtopic)

                    tasks.append({
                        "number": question_number,
                        "topic": topic,
                        "subtopic": subtopic,
                        "subtopic_data": spec_tree[subtopic],
                        "mark_structure": mark,
                        "exam_topic": exam_topic,
                        "subject": subject
                    })

        # 2. Execute all tasks in parallel using a thread pool
        exam_output: dict[str, Any] = {"structure": exam_structure, "questions": {}}
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
            # find original task topic
            q_num = res["q_num"]
            orig_task = next(t for t in tasks if t["number"] == q_num)
            topic = orig_task["topic"]
            
            # remove helper key
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
