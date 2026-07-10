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

    def generate_question(
        self,
        marks: int,
        exam_topic: str,
        subtopic: str,
        subtopic_info: str,
        question_content: str = "",
        parent_description: str = "",
        subject: str = "",
        examiner: str = ""
    ) -> str:
        """Generate a single exam question (basic/standalone) with self-critique/refinement."""
        example_questions = self.local_similarity.find_least_similar_objects(
            objects=self.questions,
            comparison=subtopic,
            n=self.config.example_questions,
            topic_value=exam_topic,
            marks_value=marks,
        )

        prompt_extension = ""
        if question_content:
            prompt_extension = self.prompts["question_prompt_extention"].format(
                question_content=question_content,
                parent_description=parent_description,
            )

        prompt = self.prompts["make_question"].format(
            subject=subject or getattr(self.config, "subject", "GCSE") or "GCSE",
            examiner=examiner or getattr(self.config, "examiner", "") or "",
            marks=marks,
            random_questions="\n\n".join(example_questions),
            topic_info=subtopic_info,
            subtopic=subtopic,
        )
        
        # Append calibration examples to guide mark weight depth
        prompt += "\n\n" + self._get_mark_calibration_examples()

        if prompt_extension:
            prompt += "\n" + prompt_extension

        temp_gen = getattr(self.config, "temperature_generation", 0.7)
        temp_struct = getattr(self.config, "temperature_structure", 0.0)

        # 1. Draft the question
        draft = self.llm.invoke(prompt, temperature=temp_gen)

        # 2. Critique the draft
        critique_prompt = f"""You are a senior GCSE {subject or 'science'} examiner.
Critique the drafted question against GCSE standards:
DRAFT QUESTION:
{draft}

GCSE standard requirements:
- The question must be scientifically accurate, clear, and concise.
- If it involves calculations (e.g., math, equations), it MUST provide realistic numerical values/units and ask the student to show their working.
- If it involves practical investigations, graphs, data, or experiments, it must focus on interpreting or evaluating that experimental setup.
- It MUST end with answer lines represented by dotted lines (like '........................................') proportional to the mark allocation (more marks = more lines), unless it is a multiple-choice question.
- Do NOT include question numbers or the mark value inside the question text.

Output a brief list of critique points or state 'Approved' if the question is perfect.
Do not write a revised question here, only write the critique feedback."""

        critique = self.llm.invoke(critique_prompt, temperature=temp_struct)

        # If approved, return draft directly
        if "approved" in critique.strip().lower() and len(critique.strip()) < 20:
            return draft

        # 3. Refine the question
        refinement_prompt = f"""You are an expert GCSE {subject or 'science'} question writer.
Revise the draft question below to address the examiner feedback and ensure it meets all GCSE standards.

DRAFT QUESTION:
{draft}

EXAMINER FEEDBACK:
{critique}

Rules:
- Output ONLY the final revised question text.
- Do not include explanations, intro text, question numbers, or markdown other than the question content itself.
- Ensure the question ends with dotted lines for answer space (proportional to {marks} marks)."""

        refined = self.llm.invoke(refinement_prompt, temperature=temp_gen)
        return refined

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
            logger.info("Generating standalone question Q%d (%d marks)...", q_num, mark_structure)
            question_text = self.generate_question(
                marks=mark_structure,
                exam_topic=exam_topic,
                subtopic=subtopic,
                subtopic_info=subtopic_data["description"],
                subject=subject
            )
            return {
                "number": f"{q_num})",
                "text": question_text,
                "marks": mark_structure,
                "subtopic": subtopic,
                "q_num": q_num
            }

        # Parent Question with sub-parts
        logger.info("Generating parent question Q%d (structure: %s)...", q_num, mark_structure)

        # 1. Collect past parent questions for style exemplars
        parent_exemplars = [
            q for q in self.questions
            if q.get("type") == "parent_question" and q.get("topic") == exam_topic
        ]
        if not parent_exemplars:
            parent_exemplars = [q for q in self.questions if q.get("type") == "parent_question"]

        # Group flat past questions by parent description to reconstruct complete multi-part questions
        parent_groups = {}
        for q in self.questions:
            desc = q.get("parent_question_description") or q.get("parent_context")
            if desc:
                parent_groups.setdefault(desc, []).append(q)

        exemplar_texts = []
        if parent_groups:
            # Find parent descriptions matching semantic context
            descs = list(parent_groups.keys())
            comp_string = task["topic"]  # use broad topic name for best contextual matches
            max_exemplars = getattr(self.config, "max_parent_exemplars", 3)
            
            # Make a pool of the top max_parent_exemplars * 2 most similar questions to the broad topic
            pool_size = max_exemplars * 2
            candidate_pool = []
            temp_descs = descs.copy()
            for _ in range(min(pool_size, len(temp_descs))):
                matched = self.local_similarity.find_most_similar(comp_string, temp_descs)
                candidate_pool.append(matched)
                if matched in temp_descs:
                    temp_descs.remove(matched)
            
            # From within the candidate pool, pick max_exemplars that are least similar to each other
            matched_descs = self.local_similarity.pick_diverse_subset(candidate_pool, min(max_exemplars, len(candidate_pool)))

            for desc in matched_descs:
                group = parent_groups[desc]
                group_text = f"Parent Context:\n{desc}\nSub-questions:\n"
                for sq in group:
                    marks = sq.get("marks", 1)
                    q_text = sq.get("question_content", "") or sq.get("text", "")
                    group_text += f"- ({marks} marks): {q_text}\n"
                exemplar_texts.append(group_text)

        exemplars_joined = "\n\n---\n\n".join(exemplar_texts) if exemplar_texts else "No exam exemplars available."

        # 2. Assign distinct specification subtopics and sub-subtopics to each leaf question part to prevent redundancies
        top_level_subtopics = task["subtopics"]
        top_level_subtopics_data = task["subtopics_data"]

        shuffled_ssts = []
        for sst_data in top_level_subtopics_data:
            sst_list = sst_data.get("sub_subtopics", []).copy()
            random.shuffle(sst_list)
            shuffled_ssts.append(sst_list)

        def assign_recursively(structure, letters="abcdefghijklmn", romans=["i", "ii", "iii", "iv", "v"], depth=0, parent_label="", top_idx=0, sst_indices=None):
            if sst_indices is None:
                sst_indices = [0] * len(top_level_subtopics)
            res = []
            for idx, item in enumerate(structure):
                if depth == 0:
                    label = f"{letters[idx]})"
                    part_name = f"Part {label}"
                    current_top_idx = idx
                else:
                    label = f"{romans[idx]})"
                    part_name = f"Sub-part {parent_label} {label}"
                    current_top_idx = top_idx

                subtopic_name = top_level_subtopics[current_top_idx]
                subtopic_info = top_level_subtopics_data[current_top_idx]["description"]
                shuffled_sst = shuffled_ssts[current_top_idx]

                if isinstance(item, list):
                    # Pass the current top_idx down to grandchild parts
                    sub_res = assign_recursively(item, letters, romans, depth + 1, label, current_top_idx, sst_indices)
                    res.append({
                        "label": label,
                        "type": "sub_parent",
                        "subtopic": subtopic_name,
                        "subtopic_info": subtopic_info,
                        "context_desc": f"Context for {part_name} sub-parts...",
                        "sub_parts": sub_res
                    })
                else:
                    if shuffled_sst:
                        sst_idx = sst_indices[current_top_idx]
                        sst = shuffled_sst[sst_idx % len(shuffled_sst)]
                        sst_indices[current_top_idx] += 1
                        concept_desc = f"Concept: {sst.get('name')} (Requirements: {sst.get('description')})"
                    else:
                        concept_desc = f"Concept: General details of {subtopic_name}"

                    res.append({
                        "label": label,
                        "type": "basic",
                        "marks": item,
                        "subtopic": subtopic_name,
                        "subtopic_info": subtopic_info,
                        "concept": concept_desc
                    })
            return res

        sst_indices = [0] * len(top_level_subtopics)
        assigned_structure = assign_recursively(mark_structure, sst_indices=sst_indices)

        # 3. Format the assigned target structure text block for the LLM prompt
        def format_assigned_text(structure, indent=""):
            lines = []
            for item in structure:
                label = item["label"]
                if item["type"] == "sub_parent":
                    lines.append(f"{indent}- Part {label}: Sub-parent context section containing Roman numeral sub-parts (Subtopic: {item['subtopic']}).")
                    lines.extend(format_assigned_text(item["sub_parts"], indent + "  "))
                else:
                    lines.append(f"{indent}- Part {label} ({item['marks']} marks): MUST test exactly the following {item['concept']} (from Subtopic: {item['subtopic']}).")
            return lines

        parts_text = "\n".join(format_assigned_text(assigned_structure))

        # Compile subtopics details list
        subtopics_details_list = []
        for s_name, s_data in zip(top_level_subtopics, top_level_subtopics_data):
            subtopics_details_list.append(f"- Subtopic: {s_name}\n  Details: {s_data['description']}")
        subtopics_details_text = "\n".join(subtopics_details_list)

        prompt = f"""You are an expert GCSE {subject} question writer.
Write a multi-part exam question.

Main Topic: {task['topic']}
Subtopics Covered:
{subtopics_details_text}

Here is the exact target question parts structure to generate, along with the distinct concept assigned to each part:
{parts_text}

Here are past paper exam exemplars showing the tone, style, and structure to replicate:
{exemplars_joined}

{self._get_mark_calibration_examples()}

Parent Description Guidelines:
1. The parent description MUST NOT be a basic textbook definition or general summary of the topic.
2. It should setup a realistic scenario: for example, a specific case study, a practical/experimental setup (e.g. "A student investigated..."), a graph description, a data table, or a diagram.
3. Crucial: If the parent description describes a practical investigation or experiment where students collected data (or if any sub-question asks to calculate values, rates, or percentages from an experiment):
   - You MUST include a formatted markdown data table presenting this collected data.
   - The table columns, labels, and units must follow standard subject/exam board conventions (e.g. columns for independent and dependent variables, with units in headers like 'Temperature / °C' or 'Time / s').
   - Provide realistic, concrete values.
   - Any subsequent calculation parts must refer to this table and use its values.
4. If the scenario involves a graph or diagram, describe its key values, labels, or trends clearly in text so subsequent sub-questions can refer to it logically.
5. Make the language active, authentic, and scientific, matching real GCSE past papers.

Question Generation Rules:
1. Replicate the formatting and scientific tone of the exemplars, but do NOT copy their content.
2. For each question, end with answer lines represented by dotted lines (like '........................................') matching the mark allocation (more marks = more lines).
3. For multiple-choice questions (1 mark parts), provide A, B, C, D options and do not include dotted lines.
4. Ensure parts testing the same scenario are coherent, but test exactly the assigned distinct concept to avoid duplicate testing and redundancy.
5. If an assigned concept or subtopic description mentions calculations, math, formulas, equations, rates, percentages, or mathematical operations:
   - The question part MUST be a calculation question.
   - Provide concrete, realistic numerical values and units.
   - Instruct the student to show their working (e.g., "State the formula used and show your working.").
6. Data Presentation & Calculation Link in Practicals: If the scenario involves a practical investigation, experiment, or core practical where data was collected (or if subsequent parts require calculations using experimental data):
   - You MUST present all experimental data in a structured markdown data table in the `parent_description` or the sub-question `context`.
   - The table columns, labels, and units must follow standard subject/exam board conventions (e.g., `Time / s`, `Temperature / °C`, `Volume / cm³`).
   - Any calculation question parts MUST reference this table and be mathematically solvable using the specific values provided in it.
7. Context Formatting & Part a) direct start:
   - The first sub-question (part a)) MUST ask a question directly after the main `parent_description` without introducing any extra context or introductory paragraph. Do not write a context paragraph for part a) (keep `"context"` empty or very short).
   - Extra context/scenario descriptions (e.g. `"context"` in the JSON schema) are only allowed when introducing subsequent parts like b) or c) if a new detail, table, or variable is introduced.
8. Return the result strictly as a JSON object matching this schema:
{{
  "parent_description": "intro context scenario setup (case study, experiment, graph/table description). DO NOT include question numbers here.",
  "sub_questions": [
    {{
      "label": "a)",
      "type": "basic",
      "text": "Question text for part a) including the dotted lines",
      "marks": 3,
      "subtopic": "Exact Name of the Subtopic being tested"
    }},
    {{
      "label": "b)",
      "type": "sub_parent",
      "context": "Context paragraph for parts i, ii, iii (if needed, otherwise leave empty)",
      "sub_parts": [
        {{
          "label": "i)",
          "text": "Question text for part i) here...",
          "marks": 1,
          "subtopic": "Exact Name of the Subtopic being tested"
        }}
      ]
    }}
  ]
}}
Do not return any explanations, markdown code blocks other than ```json, or other text. Output only raw JSON.
"""

        temp_gen = getattr(self.config, "temperature_generation", 0.7)
        temp_struct = getattr(self.config, "temperature_structure", 0.0)

        # 1. Draft the parent question JSON
        try:
            res_json = self.llm.invoke_json(prompt, temperature=temp_gen)
        except Exception as e:
            logger.error("JSON generation failed for Q%d: %s. Retrying with basic prompts.", q_num, e)
            res_json = {
                "parent_description": f"Figure shows details about {subtopic}.",
                "sub_questions": [{"label": "a)", "type": "basic", "text": f"Describe {subtopic}.", "marks": 3}]
            }

        # 2. Critique the draft JSON
        critique_prompt = f"""You are a senior GCSE {subject or 'science'} examiner.
Critique this drafted multi-part GCSE question for quality, scientific accuracy, and originality:

DRAFT QUESTION JSON:
{json.dumps(res_json, indent=2)}

Assigned subtopics/concepts to verify:
{parts_text}

Verify these GCSE requirements:
1. Structure Adherence: The generated sub-question labels (a, b, etc. and their sub-parts i, ii, etc.) and their mark allocations MUST MATCH the requested formatting requirements EXACTLY. Omit no parts and do not merge parts or change their marks.
2. Originality: The "parent_description" MUST NOT copy past papers or textbooks. It should present a unique, realistic scenario, such as a student's experimental setup, a graph, or data. (Sub-questions themselves can be standard similar questions, but the setup/scenario must be unique).
3. Scientific Accuracy: Are the values, facts, and scientific terminology correct?
4. Concept Adherence: Do the sub-questions match their assigned concept requirements (e.g. calculation parts must be calculation questions with numbers and units; practical parts must ask for data interpretation/evaluation)?
5. Formatting: Do basic question texts end with dotted lines matching the mark allocation (more marks = more lines), and multiple choice questions have A, B, C, D choices without dotted lines?
6. Data Presentation & Solvability: If this is an experimental/practical question, is the data presented as a formatted markdown table with columns and units? Are calculation parts solvable using values from this table?
7. Direct Start for Part a): Does the first sub-question (part a) or its sub-parts) ask a question directly after the main parent_description without any separate context/introductory paragraph? (Part a's `"context"` should be empty/omitted).

Output a list of improvements or state 'Approved' if the question is perfect.
Do not output a revised question here, only write the critique feedback."""

        try:
            critique = self.llm.invoke(critique_prompt, temperature=temp_struct)
            
            # If not approved, refine the JSON
            if not ("approved" in critique.strip().lower() and len(critique.strip()) < 20):
                refinement_prompt = f"""You are an expert GCSE {subject or 'science'} question writer.
Revise the draft question JSON to address the examiner feedback and ensure it meets all GCSE standards.

DRAFT QUESTION JSON:
{json.dumps(res_json, indent=2)}

EXAMINER FEEDBACK:
{critique}

Original formatting requirements:
{parts_text}

Output ONLY a valid JSON object matching this schema:
{{
  "parent_description": "intro context scenario setup (case study, experiment, graph/table description). DO NOT include question numbers here.",
  "sub_questions": [
    {{
      "label": "a)",
      "type": "basic",
      "text": "Question text for part a) including the dotted lines",
      "marks": 3,
      "subtopic": "Exact Name of the Subtopic being tested"
    }},
    {{
      "label": "b)",
      "type": "sub_parent",
      "context": "Context paragraph for parts i, ii, iii (if needed, otherwise leave empty)",
      "sub_parts": [
        {{
          "label": "i)",
          "text": "Question text for part i) here...",
          "marks": 1,
          "subtopic": "Exact Name of the Subtopic being tested"
        }}
      ]
    }}
  ]
}}
Do not return any explanations, markdown code blocks other than ```json, or other text. Output only raw JSON."""

                refined_json = self.llm.invoke_json(refinement_prompt, temperature=temp_gen)
                res_json = refined_json
        except Exception as e:
            logger.warning("Self-critique or refinement failed for Q%d: %s. Using original draft.", q_num, e)

        # 4. Clean and format the JSON response to exactly match the expected schema
        sub_questions = self._parse_llm_json_to_exam_structure(res_json, assigned_structure)

        return {
            "number": f"{q_num})",
            "parent_description": res_json.get("parent_description", ""),
            "sub_questions": sub_questions,
            "subtopic": subtopic,
            "q_num": q_num
        }

    def _parse_llm_json_to_exam_structure(self, res_json: dict, assigned_structure: list, letters: str = "abcdefghijklmn", romans: list = None) -> list[dict]:
        """Convert LLM JSON output to the format expected by GcseAssistant / ExamQualityAnalyzer, appending assigned subtopics."""
        if romans is None:
            romans = ["i", "ii", "iii", "iv", "v", "vi"]
        
        parsed = []
        for idx, sq in enumerate(res_json.get("sub_questions", [])):
            label = f"{letters[idx]})"
            
            # Get subtopic from JSON, fallback to assigned_structure
            subtopic = sq.get("subtopic")
            assigned_item = None
            if not subtopic:
                assigned_item = next((item for item in assigned_structure if item["label"] == label), None) if assigned_structure else None
                subtopic = assigned_item["subtopic"] if assigned_item else ""
            
            # Check if it has grandchild sub-parts
            sub_parts = sq.get("sub_parts")
            if sub_parts or sq.get("type") == "sub_parent":
                roman_list = []
                for ridx, rq in enumerate(sub_parts or []):
                    r_label = f"{romans[ridx]})"
                    
                    # Find corresponding grandchild subtopic
                    gc_subtopic = rq.get("subtopic")
                    if not gc_subtopic:
                        gc_subtopic = subtopic
                        # Fallback to assigned structure if needed
                        if not assigned_item and assigned_structure:
                            assigned_item = next((item for item in assigned_structure if item["label"] == label), None)
                        if assigned_item and "sub_parts" in assigned_item:
                            gc_item = next((item for item in assigned_item["sub_parts"] if item["label"] == r_label), None)
                            if gc_item:
                                gc_subtopic = gc_item["subtopic"]
                    
                    roman_list.append({
                        "label": r_label,
                        "text": rq.get("text", "") or rq.get("question", ""),
                        "marks": rq.get("marks", 1),
                        "subtopic": gc_subtopic
                    })
                parsed.append({
                    "label": label,
                    "context": sq.get("context", "") or sq.get("parent_description", "") or "",
                    "sub_parts": roman_list,
                    "subtopic": subtopic
                })
            else:
                parsed.append({
                    "label": label,
                    "text": sq.get("text", "") or sq.get("question", ""),
                    "marks": sq.get("marks", 1),
                    "subtopic": subtopic
                })
        return parsed

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
