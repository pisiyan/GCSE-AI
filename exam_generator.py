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

        for retry in range(MAX_STRUCTURE_RETRIES):
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
            f"Could not build valid exam structure after {MAX_STRUCTURE_RETRIES} attempts "
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

    def __init__(self, embedding_model: Any = None, llm_client: Optional[LLMClient] = None):
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.cache: Dict[str, list[float]] = {}

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
                    self.cache[text] = [random.random() for _ in range(384)]

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

    def find_least_similar_objects(
        self,
        objects: list[dict],
        comparison: str,
        n: int,
        topic_value: str = "",
        marks_value: Optional[int] = None,
    ) -> list[str]:
        """Filter past questions, score by similarity, return n LEAST similar to prevent duplicates."""
        filtered = []
        shuffled = objects.copy()
        random.shuffle(shuffled)

        for obj in shuffled:
            matches_topic = (not topic_value) or obj.get("topic") == topic_value
            matches_marks = (marks_value is None) or obj.get("marks") == marks_value
            if matches_topic and matches_marks:
                filtered.append(obj)
            if len(filtered) >= n * 2:  # Subset limit for speed
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
    ):
        self.config = config
        self.llm = llm_client
        self.similarity = similarity_engine
        self.questions = questions
        self.prompts = prompts
        self.queries = queries
        self.spec_qa_chain = spec_qa_chain

        # Initialize local fast similarity engine
        self.local_similarity = LocalSimilarityEngine(embedding_model, llm_client)

        # Cache for specification trees
        self.spec_trees_cache: Dict[str, dict] = {}

    def _get_spec_tree_cached(self, topic: str, exam_topic: str) -> dict:
        """Fetch specification chunks and compile them into a structured tree using a single LLM call."""
        cache_key = f"{exam_topic}-{topic}"
        if cache_key in self.spec_trees_cache:
            return self.spec_trees_cache[cache_key]

        logger.info("Extracting and compiling specification hierarchy for: %s", cache_key)
        
        # 1. Retrieve raw spec documents directly from FAISS without LLM
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
        """Generate a single exam question (basic/standalone)."""
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
        if prompt_extension:
            prompt += "\n" + prompt_extension

        return self.llm.invoke(prompt)

    def _execute_generation_task(self, task: dict) -> dict:
        """Executes a single question generation task (called in parallel thread)."""
        mark_structure = task["mark_structure"]
        exam_topic = task["exam_topic"]
        subtopic = task["subtopic"]
        subtopic_data = task["subtopic_data"]
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
            matched_descs = []
            for _ in range(min(3, len(descs))):
                matched = self.local_similarity.find_most_similar(subtopic, descs)
                matched_descs.append(matched)
                if matched in descs:
                    descs.remove(matched)

            for desc in matched_descs:
                group = parent_groups[desc]
                group_text = f"Parent Context:\n{desc}\nSub-questions:\n"
                for sq in group:
                    marks = sq.get("marks", 1)
                    q_text = sq.get("question_content", "") or sq.get("text", "")
                    group_text += f"- ({marks} marks): {q_text}\n"
                exemplar_texts.append(group_text)

        exemplars_joined = "\n\n---\n\n".join(exemplar_texts) if exemplar_texts else "No exam exemplars available."

        # 2. Assign distinct specification sub-subtopics to each leaf question part to prevent redundancies
        sub_subtopics = subtopic_data.get("sub_subtopics", [])
        # Shuffle pool for random distinct assignment
        shuffled_sst = sub_subtopics.copy()
        random.shuffle(shuffled_sst)
        
        sst_idx = 0
        def assign_recursively(structure, letters="abcdefghijklmn", romans=["i", "ii", "iii", "iv", "v"], depth=0, parent_label=""):
            nonlocal sst_idx
            res = []
            for idx, item in enumerate(structure):
                if depth == 0:
                    label = f"{letters[idx]})"
                    part_name = f"Part {label}"
                else:
                    label = f"{romans[idx]})"
                    part_name = f"Sub-part {parent_label} {label}"

                if isinstance(item, list):
                    sub_res = assign_recursively(item, letters, romans, depth + 1, label)
                    res.append({
                        "label": label,
                        "type": "sub_parent",
                        "context_desc": f"Context for {part_name} sub-parts...",
                        "sub_parts": sub_res
                    })
                else:
                    # Retrieve next distinct spec point from pool
                    if shuffled_sst:
                        sst = shuffled_sst[sst_idx % len(shuffled_sst)]
                        sst_idx += 1
                        concept_desc = f"Concept: {sst.get('name')} (Requirements: {sst.get('description')})"
                    else:
                        concept_desc = f"Concept: General details of {subtopic}"
                    
                    res.append({
                        "label": label,
                        "type": "basic",
                        "marks": item,
                        "concept": concept_desc
                    })
            return res

        assigned_structure = assign_recursively(mark_structure)

        # 3. Format the assigned target structure text block for the LLM prompt
        def format_assigned_text(structure, indent=""):
            lines = []
            for item in structure:
                label = item["label"]
                if item["type"] == "sub_parent":
                    lines.append(f"{indent}- Part {label}: Sub-parent context section containing Roman numeral sub-parts.")
                    lines.extend(format_assigned_text(item["sub_parts"], indent + "  "))
                else:
                    lines.append(f"{indent}- Part {label} ({item['marks']} marks): MUST test exactly the following {item['concept']}.")
            return lines

        parts_text = "\n".join(format_assigned_text(assigned_structure))

        prompt = f"""You are an expert GCSE {subject} question writer.
Write a multi-part exam question.

Main Topic: {task['topic']}
Subtopic: {subtopic}
Specification Details: {subtopic_data['description']}

Here is the exact target question parts structure to generate, along with the distinct concept assigned to each part:
{parts_text}

Here are past paper exam exemplars showing the tone, style, and structure to replicate:
{exemplars_joined}

Rules:
1. Replicate the formatting and scientific tone of the exemplars, but do NOT copy their content.
2. For each question, end with answer lines represented by dotted lines (like '........................................') matching the mark allocation (more marks = more lines).
3. For multiple-choice questions (1 mark parts), provide A, B, C, D options and do not include dotted lines.
4. Ensure parts testing the same scenario are coherent, but test exactly the assigned distinct concept to avoid duplicate testing and redundancy.
5. Return the result strictly as a JSON object matching this schema:
{{
  "parent_description": "concise intro context scenario text matching the exemplars. DO NOT include question numbers here.",
  "sub_questions": [
    {{
      "label": "a)",
      "type": "basic",
      "text": "Question text for part a) including the dotted lines",
      "marks": 3
    }},
    {{
      "label": "b)",
      "type": "sub_parent",
      "context": "Context paragraph for parts i, ii, iii (if needed, otherwise leave empty)",
      "sub_parts": [
        {{
          "label": "i)",
          "text": "Question text for part i) here...",
          "marks": 1
        }}
      ]
    }}
  ]
}}
Do not return any explanations, markdown code blocks other than ```json, or other text. Output only raw JSON."""

        try:
            res_json = self.llm.invoke_json(prompt)
        except Exception as e:
            logger.error("JSON generation failed for Q%d: %s. Retrying with basic prompts.", q_num, e)
            res_json = {
                "parent_description": f"Figure shows details about {subtopic}.",
                "sub_questions": [{"label": "a)", "type": "basic", "text": f"Describe {subtopic}.", "marks": 3}]
            }

        # 4. Clean and format the JSON response to exactly match the expected schema
        sub_questions = self._parse_llm_json_to_exam_structure(res_json)

        return {
            "number": f"{q_num})",
            "parent_description": res_json.get("parent_description", ""),
            "sub_questions": sub_questions,
            "subtopic": subtopic,
            "q_num": q_num
        }

    def _parse_llm_json_to_exam_structure(self, res_json: dict, letters: str = "abcdefghijklmn", romans: list = None) -> list[dict]:
        """Convert LLM JSON output to the format expected by GcseAssistant / ExamQualityAnalyzer."""
        if romans is None:
            romans = ["i", "ii", "iii", "iv", "v", "vi"]
        
        parsed = []
        for idx, sq in enumerate(res_json.get("sub_questions", [])):
            label = f"{letters[idx]})"
            
            # Check if it has grandchild sub-parts
            sub_parts = sq.get("sub_parts")
            if sub_parts or sq.get("type") == "sub_parent":
                roman_list = []
                for ridx, rq in enumerate(sub_parts or []):
                    r_label = f"{romans[ridx]})"
                    roman_list.append({
                        "label": r_label,
                        "text": rq.get("text", "") or rq.get("question", ""),
                        "marks": rq.get("marks", 1)
                    })
                parsed.append({
                    "label": label,
                    "context": sq.get("context", "") or sq.get("parent_description", "") or "",
                    "sub_parts": roman_list
                })
            else:
                parsed.append({
                    "label": label,
                    "text": sq.get("text", "") or sq.get("question", ""),
                    "marks": sq.get("marks", 1)
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
                
                # Pick a subtopic that's different from ones already used
                if not subtopics_pool:
                    subtopics_pool = list(spec_tree.keys())
                subtopic = self.local_similarity.pick_least_similar(subtopics_pool, used_subtopics)
                if subtopic in subtopics_pool:
                    subtopics_pool.remove(subtopic)
                used_subtopics.append(subtopic)

                subtopic_data = spec_tree[subtopic]

                tasks.append({
                    "number": question_number,
                    "topic": topic,
                    "subtopic": subtopic,
                    "subtopic_data": subtopic_data,
                    "mark_structure": mark,
                    "exam_topic": exam_topic,
                    "subject": subject
                })

        # 2. Execute all tasks in parallel using a thread pool
        exam_output: dict[str, Any] = {"structure": exam_structure, "questions": {}}
        results = []
        
        logger.info("Submitting %d question generation tasks to ThreadPoolExecutor...", len(tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(tasks) or 1)) as executor:
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
        query = self.queries["revision_materials"].format(
            topic=topic, subject=subject
        )
        return self.llm.invoke_qa(spec_qa_chain, query)
