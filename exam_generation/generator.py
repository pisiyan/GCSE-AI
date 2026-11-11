"""Parallel and single-question generator for GCSE AI."""

import concurrent.futures
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional
import numpy as np

from config import SubjectConfig
from llm.client import LLMClient
from llm.prompts import load_prompt
from llm.similarity import SimilarityEngine, LocalSimilarityEngine
from exam_generation.structure import ExamStructureBuilder
from exam_generation.formatter import clean_question_text

logger = logging.getLogger(__name__)


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


class QuestionGenerator:
    """Generates GCSE exam questions using parallel, single-pass LLM prompts."""

    def __init__(
        self,
        config: SubjectConfig,
        llm_client: LLMClient,
        similarity_engine: SimilarityEngine,
        questions: List[dict],
        embedding_model: Any = None,
        specification_text: str = "",
        prompts: Optional[Dict[str, str]] = None,
        queries: Optional[Dict[str, str]] = None,
        spec_qa_chain: Any = None,
    ):
        self.config = config
        self.llm = llm_client
        self.similarity = similarity_engine
        self.questions = questions
        self.embedding_model = embedding_model
        self.specification_text = specification_text
        self.prompts = prompts or {}

        # Local fast similarity engine
        fallback_dim = getattr(config, "fallback_embedding_dim", 384)
        self.local_similarity = LocalSimilarityEngine(
            embedding_model, llm_client, fallback_dim=fallback_dim
        )

        # Caching for specification trees and embeddings
        self.spec_trees_cache: Dict[str, dict] = {}
        self._exemplars_by_topic: Dict[str, List[dict]] = {}
        self._topic_embeddings_cache: Dict[str, Any] = {}

    def _get_spec_tree_cached(self, topic: str, exam_type: str = "", subject: str = "") -> dict:
        """Fetch specification tree from pre-computed specification_summary.json or LLM extraction."""
        et = exam_type if exam_type else getattr(self.config, "examiner", "GCSE")
        cache_key = f"{et}-{topic}"
        if cache_key in self.spec_trees_cache:
            return self.spec_trees_cache[cache_key]

        # Fast lookup from specification summary JSON
        subj = subject or getattr(self.config, "subject", "")
        examiner = getattr(self.config, "examiner", "")
        if subj and examiner:
            sum_path = f"data/{subj}/{examiner}/{subj}-{examiner}-specification_summary.json"
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
                                logger.info("Found specification summary lookup for %s: %s (%d subtopics)", cache_key, spec_code, len(subtopics))
                                return tree
                except Exception as e:
                    logger.warning("Failed summary lookup for %s: %s", cache_key, e)

        # Fallback to LLM extraction from full specification text
        logger.info("Extracting specification hierarchy via LLM for: %s", cache_key)
        full_topic_ref = f"{exam_type} - {topic}" if exam_type and exam_type.lower() not in topic.lower() else topic
        prompt = load_prompt(
            "extract_spec_tree",
            full_topic_ref=full_topic_ref,
            exam_type=exam_type or "this exam",
            spec_content=self.specification_text or "No specification details found."
        )

        try:
            tree_data = self.llm.invoke_json(prompt)
            spec_code = tree_data.get("spec_code", topic)
            subtopics = tree_data.get("subtopics", [topic])
            if not subtopics:
                subtopics = [topic]
            tree = {"spec_code": spec_code, "subtopics": subtopics}
            self.spec_trees_cache[cache_key] = tree
            return tree
        except Exception as e:
            logger.error("Failed to parse spec tree via LLM for %s: %s. Using default.", topic, e)
            tree = {"spec_code": topic, "subtopics": [topic]}
            self.spec_trees_cache[cache_key] = tree
            return tree

    def _execute_generation_task(self, task: dict) -> dict:
        """Worker function that generates 1 complete question item."""
        mark_structure = task["mark_structure"]
        exam_type = task["exam_type"]
        subtopic = task["subtopic"]
        subject = task["subject"]
        q_number = task["number"]
        user_prefs = task.get("user_preferences")

        is_parent = isinstance(mark_structure, list)
        total_marks = mark_structure[0] if is_parent else mark_structure

        logger.info("Executing generation task: Q#%d (%d marks, subtopic='%s', parent=%s)", q_number, total_marks, subtopic, is_parent)

        # Filter candidate past paper exemplars for style matching
        candidates = [q for q in self.questions if q.get("marks") == total_marks]
        if not candidates:
            candidates = self.questions

        # Find best matched exemplar using semantic similarity
        matched_exemplar, sim_score = self.local_similarity.match_best(
            subtopic, candidates, text_key="question_content"
        )
        matched_subtopic = matched_exemplar.get("subtopic", subtopic) if matched_exemplar else subtopic

        user_prefs_sec = f"\nUser Preferences:\n{user_prefs}\n" if user_prefs else ""

        if is_parent:
            # Generate parent question with sub-parts
            subparts_spec = f"Create sub-questions summing to {total_marks} marks."
            exemplars_text = clean_question_text(matched_exemplar.get("question_content", "")) if matched_exemplar else ""

            prompt = load_prompt(
                "make_parent_question",
                subject=subject,
                examiner=getattr(self.config, "examiner", "GCSE"),
                total_marks=total_marks,
                subtopic=subtopic,
                subparts_structure=subparts_spec,
                spec_context=f"Focus on {subtopic}",
                exemplars_text=exemplars_text,
                user_preferences_section=user_prefs_sec,
            )

            try:
                q_json = self.llm.invoke_json(prompt, temperature=self.config.temperature_generation)
                return {
                    "number": f"{q_number})",
                    "marks": total_marks,
                    "subtopic": subtopic,
                    "matched_subtopic": matched_subtopic,
                    "similarity_score": sim_score,
                    "parent_description": q_json.get("parent_description", ""),
                    "sub_questions": q_json.get("sub_questions", []),
                    "type": "parent",
                }
            except Exception as e:
                logger.error("Failed parent question generation for Q#%d: %s. Using fallback.", q_number, e)
                return {
                    "number": f"{q_number})",
                    "marks": total_marks,
                    "subtopic": subtopic,
                    "matched_subtopic": matched_subtopic,
                    "similarity_score": sim_score,
                    "parent_description": f"Investigation into {subtopic}.",
                    "sub_questions": [
                        {"label": "(a)", "text": f"Explain the principle of {subtopic}.", "marks": total_marks}
                    ],
                    "type": "parent",
                }

        else:
            # Generate standalone question
            exemplar_text = clean_question_text(matched_exemplar.get("question_content", "")) if matched_exemplar else ""
            prompt = (
                f"You are an expert GCSE {subject} ({getattr(self.config, 'examiner', 'GCSE')}) exam writer.\n"
                f"Create an authentic {total_marks}-mark GCSE exam question testing: '{subtopic}'.\n\n"
                f"Exemplar Style:\n{exemplar_text}\n\n"
                f"{user_prefs_sec}"
                f"Format requirements:\n"
                f"Return ONLY the question prompt text. Do not include answers, mark schemes, or preamble."
            )

            try:
                q_text = self.llm.invoke(prompt, temperature=self.config.temperature_generation).strip()
                return {
                    "number": f"{q_number})",
                    "marks": total_marks,
                    "subtopic": subtopic,
                    "matched_subtopic": matched_subtopic,
                    "similarity_score": sim_score,
                    "text": q_text,
                    "type": "basic",
                }
            except Exception as e:
                logger.error("Failed basic question generation for Q#%d: %s", q_number, e)
                return {
                    "number": f"{q_number})",
                    "marks": total_marks,
                    "subtopic": subtopic,
                    "matched_subtopic": matched_subtopic,
                    "similarity_score": 0.0,
                    "text": f"Explain the significance of {subtopic} ({total_marks} marks).",
                    "type": "basic",
                }

    def _optimize_topic_subtopic_assignments(
        self, exam_structure: Dict[str, List[Any]], exam_type: str, spec_trees: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Assign subtopics to slots in exam structure maximizing match quality and diversity."""
        assignments = {}
        for top, slots in exam_structure.items():
            tree = spec_trees.get(top, {})
            subtopics = tree.get("subtopics", []) if isinstance(tree, dict) else (tree if isinstance(tree, list) else [])
            if not subtopics:
                assignments[top] = [top] * len(slots)
                continue

            top_qs = [q for q in self.questions if q.get("topic") == top or not q.get("topic")]
            assigned_for_topic = []
            used_counts = {st: 0 for st in subtopics}

            for slot in slots:
                slot_marks = slot[0] if isinstance(slot, list) else slot
                slot_type = slot[1] if isinstance(slot, list) and len(slot) > 1 else ("parent" if slot_marks >= 4 else "basic")

                best_st = None
                best_score = -1e9

                for st in subtopics:
                    matching_qs = [
                        q for q in top_qs
                        if str(q.get("subtopic", "")).lower() == st.lower()
                        and ("parent" in str(q.get("type", "")).lower() if slot_type == "parent" else "basic" in str(q.get("type", "")).lower())
                        and q.get("marks") == slot_marks
                    ]
                    base_sim = 1.0 if matching_qs else 0.5
                    usage_penalty = used_counts[st] * 10.0
                    score = base_sim * 100.0 - usage_penalty

                    if hasattr(self.local_similarity, "get_embeddings"):
                        try:
                            embs = self.local_similarity.get_embeddings([st])
                            if embs and len(embs[0]) > 0:
                                score += embs[0][0] * 5.0
                        except Exception:
                            pass

                    if score > best_score:
                        best_score = score
                        best_st = st

                chosen = best_st or subtopics[0]
                assigned_for_topic.append(chosen)
                used_counts[chosen] = used_counts.get(chosen, 0) + 1

            assignments[top] = assigned_for_topic
        return assignments

    def generate_exam(
        self,
        exam_type: str,
        total_marks: int,
        topics: List[str],
        subject: str,
        structure_builder: ExamStructureBuilder,
        user_preferences: Optional[Any] = None,
        num_questions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a full GCSE exam paper across requested topics."""
        logger.info("Starting exam generation for %s (%s) — %d marks across %s", subject, exam_type, total_marks, topics)

        # 1. Build exam structure sequence
        structure = structure_builder.generate_exam_structure(
            exam_type=exam_type,
            total_marks=total_marks,
            num_questions=num_questions,
        )

        # 2. Extract specification hierarchy for each topic
        spec_trees: Dict[str, dict] = {}
        for top in topics:
            spec_trees[top] = self._get_spec_tree_cached(top, exam_type, subject)

        # 3. Distribute question structure items across topics
        tasks = []
        q_idx = 1
        for i, struct_item in enumerate(structure):
            assigned_topic = topics[i % len(topics)]
            tree = spec_trees[assigned_topic]
            subtopics = tree.get("subtopics", [assigned_topic])
            chosen_subtopic = subtopics[(i // len(topics)) % len(subtopics)]

            tasks.append({
                "mark_structure": struct_item,
                "exam_type": exam_type,
                "topic": assigned_topic,
                "subtopic": chosen_subtopic,
                "subject": subject,
                "number": q_idx,
                "user_preferences": user_preferences,
            })
            q_idx += 1

        # 4. Parallel execution of question generation tasks
        max_workers = min(len(tasks), self.config.max_parallel_workers)
        logger.info("Executing %d generation tasks across %d parallel workers...", len(tasks), max_workers)

        generated_questions_by_topic: Dict[str, List[dict]] = {top: [] for top in topics}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(self._execute_generation_task, t): t for t in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                orig_task = future_to_task[future]
                try:
                    q_res = future.result()
                    generated_questions_by_topic[orig_task["topic"]].append(q_res)
                except Exception as e:
                    logger.error("Task failed for topic %s: %s", orig_task["topic"], e)

        # Sort questions inside each topic by original question number
        for top in generated_questions_by_topic:
            generated_questions_by_topic[top].sort(
                key=lambda x: int(x.get("number", "0").rstrip(")")) if x.get("number", "").rstrip(")").isdigit() else 0
            )

        exam_data = {
            "subject": subject,
            "examiner": getattr(self.config, "examiner", "GCSE"),
            "exam_type": exam_type,
            "total_marks": total_marks,
            "spec_trees": spec_trees,
            "questions": generated_questions_by_topic,
        }

        logger.info("Completed exam generation successfully with %d questions.", len(tasks))
        return exam_data
