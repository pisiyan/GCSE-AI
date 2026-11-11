"""Exam structure planning and pattern synthesis for GCSE AI."""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from config import SubjectConfig

logger = logging.getLogger(__name__)

MAX_STRUCTURE_RETRIES = 50


def _filter_by_exam_type(questions: List[dict], exam_type: str) -> List[dict]:
    """Filter question dicts by their exam_type field."""
    et_lower = exam_type.strip().lower()
    matched = []
    unspecified = []
    for q in questions:
        q_et = str(q.get("exam_type", "")).strip().lower()
        if not q_et:
            unspecified.append(q)
        elif et_lower in q_et or q_et in et_lower:
            matched.append(q)
    return matched if matched else unspecified


class ExamStructureBuilder:
    """Builds valid exam structures conforming to mark totals, tiers, and past paper patterns."""

    def __init__(self, config: Optional[SubjectConfig] = None, questions: Optional[List[dict]] = None):
        self.config = config or SubjectConfig()
        self.questions = questions or []

    @staticmethod
    def flatten_marks(item: Any) -> int:
        """Calculate total marks from an int, list, or nested list."""
        if isinstance(item, int):
            return item
        if isinstance(item, list):
            # Check if [marks, type_str]
            if len(item) == 2 and isinstance(item[0], int) and isinstance(item[1], str):
                return item[0]
            total = 0
            for sub in item:
                total += ExamStructureBuilder.flatten_marks(sub)
            return total
        try:
            return int(item)
        except (ValueError, TypeError):
            return 1

    @staticmethod
    def distribute_to_topics(items: List[Any], topics: List[str]) -> Dict[str, List[Any]]:
        """Distribute structure items as evenly as possible across topic names."""
        if not topics:
            raise ValueError("Topics list cannot be empty.")
        k, m = divmod(len(items), len(topics))
        distribution: Dict[str, List[Any]] = {}
        start = 0
        for idx, t in enumerate(topics):
            chunk_size = k + 1 if idx < m else k
            distribution[t] = items[start:start + chunk_size]
            start += chunk_size
        return distribution

    def get_available_marks(self, exam_type: str) -> List[Tuple[int, str]]:
        """Extract unique question mark sizes and question types available for an exam type."""
        mark_types = set()
        for q in self.questions:
            q_et = q.get("exam_type", "")
            q_topic = q.get("topic", "")
            if exam_type.lower() in q_et.lower() or exam_type.lower() in q_topic.lower() or not q_et:
                marks = q.get("marks")
                q_type = "parent" if "parent" in str(q.get("type", "")).lower() else "basic"
                if isinstance(marks, int) and marks > 0:
                    mark_types.add((marks, q_type))

        if not mark_types:
            for q in self.questions:
                marks = q.get("marks")
                q_type = "parent" if "parent" in str(q.get("type", "")).lower() else "basic"
                if isinstance(marks, int) and marks > 0:
                    mark_types.add((marks, q_type))

        if not mark_types:
            mark_types = {(1, "basic"), (2, "basic"), (3, "basic"), (4, "basic"), (6, "basic")}

        result = sorted(list(mark_types), key=lambda x: x[0])
        logger.info("Recorded %d valid question mark options: %s", len(result), result)
        return result

    def get_past_exam_structures(self, topic: str = "") -> List[List[Any]]:
        """Extract exact question structures grouped by exam from past papers."""
        papers: Dict[str, List[Any]] = {}
        for q in self.questions:
            if topic and q.get("topic") != topic:
                continue
            exam_name = q.get("exam", "default")
            if exam_name not in papers:
                papers[exam_name] = []

            q_type = "parent" if "parent" in str(q.get("type", "")).lower() else "basic"
            if q_type == "parent":
                p_struct = q.get("parent_question_structure", [])
                marks = sum(p_struct) if isinstance(p_struct, list) and p_struct else q.get("marks", 3)
            else:
                marks = q.get("marks", 1)

            papers[exam_name].append([marks, q_type])

        return list(papers.values())

    def get_position_options(self, topic: str = "") -> Dict[int, List[Tuple[int, str]]]:
        """Extract allowed (marks, type) options for each question position index."""
        pos_options: Dict[int, set] = {}
        for structure in self.get_past_exam_structures(topic):
            for pos, item in enumerate(structure):
                if pos not in pos_options:
                    pos_options[pos] = set()
                pos_options[pos].add((item[0], item[1]))
        return {pos: list(opts) for pos, opts in pos_options.items()}

    def build_structure(
        self,
        total_marks: int,
        topic: str = "",
        num_questions: Optional[int] = None,
    ) -> List[Any]:
        """Alias for generate_exam_structure for backward compatibility."""
        return self.generate_exam_structure(
            exam_type=topic,
            total_marks=total_marks,
            num_questions=num_questions,
        )

    def generate_exam_structure(
        self,
        exam_type: str,
        total_marks: int,
        num_questions: Optional[int] = None,
    ) -> List[Any]:
        """Generate a valid sequence of question marks summing to total_marks."""
        available = self.get_available_marks(exam_type)
        past_structures = self.get_past_exam_structures(exam_type)

        # Tier 1: Exact past paper match / tiling if marks align
        for past_seq in past_structures:
            seq_sum = sum(item[0] if isinstance(item, list) else item for item in past_seq)
            if seq_sum == total_marks:
                if num_questions is None or len(past_seq) == num_questions:
                    logger.info("Tier 1: Matched exact past paper structure: %s", past_seq)
                    return list(past_seq)

            if seq_sum > 0 and total_marks % seq_sum == 0:
                repeat = total_marks // seq_sum
                candidate = past_seq * repeat
                if num_questions is None or len(candidate) == num_questions:
                    logger.info("Tier 1: Matched block pattern tiling (%d questions, %d marks): %s", len(candidate), total_marks, candidate)
                    return candidate

        # Tier 2: Constrained question count solver
        if num_questions is not None and num_questions > 0:
            target_count = num_questions
            options = [[item[0], item[1]] for item in available]

            def backtrack_fixed(curr_seq: List[Any], curr_sum: int) -> Optional[List[Any]]:
                if len(curr_seq) == target_count:
                    return curr_seq if curr_sum == total_marks else None
                remaining_q = target_count - len(curr_seq)
                remaining_marks = total_marks - curr_sum
                if remaining_marks < remaining_q * 1 or remaining_marks > remaining_q * 20:
                    return None

                shuffled_opts = list(options)
                random.shuffle(shuffled_opts)
                for opt in shuffled_opts:
                    if curr_sum + opt[0] <= total_marks:
                        res = backtrack_fixed(curr_seq + [opt], curr_sum + opt[0])
                        if res is not None:
                            return res
                return None

            res = backtrack_fixed([], 0)
            if res:
                logger.info("Tier 1: Matched block pattern tiling (%d questions, %d marks): %s", len(res), total_marks, res)
                return res

        # Tier 3: Random backtracking solver
        options = [[item[0], item[1]] for item in available]
        for attempt in range(MAX_STRUCTURE_RETRIES):
            random.shuffle(options)
            curr = []
            curr_sum = 0
            while curr_sum < total_marks:
                valid_next = [o for o in options if curr_sum + o[0] <= total_marks]
                if not valid_next:
                    break
                choice = random.choice(valid_next)
                curr.append(choice)
                curr_sum += choice[0]

            if curr_sum == total_marks:
                if num_questions is None or len(curr) == num_questions:
                    logger.info("Tier 1: Matched block pattern tiling (%d marks): %s", total_marks, curr)
                    return curr

        # Fallback simple structure
        logger.warning("Solver reached fallback structure for %d marks.", total_marks)
        return [[4, "basic"]] * (total_marks // 4) + ([[total_marks % 4, "basic"]] if total_marks % 4 else [])
