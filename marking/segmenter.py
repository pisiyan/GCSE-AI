"""Exam paper and answer script segmentation module for GCSE AI."""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from config import SubjectConfig

logger = logging.getLogger(__name__)


def extract_mark_val(text: str, pattern: str = r"\((\d+)\)") -> Optional[int]:
    """Extract integer mark value from text using pattern with common fallbacks."""
    if not text:
        return None
    try:
        regex = re.compile(pattern)
        matches = list(regex.finditer(text))
        if matches:
            last_match = matches[-1]
            if last_match.groups():
                return int(last_match.group(1))
            return int(last_match.group())
    except (ValueError, TypeError, re.error):
        pass

    fallback_patterns = [
        r"\((\d{1,2})\)",
        r"\[\s*(\d{1,2})\s*(?:marks?)?\s*\]",
        r"(?i)(\d{1,2})\s*marks?"
    ]
    for fb_pat in fallback_patterns:
        fb_matches = list(re.finditer(fb_pat, text))
        if fb_matches:
            try:
                return int(fb_matches[-1].group(1))
            except (ValueError, TypeError):
                pass
    return None


class ExamSegmenter:
    """Segments raw exam text into individual question items, mark schemes, and student answers."""

    def __init__(
        self,
        config_or_pattern: Optional[Union[SubjectConfig, str]] = None,
        config: Optional[SubjectConfig] = None,
        mark_pattern: Optional[str] = None,
        sub_question_pattern: Optional[str] = None,
        **kwargs: Any,
    ):
        cfg = config or (config_or_pattern if isinstance(config_or_pattern, SubjectConfig) else None)
        if cfg:
            self.config = cfg
            self.mark_pattern = mark_pattern or getattr(cfg, "mark_pattern", r"\((\d+)\)")
            self.sub_q_pat = sub_question_pattern or getattr(cfg, "sub_question_pattern", r"\(\s*[a-h]\s*\)")
        else:
            self.config = None
            self.mark_pattern = mark_pattern or (config_or_pattern if isinstance(config_or_pattern, str) else r"\((\d+)\)")
            self.sub_q_pat = sub_question_pattern or r"\(\s*[a-h]\s*\)"

    def parse_exam_paper(self, paper_text: str) -> List[Dict[str, Any]]:
        """Parse exam paper text into distinct question/subquestion items."""
        if not paper_text:
            return []

        # Check for main questions
        q_splits = re.split(r"(?im)(?=^\s*Question\s*\d+)", paper_text)
        items = []

        for q_block in q_splits:
            q_block = q_block.strip()
            if not q_block:
                continue

            first_line = q_block.splitlines()[0]
            q_match = re.search(r"(?i)Question\s*(\d+)", first_line)
            q_num = q_match.group(1) if q_match else "1"

            # Check for subquestions
            if self.sub_q_pat:
                sub_parts = re.split(f"(?m)(?={self.sub_q_pat})", q_block)
                if len(sub_parts) > 1:
                    intro = sub_parts[0].strip()
                    for sp in sub_parts[1:]:
                        sp_clean = sp.strip()
                        if not sp_clean:
                            continue
                        sub_match = re.search(r"\(\s*([a-h])\s*\)", sp_clean)
                        sub_lbl = sub_match.group(1) if sub_match else "a"
                        marks = extract_mark_val(sp_clean, self.mark_pattern)
                        label = f"Q{q_num}({sub_lbl})"
                        items.append({
                            "label": label,
                            "question": sp_clean,
                            "context": intro,
                            "marks": marks
                        })
                    continue

            # Standalone question
            marks = extract_mark_val(q_block, self.mark_pattern)
            items.append({
                "label": f"Q{q_num}",
                "question": q_block,
                "marks": marks
            })

        logger.info("Parsed %d individual question items from question paper", len(items))
        return items

    def parse_mark_scheme(self, ms_text: str) -> Dict[str, str]:
        """Parse mark scheme text into a dictionary mapping question codes/numbers to mark scheme text."""
        if not ms_text:
            return {}

        ms_map = {}
        entries = re.split(r"(?im)(?=^\s*(?:Question\s*Number|Question\s*\d+|\d+\s*\([a-h]\)))", ms_text)

        for entry in entries:
            entry_clean = entry.strip()
            if not entry_clean:
                continue

            code_match = re.search(r"(?i)(?:Question\s*Number|Question)?\s*(\d+\s*(?:\([a-h]\)|[a-h])?)", entry_clean)
            if code_match:
                code = code_match.group(1).replace(" ", "")
                ms_map[code] = entry_clean
            else:
                ms_map[f"item_{len(ms_map)+1}"] = entry_clean

        return ms_map

    def segment_student_answers(
        self, answers_text: str, questions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """Map student answers to question labels."""
        if not answers_text:
            return {}

        answers_map: Dict[str, str] = {}
        splits = re.split(r"(?im)(?=^\s*(?:Question\s*\d+|\d+\s*[\(\:\.]|[a-h]\)))", answers_text)

        for split in splits:
            split_clean = split.strip()
            if not split_clean:
                continue

            m = re.search(r"(?i)(?:Question\s*)?(\d+)?\s*\(?([a-h])?\)?\s*[:\.]?", split_clean)
            if m and (m.group(1) or m.group(2)):
                q_n = m.group(1) or ""
                sub_l = m.group(2) or ""
                lbl = f"Q{q_n}({sub_l})" if q_n and sub_l else (f"Q{q_n}" if q_n else f"({sub_l})")
                answers_map[lbl] = split_clean
                if q_n and sub_l:
                    answers_map[f"{q_n}({sub_l})"] = split_clean
                    answers_map[f"Q{q_n}{sub_l}"] = split_clean

        # Fallback if questions provided
        if questions:
            for q in questions:
                lbl = q.get("label", "")
                if lbl not in answers_map:
                    clean_key = lbl.replace("Q", "").replace(" ", "")
                    if clean_key in answers_map:
                        answers_map[lbl] = answers_map[clean_key]

        return answers_map

    def segment_full_paper(
        self,
        question_paper_text: str,
        student_answers_text: str,
        mark_scheme_text: str = "",
    ) -> List[Dict[str, Any]]:
        """Combine question paper, student answers, and mark schemes into structured list."""
        questions = self.parse_exam_paper(question_paper_text)
        answers_map = self.segment_student_answers(student_answers_text, questions)
        ms_map = self.parse_mark_scheme(mark_scheme_text)

        items = []
        for q in questions:
            lbl = q.get("label", "")
            clean_lbl = lbl.replace("Q", "").replace(" ", "")

            student_ans = answers_map.get(lbl, answers_map.get(clean_lbl, ""))
            ms_entry = ms_map.get(clean_lbl, ms_map.get(lbl, ""))

            items.append({
                "label": lbl,
                "question": q.get("question", ""),
                "marks": q.get("marks"),
                "student_answer": student_ans,
                "answer": student_ans,
                "mark_scheme": ms_entry,
            })

        logger.info("Successfully segmented paper into %d items", len(items))
        return items
