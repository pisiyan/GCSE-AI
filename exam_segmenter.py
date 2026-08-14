"""Exam paper and mark scheme segmentation module for GCSE AI.

Parses full exam papers (PDFs/text), mark schemes, and student answer documents
into individual question items for separate, parallel evaluation.
"""

import json
import logging
import re
from typing import Any, Optional, Dict, List

from config import SubjectConfig

logger = logging.getLogger(__name__)


def extract_mark_val(text: str, pattern: str) -> int:
    """Extract mark value from text using pattern or fallbacks."""
    if not text:
        return 1

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
    return 1


class ExamSegmenter:
    """Parses full question papers, mark schemes, and student answers into individual items."""

    def __init__(self, config: SubjectConfig) -> None:
        self.config = config

    def parse_exam_paper(self, paper_text: str) -> List[Dict[str, Any]]:
        """Parse full question paper text into individual question items.

        Args:
            paper_text: The complete text of the question paper.

        Returns:
            List of question dicts, each with 'label', 'parent_description',
            'question', and 'marks'.
        """
        if not paper_text or not paper_text.strip():
            return []

        q_pattern = getattr(self.config, "question_pattern", r"(?i)question\s*\d+")
        sub_q_pat = getattr(self.config, "sub_question_pattern", r"\(\s*[a-h]\s*\)")
        sub_sub_q_pat = getattr(self.config, "sub_sub_question_pattern", r"\((?:i{1,3}|iv|v|vi{1,3}|ix|x)\)")
        mark_pat = getattr(self.config, "mark_pattern", r"\((\d+)\)")

        # Split into main question blocks
        raw_blocks = re.split(q_pattern, paper_text)
        
        # Fallback split if primary pattern returns only 1 block for long text
        if len(raw_blocks) <= 1 and len(paper_text) > 800:
            fallback_pattern = r"(?i)(?:\n|\r|^)\s*(?:Question|Q)\s*(\d+)"
            raw_blocks = re.split(fallback_pattern, paper_text)

        items: List[Dict[str, Any]] = []
        letters = "abcdefghijklmn"
        romans = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

        main_q_counter = 0

        for block in raw_blocks:
            cleaned_block = block.strip()
            if not cleaned_block or len(cleaned_block) < 15:
                continue

            main_q_counter += 1
            
            # Check if block has sub-questions (e.g. (a), (b))
            has_sub = False
            if sub_q_pat and sub_q_pat != "None":
                splits = re.split(sub_q_pat, cleaned_block)
                if len(splits) > 1:
                    has_sub = True
                    parent_desc = splits[0].strip()
                    sub_texts = splits[1:]

                    for idx, sq_text in enumerate(sub_texts):
                        label = f"Q{main_q_counter}({letters[idx]})" if idx < len(letters) else f"Q{main_q_counter}(sub{idx+1})"
                        
                        # Check for sub-sub-questions (e.g. (i), (ii))
                        has_sub_sub = False
                        if sub_sub_q_pat and sub_sub_q_pat != "None":
                            ss_splits = re.split(sub_sub_q_pat, sq_text)
                            if len(ss_splits) > 1:
                                has_sub_sub = True
                                sq_intro = ss_splits[0].strip()
                                ss_texts = ss_splits[1:]

                                for ss_idx, ss_text in enumerate(ss_texts):
                                    ss_label = f"Q{main_q_counter}({letters[idx]})({romans[ss_idx]})" if idx < len(letters) and ss_idx < len(romans) else f"{label}({ss_idx+1})"
                                    ss_marks = extract_mark_val(ss_text, mark_pat)
                                    full_q_text = (parent_desc + "\n" + sq_intro + "\n" + ss_text).strip()
                                    items.append({
                                        "label": ss_label,
                                        "parent_description": parent_desc,
                                        "question": (sq_intro + "\n" + ss_text).strip(),
                                        "full_question": full_q_text,
                                        "marks": ss_marks,
                                    })

                        if not has_sub_sub:
                            sq_marks = extract_mark_val(sq_text, mark_pat)
                            full_q_text = (parent_desc + "\n" + sq_text).strip() if parent_desc else sq_text.strip()
                            items.append({
                                "label": label,
                                "parent_description": parent_desc,
                                "question": sq_text.strip(),
                                "full_question": full_q_text,
                                "marks": sq_marks,
                            })

            if not has_sub:
                q_label = f"Q{main_q_counter}"
                q_marks = extract_mark_val(cleaned_block, mark_pat)
                items.append({
                    "label": q_label,
                    "parent_description": "",
                    "question": cleaned_block,
                    "full_question": cleaned_block,
                    "marks": q_marks,
                })

        logger.info("Parsed %d individual question items from question paper", len(items))
        return items

    def parse_mark_scheme(self, ms_text: str) -> Dict[str, str]:
        """Parse full mark scheme text into a dict mapping question labels to mark schemes.

        Args:
            ms_text: The complete text of the mark scheme.

        Returns:
            Dict mapping normalized question labels (e.g. '1(a)', 'Q1(a)', '1a') to mark scheme text.
        """
        if not ms_text or not ms_text.strip():
            return {}

        ms_map: Dict[str, str] = {}
        
        # Match headers like "Question Number 1(a)", "Question 1(a)", "1(a)", "Q1(a)", "1a"
        header_regex = re.compile(
            r"(?i)(?:\n|\r|^)\s*(?:Question\s*Number\s*|Question\s*|Q)?\s*(\d+[a-z]|\d+\s*\([a-z]\)(?:\([i|v|x]+\))?|\d+)",
            re.MULTILINE
        )

        matches = list(header_regex.finditer(ms_text))
        if not matches:
            return {"1": ms_text.strip()}

        for i, match in enumerate(matches):
            lbl_raw = match.group(1).replace(" ", "").lower()
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(ms_text)
            chunk = ms_text[start_pos:end_pos].strip()

            if chunk:
                ms_map[lbl_raw] = chunk
                ms_map[f"q{lbl_raw}"] = chunk
                clean_lbl = lbl_raw.replace("(", "").replace(")", "")
                ms_map[clean_lbl] = chunk
                ms_map[f"q{clean_lbl}"] = chunk

        logger.info("Parsed %d mark scheme entries", len(ms_map))
        return ms_map

    def segment_student_answers(
        self, answer_text: str, questions: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Segment student answer text into per-question answers.

        Args:
            answer_text: Complete text of student answers.
            questions: List of parsed question dicts with 'label' fields.

        Returns:
            Dict mapping question label to student answer string.
        """
        if not answer_text or not answer_text.strip():
            return {q.get("label", f"Q{i+1}"): "" for i, q in enumerate(questions)}

        answers_map: Dict[str, str] = {}

        # 1. Parse question headers like Question 1, Q1, 1(a), 1a, Q1a...
        header_regex = re.compile(
            r"(?i)(?:\bQuestion\s*|\bQ\s*)(\d+\s*(?:\([a-z]\))?(?:\([i|v|x]+\))?|\d+[a-z]|\d+)\s*[:\.\)\-]*",
            re.IGNORECASE
        )

        matches = list(header_regex.finditer(answer_text))

        if matches:
            for i, match in enumerate(matches):
                lbl_raw = match.group(1).replace(" ", "").lower()
                start_pos = match.end()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(answer_text)
                ans_chunk = answer_text[start_pos:end_pos].strip()

                answers_map[lbl_raw] = ans_chunk
                answers_map[f"q{lbl_raw}"] = ans_chunk
                clean_lbl = lbl_raw.replace("(", "").replace(")", "")
                answers_map[clean_lbl] = ans_chunk
                answers_map[f"q{clean_lbl}"] = ans_chunk

                # If label is like '1a' or 'q1a', also format as '1(a)' and 'q1(a)'
                m_sub = re.match(r"^(\d+)([a-z])$", clean_lbl)
                if m_sub:
                    p_num, s_let = m_sub.group(1), m_sub.group(2)
                    answers_map[f"{p_num}({s_let})"] = ans_chunk
                    answers_map[f"q{p_num}({s_let})"] = ans_chunk
                    if p_num not in answers_map:
                        answers_map[p_num] = ans_chunk
                        answers_map[f"q{p_num}"] = ans_chunk
                else:
                    m_num = re.match(r"^(\d+)", clean_lbl)
                    if m_num:
                        p_num = m_num.group(1)
                        if p_num not in answers_map:
                            answers_map[p_num] = ans_chunk
                            answers_map[f"q{p_num}"] = ans_chunk

        # Match parsed question labels to answers_map
        result: Dict[str, str] = {}
        unmatched_answers = []

        for q in questions:
            raw_label = q.get("label", "").strip()
            norm_label = raw_label.lower().replace(" ", "")

            matched_ans = ""
            clean_lbl = norm_label.replace("q", "").replace("(", "").replace(")", "")
            for key in (norm_label, f"q{clean_lbl}", clean_lbl):
                if key in answers_map and answers_map[key]:
                    matched_ans = answers_map[key]
                    break

            # Fallback to main parent question text if specific sub-question answer not isolated
            if not matched_ans:
                m_parent = re.match(r"(?i)^q?(\d+)", norm_label)
                if m_parent:
                    p_num = m_parent.group(1)
                    for p_key in (f"q{p_num}", p_num):
                        if p_key in answers_map and answers_map[p_key]:
                            matched_ans = answers_map[p_key]
                            break

            if matched_ans:
                result[raw_label] = matched_ans
            else:
                unmatched_answers.append(raw_label)

        # Fallback for questions without explicit headers in student answer
        if len(unmatched_answers) == len(questions) and len(questions) > 0:
            if len(questions) == 1:
                result[questions[0]["label"]] = answer_text.strip()
            else:
                lines = [line for line in answer_text.splitlines() if line.strip()]
                n_qs = len(questions)
                chunk_len = max(1, len(lines) // n_qs)
                for idx, q in enumerate(questions):
                    start = idx * chunk_len
                    end = start + chunk_len if idx < n_qs - 1 else len(lines)
                    result[q["label"]] = "\n".join(lines[start:end])

        logger.info("Segmented student answers for %d questions", len(result))
        return result

    def segment_full_paper(
        self,
        question_text: str,
        answer_text: str,
        ms_text: str = "",
    ) -> List[Dict[str, Any]]:
        """Segment a full exam paper, mark scheme, and student answer into matching question items.

        Args:
            question_text: Complete text of the question paper.
            answer_text: Complete text of student answers.
            ms_text: Optional complete text of mark scheme.

        Returns:
            List of question dicts ready for parallel marking via ExamMarker.mark_exam.
        """
        parsed_questions = self.parse_exam_paper(question_text)
        ms_map = self.parse_mark_scheme(ms_text) if ms_text else {}
        answer_map = self.segment_student_answers(answer_text, parsed_questions)

        segmented_items: List[Dict[str, Any]] = []

        for q in parsed_questions:
            label = q.get("label", "")
            clean_lbl = label.lower().replace("q", "").replace("(", "").replace(")", "")

            # Match per-question mark scheme if available
            matched_ms = ""
            for key in (label.lower(), f"q{clean_lbl}", clean_lbl):
                if key in ms_map:
                    matched_ms = ms_map[key]
                    break

            student_ans = answer_map.get(label, "")

            segmented_items.append({
                "label": label,
                "parent_description": q.get("parent_description", ""),
                "question": q.get("question", ""),
                "full_question": q.get("full_question", q.get("question", "")),
                "marks": q.get("marks", 1),
                "answer": student_ans,
                "student_answer": student_ans,
                "mark_scheme": matched_ms,
            })

        logger.info("Successfully segmented full paper into %d ready-to-mark question items", len(segmented_items))
        return segmented_items
