"""GCSE command word explainer and syllabus breakdown module."""

import json
import logging
import os
from typing import Any, Dict, Optional

from llm.client import LLMClient
from llm.prompts import load_prompt

logger = logging.getLogger(__name__)


def explain_command_word(
    command_word: str,
    subject: str,
    examiner: str,
    llm_client: LLMClient,
    question: str = "",
) -> str:
    """Explain GCSE command word requirements, mark criteria, and structure."""
    cw = command_word.strip()
    if not cw and question:
        try:
            cw_prompt = load_prompt("get_command_word", question=question)
            cw = llm_client.invoke(cw_prompt).strip()
        except Exception as e:
            logger.warning("Failed to extract command word: %s", e)

    if not cw:
        cw = "Explain"

    prompt = load_prompt(
        "command_word_guidance",
        command_word=cw,
        subject=subject,
        examiner=examiner,
    )
    logger.info("Explaining command word '%s' for %s (%s)...", cw, subject, examiner)
    return llm_client.invoke(prompt)


def get_spec_breakdown(
    subject: str,
    examiner: str,
    topic: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve specification codes, topics, and subtopics breakdown."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sum_path = f"data/{subject}/{examiner}/{subject}-{examiner}-specification_summary.json"
    if not os.path.exists(sum_path):
        sum_path = os.path.join(root_dir, "data", subject, examiner, f"{subject}-{examiner}-specification_summary.json")

    summary_data = {}
    if os.path.exists(sum_path):
        try:
            with open(sum_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception as e:
            logger.warning("Could not read spec summary: %s", e)

    if topic and topic.strip():
        # Look for specific topic in summary
        topic_clean = topic.strip().lower()
        matched_spec_code = topic
        matched_subtopics = [topic]

        for et_item in summary_data.get("exam_types", []):
            for t_item in et_item.get("topics", []):
                t_name = str(t_item.get("topic", "")).strip()
                if topic_clean in t_name.lower() or t_name.lower() in topic_clean:
                    matched_spec_code = t_item.get("spec_code", topic)
                    raw_subs = t_item.get("subtopics", [topic])
                    matched_subtopics = [
                        s if isinstance(s, str) else s.get("name", str(s)) for s in raw_subs
                    ]
                    break

        return {
            "subject": subject,
            "examiner": examiner,
            "topic": topic,
            "spec_code": matched_spec_code,
            "subtopics": matched_subtopics
        }

    return {
        "subject": subject,
        "examiner": examiner,
        "summary": summary_data
    }
