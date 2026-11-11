"""Specification content summary extractor for GCSE AI."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from llm.client import LLMClient
from llm.prompts import load_prompt

logger = logging.getLogger(__name__)


def build_spec_summary(
    subject: str,
    examiner: str,
    spec_text: str,
    exam_types: List[str],
    llm_client: Optional[LLMClient] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract and build a structured specification summary JSON using a single LLM call."""
    llm = llm_client or LLMClient()
    logger.info("Extracting specification summary with LLM for %s (%s)...", subject, examiner)

    prompt = load_prompt(
        "build_spec_summary",
        subject=subject,
        examiner=examiner,
        spec_text=spec_text[:35000],
        exam_types_str=", ".join(exam_types) if exam_types else "Standard GCSE Papers",
    )

    try:
        summary_data = llm.invoke_json(prompt)
    except Exception as e:
        logger.error("Failed to build specification summary via LLM: %s", e)
        summary_data = {
            "subject": subject,
            "examiner": examiner,
            "exam_types": [{"exam_type": et, "topics": []} for et in (exam_types or ["General"])]
        }

    # Save to data directory if output_dir provided
    target_dir = output_dir or f"data/{subject}/{examiner}"
    os.makedirs(target_dir, exist_ok=True)

    json_path = os.path.join(target_dir, f"{subject}-{examiner}-specification_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save markdown summary
    md_lines = [
        f"# 📚 GCSE {subject} ({examiner}) — Specification Summary",
        "",
        "> Canonical syllabus topics and subtopics tree extracted from official specification.",
        "",
    ]
    for et in summary_data.get("exam_types", []):
        md_lines.append(f"## 📄 Exam Tier / Paper: `{et.get('exam_type', '')}`")
        for t in et.get("topics", []):
            code_str = f" `[{t.get('spec_code', '')}]`" if t.get("spec_code") else ""
            md_lines.append(f"### 🎯 {t.get('topic', '')}{code_str}")
            for st in t.get("subtopics", []):
                st_name = st if isinstance(st, str) else st.get("name", str(st))
                md_lines.append(f"- {st_name}")
            md_lines.append("")

    md_path = os.path.join(target_dir, f"{subject}-{examiner}-specification_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Saved specification summary JSON to: %s", json_path)
    logger.info("Saved specification summary Markdown to: %s", md_path)
    return summary_data
