"""Specification Content Summary Generator for GCSE AI.

Extracts structured hierarchy (Exam Types -> Topics -> Subtopics)
from subject specifications using a single LLM call.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"), override=True)

from config import load_subject_config
from llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def extract_spec_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a specification PDF file."""
    if not os.path.exists(pdf_path):
        return ""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # Skip cover page if multiple pages
        start_idx = 1 if len(pages) > 1 else 0
        return "\n\n".join(pages[i].page_content for i in range(start_idx, len(pages)))
    except Exception as e:
        logger.error("Failed to extract text from PDF %s: %s", pdf_path, e)
        return ""


def get_specification_text(subject: str, examiner: str) -> str:
    """Load cached specification text or extract from PDF."""
    root_dir = f"data/{subject}/{examiner}/Specification"
    cache_path = os.path.join(root_dir, f"{subject}-{examiner}-Specification.txt")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning("Could not read spec text cache %s: %s", cache_path, e)

    # Search for PDF file in Specification directory
    if os.path.exists(root_dir):
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(".pdf"):
                full_pdf = os.path.join(root_dir, fname)
                text = extract_spec_text_from_pdf(full_pdf)
                if text:
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            f.write(text)
                    except Exception:
                        pass
                    return text
    return ""


def discover_exam_types(subject: str, examiner: str) -> List[str]:
    """Discover available exam types from Exam-Types subdirectories or fallback."""
    exam_types_dir = f"data/{subject}/{examiner}/Exam-Types"
    if os.path.exists(exam_types_dir):
        subdirs = [
            d for d in os.listdir(exam_types_dir)
            if os.path.isdir(os.path.join(exam_types_dir, d))
        ]
        if subdirs:
            return sorted(subdirs)
    return ["General"]


def extract_spec_summary_single_call(
    llm: LLMClient, spec_text: str, exam_types: List[str], subject: str, examiner: str
) -> Dict[str, Any]:
    """Extract full specification summary (topics and subtopics for each exam type) in a single LLM call."""
    logger.info("Extracting specification summary with single LLM call for %s (%s), exam types: %s", subject, examiner, exam_types)

    # Truncate spec_text preview if excessively long to fit context window
    context = spec_text[:60000] if len(spec_text) > 60000 else spec_text

    formatted_exam_types = json.dumps(exam_types)

    prompt = f"""You are an expert GCSE specification analyzer.
Read the specification text below for {subject} ({examiner}).

The specification covers the following exam types / components / options: {formatted_exam_types}.

Your task is to analyze the specification text and return a complete summary hierarchy for EVERY exam type in {formatted_exam_types} in ONE response.

For each exam type:
1. Identify all main official Topic titles/sections that belong to that exam type (e.g., 'Topic 1 - Key concepts in biology', 'Section 3.1 Christianity Beliefs').
2. Provide the specification code/number range if found (e.g., 'Topic 1 (1.1 - 1.17)').
3. List all the subtopics (their names) belonging to each topic directly from the specification text. Ensure subtopics are clear, distinct names appropriate for exam question creation.

Specification Text:
{context}

Return strictly a JSON object with this structure:
{{
  "exam_types": [
    {{
      "exam_type": "Exam Type Name",
      "topics": [
        {{
          "topic": "Topic Name",
          "spec_code": "Specification code range or Topic Name",
          "subtopics": [
            "Subtopic Name 1",
            "Subtopic Name 2"
          ]
        }}
      ]
    }}
  ]
}}

Requirements:
- Include entries for all requested exam types: {formatted_exam_types}.
- Under each exam type, include all its topics and below them the subtopic names from the specification.
- Return ONLY raw JSON, with no explanation or markdown formatting."""

    try:
        data = llm.invoke_json(prompt)
        raw_exam_types = data.get("exam_types", [])
        if isinstance(raw_exam_types, list) and raw_exam_types:
            cleaned_et_list = []
            for et_item in raw_exam_types:
                if not isinstance(et_item, dict):
                    continue
                et_name = str(et_item.get("exam_type", "General")).strip()
                raw_topics = et_item.get("topics", [])
                cleaned_topics = []
                if isinstance(raw_topics, list):
                    for top_item in raw_topics:
                        if isinstance(top_item, str):
                            cleaned_topics.append({
                                "topic": top_item.strip(),
                                "spec_code": top_item.strip(),
                                "subtopics": [top_item.strip()]
                            })
                        elif isinstance(top_item, dict):
                            t_name = str(top_item.get("topic", "")).strip() or "Topic"
                            s_code = str(top_item.get("spec_code", t_name)).strip() or t_name
                            subs = top_item.get("subtopics", [])
                            sub_list = []
                            if isinstance(subs, list):
                                for s in subs:
                                    if isinstance(s, str) and s.strip():
                                        sub_list.append(s.strip())
                                    elif isinstance(s, dict) and s.get("name"):
                                        sub_list.append(str(s.get("name")).strip())
                            if not sub_list:
                                sub_list = [t_name]
                            cleaned_topics.append({
                                "topic": t_name,
                                "spec_code": s_code,
                                "subtopics": sub_list
                            })
                if not cleaned_topics:
                    cleaned_topics = [{
                        "topic": f"{subject} {et_name} Main Topic",
                        "spec_code": f"{subject} {et_name} Main Topic",
                        "subtopics": [f"{subject} {et_name} Main Topic"]
                    }]
                cleaned_et_list.append({
                    "exam_type": et_name,
                    "topics": cleaned_topics
                })

            if cleaned_et_list:
                return {"exam_types": cleaned_et_list}
    except Exception as e:
        logger.error("Single LLM call failed for specification summary: %s", e)

    # Fallback structure if LLM call fails or returns unexpected format
    fallback_et_list = []
    for et in exam_types:
        fallback_et_list.append({
            "exam_type": et,
            "topics": [{
                "topic": f"{subject} {et} Main Topic",
                "spec_code": f"{subject} {et} Main Topic",
                "subtopics": [f"{subject} {et} Main Topic"]
            }]
        })
    return {"exam_types": fallback_et_list}


def generate_specification_summary(subject: str, examiner: str, skip_if_exists: bool = False) -> Dict[str, Any]:
    """Generate structured Specification Content Summary using a single LLM call."""
    out_dir = f"data/{subject}/{examiner}"
    json_path = os.path.join(out_dir, f"{subject}-{examiner}-specification_summary.json")

    if skip_if_exists and os.path.exists(json_path):
        logger.info("Specification summary JSON already exists at: %s. Skipping generation.", json_path)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read existing specification summary JSON %s: %s. Re-generating...", json_path, e)

    logger.info("Generating Specification Content Summary for %s (%s)...", subject, examiner)

    try:
        config = load_subject_config(subject, examiner)
    except Exception as e:
        logger.warning("Could not load config for %s-%s: %s", subject, examiner, e)
        config = None

    model_name = getattr(config, "LLM_MODEL", "gpt-5.4-mini") if config else "gpt-5.4-mini"
    llm = LLMClient(model=model_name)

    spec_text = get_specification_text(subject, examiner)
    if not spec_text:
        logger.warning("No specification text found for %s (%s). Summary will be minimal.", subject, examiner)
        spec_text = f"{subject} ({examiner}) Specification content."

    exam_types = discover_exam_types(subject, examiner)

    raw_summary = extract_spec_summary_single_call(
        llm, spec_text, exam_types, subject, examiner
    )

    summary_data = {
        "subject": subject,
        "examiner": examiner,
        "exam_types": raw_summary.get("exam_types", [])
    }

    # Save JSON file
    out_dir = f"data/{subject}/{examiner}"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{subject}-{examiner}-specification_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved specification summary JSON to: %s", json_path)

    # Save Markdown file
    md_path = os.path.join(out_dir, f"{subject}-{examiner}-specification_summary.md")
    md_lines = [
        f"# Specification Content Summary: {subject} ({examiner})\n",
    ]
    for et_item in summary_data.get("exam_types", []):
        md_lines.append(f"## Exam Type: {et_item.get('exam_type', 'General')}\n")
        for top_item in et_item.get("topics", []):
            top_name = top_item.get("topic", "Topic")
            spec_code = top_item.get("spec_code", top_name)
            md_lines.append(f"### Topic: {top_name} (Spec Code: {spec_code})\n")
            for sub_name in top_item.get("subtopics", []):
                md_lines.append(f"- {sub_name}")
            md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info("Saved specification summary Markdown to: %s", md_path)

    return summary_data


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "Physics"
    ex = sys.argv[2] if len(sys.argv) > 2 else "Edexcel"
    generate_specification_summary(sub, ex)
