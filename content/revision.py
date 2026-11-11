"""Revision materials generator module for GCSE AI."""

import logging
from typing import Any, Optional
from llm.client import LLMClient
from llm.prompts import load_prompt

logger = logging.getLogger(__name__)


def generate_revision_materials(
    topic: str,
    subject: str,
    examiner: str,
    llm_client: LLMClient,
    specification_text: str = "",
    spec_qa_chain: Optional[Any] = None,
) -> str:
    """Generate comprehensive structured revision materials for a GCSE topic."""
    logger.info("Generating revision materials for %s: '%s'...", subject, topic)

    spec_snippet = specification_text[:10000] if specification_text else "GCSE syllabus content."

    prompt = (
        f"You are an expert GCSE {subject} ({examiner}) tutor.\n"
        f"Create comprehensive, structured revision notes and summary materials for the topic: '{topic}'.\n\n"
        f"Specification Context:\n{spec_snippet}\n\n"
        f"Include:\n"
        f"1. **Core Concept Overview**: High-yield summary of key definitions, equations, and principles.\n"
        f"2. **Key Scientific Terminology & Definitions**: Precise examiner keywords.\n"
        f"3. **Step-by-Step Mechanisms & Processes**: Clear numbered explanations.\n"
        f"4. **Required Practicals & Experimental Methods** (if applicable).\n"
        f"5. **Common Examiner Traps & Misconceptions**: Pitfalls to avoid in exams.\n"
        f"6. **Quick Self-Check Quiz Questions** (3-4 bullet questions with answers).\n\n"
        f"Format in clean, engaging GitHub-style markdown."
    )

    try:
        return llm_client.invoke(prompt)
    except Exception as e:
        logger.error("Failed to generate revision materials: %s", e)
        return f"# Revision Notes: {topic}\n\nReview the core textbook and specification points for {topic}."
