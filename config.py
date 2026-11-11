"""Central configuration module for GCSE AI.

Defines SubjectConfig dataclass with essential parsing regexes,
generation constraints, and subject registry settings.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SubjectConfig:
    """Configuration settings for a subject-examiner combination."""
    # Parsing pattern defaults
    mark_pattern: str = r"\((\d+)\)"
    sub_question_pattern: Optional[str] = None
    sub_sub_question_pattern: Optional[str] = None
    question_pattern: str = r"(?i)question\s*\d+"
    question_split_pattern: str = r"(?i)question\s*\d+"
    question_start_pattern: Optional[str] = r"(?i)^(?:Question\s*(\d+)|(\d+)\s*(?:\([a-z]\)|\.|\b[A-Z]))"
    question_end_pattern: Optional[str] = None
    ms_pattern: str = r"(?i)question\s*\d+"
    ms_mark_pattern: str = r"\((\d+)\)"

    # Exemplar selection counts
    example_questions: int = 5
    example_ms: int = 5

    # Execution limits & algorithm parameters
    max_structure_retries: int = 50
    max_parallel_workers: int = 10
    max_parent_exemplars: int = 3
    top_k_similar_choices: int = 3
    similarity_threshold: float = 0.70
    min_subtopic_similarity: float = 0.50
    min_soft_subtopic_similarity: float = 0.35
    subtopic_diversity_weight: float = 500.0
    fallback_embedding_dim: int = 384
    temperature_structure: float = 0.0
    temperature_generation: float = 0.7

    # Dynamic fields set upon loading
    subject: Optional[str] = None
    examiner: Optional[str] = None


# Self-contained configs for each subject-examiner combination
SUBJECT_CONFIGS = {
    "ReligiousStudies-AQA": {
        "mark_pattern": r"[\[\(]\s*(\d{1,2})\s*(marks?)?\s*[\]\)]?",
        "sub_question_pattern": None,
        "sub_sub_question_pattern": None,
        "question_pattern": r"(?=(?:\s*\d){2}\s*\.\s*\d\b)",
        "question_split_pattern": r"(?=(?:\s*\d){2}\s*\.\s*\d\b)",
        "question_start_pattern": r"(?i)(?:(?:\s*\d){2}\s*\.\s*\d\b|Question\s*\d+)",
        "question_end_pattern": None,
        "ms_pattern": r"(?=(?:\s*\d){2}\s*\.\s*\d\b)",
        "ms_mark_pattern": r"[\[\(]\s*(\d{1,2})\s*(marks?)?\s*[\]\)]?",
        "example_questions": 8,
        "example_ms": 5,
        "top_k_similar_choices": 3,
        "similarity_threshold": 0.70,
        "min_subtopic_similarity": 0.50,
    },
    "Biology-Edexcel": {
        "mark_pattern": r"\((\d+)\)",
        "sub_question_pattern": r"\(\s*[a-h]\s*\)",
        "sub_sub_question_pattern": r"\((?:i{1,3}|iv|v|vi{1,3}|ix|x)\)",
        "question_pattern": r"\(T[\s\n]*o[\s\n]*t[\s\n]*a[\s\n]*l[\s\n]* [\s\n]*f[\s\n]*o[\s\n]*r[\s\n]* [\s\n]*Q[\s\n]*u[\s\n]*e[\s\n]*s[\s\n]*t[\s\n]*i[\s\n]*o[\s\n]*n[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*=[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*m[\s\n]*a[\s\n]*r[\s\n]*k[\s\n]*s?\)",
        "question_split_pattern": r"\(T[\s\n]*o[\s\n]*t[\s\n]*a[\s\n]*l[\s\n]* [\s\n]*f[\s\n]*o[\s\n]*r[\s\n]* [\s\n]*Q[\s\n]*u[\s\n]*e[\s\n]*s[\s\n]*t[\s\n]*i[\s\n]*o[\s\n]*n[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*=[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*m[\s\n]*a[\s\n]*r[\s\n]*k[\s\n]*s?\)",
        "question_start_pattern": r"(?i)^(?:Question\s*(\d+)|(\d+)\s*(?:\([a-z]\)|\.|\b[A-Z]))",
        "question_end_pattern": r"(?i)\(?\s*Total\s+for\s+Question\s*(\d+)",
        "ms_pattern": r"(?i)Q\s*u\s*e\s*s\s*t\s*i\s*o\s*n\s*\s*N\s*u\s*m\s*b\s*e\s*r",
        "ms_mark_pattern": r"\((\d+)\)",
        "example_questions": 5,
        "example_ms": 5,
        "top_k_similar_choices": 3,
        "similarity_threshold": 0.70,
        "min_subtopic_similarity": 0.50,
    },
    "Physics-Edexcel": {
        "mark_pattern": r"\((\d+)\)",
        "sub_question_pattern": r"\(\s*[a-h]\s*\)",
        "sub_sub_question_pattern": r"\((?:i{1,3}|iv|v|vi{1,3}|ix|x)\)",
        "question_pattern": r"\(T[\s\n]*o[\s\n]*t[\s\n]*a[\s\n]*l[\s\n]* [\s\n]*f[\s\n]*o[\s\n]*r[\s\n]* [\s\n]*Q[\s\n]*u[\s\n]*e[\s\n]*s[\s\n]*t[\s\n]*i[\s\n]*o[\s\n]*n[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*=[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*m[\s\n]*a[\s\n]*r[\s\n]*k[\s\n]*s?\)",
        "question_split_pattern": r"\(T[\s\n]*o[\s\n]*t[\s\n]*a[\s\n]*l[\s\n]* [\s\n]*f[\s\n]*o[\s\n]*r[\s\n]* [\s\n]*Q[\s\n]*u[\s\n]*e[\s\n]*s[\s\n]*t[\s\n]*i[\s\n]*o[\s\n]*n[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*=[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*m[\s\n]*a[\s\n]*r[\s\n]*k[\s\n]*s?\)",
        "question_start_pattern": r"(?i)^(?:Question\s*(\d+)|(\d+)\s*(?:\([a-z]\)|\.|\b[A-Z]))",
        "question_end_pattern": r"(?i)\(?\s*Total\s+for\s+Question\s*(\d+)",
        "ms_pattern": r"(?i)Q\s*u\s*e\s*s\s*t\s*i\s*o\s*n\s*\s*N\s*u\s*m\s*b\s*e\s*r",
        "ms_mark_pattern": r"\((\d+)\)",
        "example_questions": 5,
        "example_ms": 5,
        "top_k_similar_choices": 3,
        "similarity_threshold": 0.70,
        "min_subtopic_similarity": 0.50,
    },
}


def load_subject_config(subject: str, examiner: str) -> SubjectConfig:
    """Load subject configuration directly from SUBJECT_CONFIGS dictionary.

    If a specific subject-examiner combination is not pre-registered,
    returns a default SubjectConfig populated with subject and examiner names.
    """
    key = f"{subject}-{examiner}"

    if key in SUBJECT_CONFIGS:
        config_dict = dict(SUBJECT_CONFIGS[key])
    else:
        logger.info("No custom config for %s; using standard SubjectConfig defaults.", key)
        config_dict = {}

    config_dict["subject"] = subject
    config_dict["examiner"] = examiner

    return SubjectConfig(**config_dict)
