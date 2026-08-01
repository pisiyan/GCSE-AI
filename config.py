import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SubjectConfig:
    """Configuration for a subject-examiner combination."""
    # Pattern defaults
    mark_pattern: str = r"\((\d+)\)"
    sub_question_pattern: Optional[str] = None
    sub_sub_question_pattern: Optional[str] = None
    question_pattern: str = r"(?i)question\s*\d+"
    ms_pattern: str = r"(?i)question\s*\d+"
    ms_mark_pattern: str = r"\((\d+)\)"
    
    # Chunking and retrieval defaults
    spec_chunk_size: int = 2000
    spec_chunk_overlap: int = 1000
    ms_chunk_size: int = 1700
    ms_chunk_overlap: int = 300
    spec_search_kwargs_k: int = 5
    ms_search_kwargs_k: int = 5
    
    # Question examples count defaults
    example_questions: int = 5
    example_ms: int = 5
    example_descriptions: int = 0
    question_no_importance: bool = True
    
    # Configurable limits and execution parameters
    max_structure_retries: int = 50
    max_parallel_workers: int = 10
    max_parent_exemplars: int = 3
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
        "ms_pattern": r"(?=(?:\s*\d){2}\s*\.\s*\d\b)",
        "ms_mark_pattern": r"[\[\(]\s*(\d{1,2})\s*(marks?)?\s*[\]\)]?",
        "spec_chunk_size": 3000,
        "spec_chunk_overlap": 2000,
        "ms_chunk_size": 4000,
        "ms_chunk_overlap": 3000,
        "spec_search_kwargs_k": 5,
        "ms_search_kwargs_k": 3,
        "example_questions": 8,
        "example_ms": 5,
        "example_descriptions": 0,
        "question_no_importance": True,
    },
    "Biology-Edexcel": {
        "mark_pattern": r"\((\d+)\)",
        "sub_question_pattern": r"\(\s*[a-h]\s*\)",
        "sub_sub_question_pattern": r"\((?:i{1,3}|iv|v|vi{1,3}|ix|x)\)",
        "question_pattern": r"\(T[\s\n]*o[\s\n]*t[\s\n]*a[\s\n]*l[\s\n]* [\s\n]*f[\s\n]*o[\s\n]*r[\s\n]* [\s\n]*Q[\s\n]*u[\s\n]*e[\s\n]*s[\s\n]*t[\s\n]*i[\s\n]*o[\s\n]*n[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*=[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*m[\s\n]*a[\s\n]*r[\s\n]*k[\s\n]*s?\)",
        "ms_pattern": r"(?i)Q\s*u\s*e\s*s\s*t\s*i\s*o\s*n\s*\s*N\s*u\s*m\s*b\s*e\s*r",
        "ms_mark_pattern": r"\((\d+)\)",
        "spec_chunk_size": 2000,
        "spec_chunk_overlap": 1000,
        "ms_chunk_size": 1700,
        "ms_chunk_overlap": 300,
        "spec_search_kwargs_k": 5,
        "ms_search_kwargs_k": 10,
        "example_questions": 5,
        "example_ms": 5,
        "example_descriptions": 5,
        "question_no_importance": True,
    },
    "Physics-Edexcel": {
        "mark_pattern": r"\((\d+)\)",
        "sub_question_pattern": r"\(\s*[a-h]\s*\)",
        "sub_sub_question_pattern": r"\((?:i{1,3}|iv|v|vi{1,3}|ix|x)\)",
        "question_pattern": r"\(T[\s\n]*o[\s\n]*t[\s\n]*a[\s\n]*l[\s\n]* [\s\n]*f[\s\n]*o[\s\n]*r[\s\n]* [\s\n]*Q[\s\n]*u[\s\n]*e[\s\n]*s[\s\n]*t[\s\n]*i[\s\n]*o[\s\n]*n[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*=[\s\n]* [\s\n]*(\d+[\s\n]*)[\s\n]* [\s\n]*m[\s\n]*a[\s\n]*r[\s\n]*k[\s\n]*s?\)",
        "ms_pattern": r"(?i)Q\s*u\s*e\s*s\s*t\s*i\s*o\s*n\s*\s*N\s*u\s*m\s*b\s*e\s*r",
        "ms_mark_pattern": r"\((\d+)\)",
        "spec_chunk_size": 2000,
        "spec_chunk_overlap": 1000,
        "ms_chunk_size": 1700,
        "ms_chunk_overlap": 300,
        "spec_search_kwargs_k": 5,
        "ms_search_kwargs_k": 10,
        "example_questions": 5,
        "example_ms": 5,
        "example_descriptions": 5,
        "question_no_importance": True,
    },
}


def load_subject_config(subject: str, examiner: str) -> SubjectConfig:
    """Load subject configuration directly from SUBJECT_CONFIGS dictionary."""
    key = f"{subject}-{examiner}"
    
    if key not in SUBJECT_CONFIGS:
        raise ValueError(f"No configuration found for {key}. Available: {list(SUBJECT_CONFIGS.keys())}")
    
    config_dict = dict(SUBJECT_CONFIGS[key])
    config_dict["subject"] = subject
    config_dict["examiner"] = examiner
    
    return SubjectConfig(**config_dict)
