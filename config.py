import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SubjectConfig:
    """Configuration for a subject-examiner combination."""
    # Pattern defaults (sensible general defaults)
    mark_pattern: str = r"\((\d+)\)"
    letter_pattern: Optional[str] = None
    roman_pattern: Optional[str] = None
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
    question_no_importance: bool = False
    
    # Dynamic fields set upon loading
    subject: Optional[str] = None
    examiner: Optional[str] = None


# Board-specific default configurations
BOARD_DEFAULTS = {
    "AQA": {
        "mark_pattern": r"[\[\(]\s*(\d{1,2})\s*(marks?)?\s*[\]\)]?",
        "letter_pattern": None,
        "roman_pattern": None,
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
    "Edexcel": {
        "mark_pattern": r"\((\d+)\)",
        "letter_pattern": r"\(\s*[a-h]\s*\)",
        "roman_pattern": r"\((?:i{1,3}|iv|v|vi{1,3}|ix|x)\)",
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
        "question_no_importance": False,
    }
}


SUBJECT_CONFIGS = {
    "ReligiousStudies-AQA": {},
    "Biology-Edexcel": {},
    "Physics-Edexcel": {},
}


def load_subject_config(subject: str, examiner: str) -> SubjectConfig:
    """Load subject configuration from embedded dict, falling back to board defaults."""
    key = f"{subject}-{examiner}"
    
    if key not in SUBJECT_CONFIGS:
        raise ValueError(f"No configuration found for {key}. Available: {list(SUBJECT_CONFIGS.keys())}")
    
    config_dict = {}
    
    # 1. Apply examiner-specific defaults if they exist
    if examiner in BOARD_DEFAULTS:
        config_dict.update(BOARD_DEFAULTS[examiner])
        
    # 2. Override with subject-specific values if defined
    config_dict.update(SUBJECT_CONFIGS[key])
    
    # 3. Dynamically set subject and examiner details
    config_dict["subject"] = subject
    config_dict["examiner"] = examiner
    
    return SubjectConfig(**config_dict)
