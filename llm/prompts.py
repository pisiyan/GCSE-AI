"""Prompt manager for loading, caching, and formatting prompt templates."""

import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# In-memory cache for prompt templates
_PROMPT_CACHE: Dict[str, str] = {}


def get_prompts_dir() -> str:
    """Get the absolute path to the prompts directory."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_dir = os.path.join(root_dir, "prompts")
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir, exist_ok=True)
    return prompts_dir


def load_prompt_template(name: str) -> str:
    """Load a raw prompt template from the prompts folder.

    Args:
        name: Name of the prompt file (with or without .txt extension).

    Returns:
        The raw template string.
    """
    clean_name = name if name.endswith(".txt") else f"{name}.txt"
    if clean_name in _PROMPT_CACHE:
        return _PROMPT_CACHE[clean_name]

    prompts_dir = get_prompts_dir()
    file_path = os.path.join(prompts_dir, clean_name)

    if not os.path.exists(file_path):
        # Check without .txt extension if file exists as-is
        alt_path = os.path.join(prompts_dir, name)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Prompt template '{clean_name}' not found in {prompts_dir}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read()
        _PROMPT_CACHE[clean_name] = template
        return template
    except Exception as e:
        logger.error("Failed to read prompt file %s: %s", file_path, e)
        raise


def load_prompt(name: str, **kwargs: Any) -> str:
    """Load and format a prompt template with the provided keyword arguments.

    Args:
        name: Name of the prompt template (e.g. 'mark_answer').
        **kwargs: Variables to interpolate into the template.

    Returns:
        Formatted prompt string.
    """
    template = load_prompt_template(name)
    if not kwargs:
        return template
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.warning("Missing key %s when formatting prompt %s. Attempting partial replacement.", e, name)
        # Safe fallback for partial formatting
        formatted = template
        for k, v in kwargs.items():
            formatted = formatted.replace(f"{{{k}}}", str(v))
        return formatted


def load_all_prompts() -> Dict[str, str]:
    """Load all prompt templates from the prompts directory into a dict."""
    prompts_dir = get_prompts_dir()
    templates: Dict[str, str] = {}
    if not os.path.isdir(prompts_dir):
        return templates

    for filename in os.listdir(prompts_dir):
        full_path = os.path.join(prompts_dir, filename)
        if os.path.isfile(full_path):
            key = filename[:-4] if filename.endswith(".txt") else filename
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    templates[key] = f.read()
            except Exception as e:
                logger.error("Error reading prompt template %s: %s", full_path, e)
    return templates
