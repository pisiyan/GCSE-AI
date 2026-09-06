"""Unified LLM client module for GCSE AI.

Supports dynamic provider switching (OpenAI, Anthropic, Google)
using LangChain's init_chat_model and robust retry handling.
"""

import os
import json
import logging
import re
import time
from typing import Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from llm.tracker import global_tracker

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client wrapper supporting multiple backend model providers."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Initialize the LLM Client.

        Args:
            provider: Model provider ('openai', 'anthropic', 'google', etc.). Defaults to LLM_PROVIDER or 'openai'.
            model: Model name. Defaults to LLM_MODEL or 'gpt-5.4-mini'.
            temperature: Sampling temperature.
            max_retries: Number of invoke retries on failure.
            retry_delay: Delay multiplier for retries.
            embedding_model: Optional override for provider's embedding model.
        """
        if provider is None:
            provider = os.environ.get("LLM_PROVIDER", "openai")
        if model is None:
            model = os.environ.get("LLM_MODEL", "gpt-5.4-mini")

        self.provider = provider.lower()
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        api_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        env_var = api_keys.get(self.provider)
        api_key = os.environ.get(env_var) if env_var else None
        if api_key:
            api_key = api_key.strip()

        logger.info("Initializing LLMClient for provider: %s, model: %s", provider, model)
        self.llm = init_chat_model(
            model,
            model_provider=provider,
            temperature=temperature,
            timeout=60.0,
            api_key=api_key
        )

        # Unified embedding setup
        self.embeddings = None
        if self.provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(
                    model=embedding_model or "text-embedding-3-small",
                    api_key=api_key
                )
            except ImportError:
                logger.warning("langchain-openai not installed. Embeddings will not be available.")
        elif self.provider == "google":
            try:
                from langchain_google_genai import GoogleGenAIEmbeddings
                self.embeddings = GoogleGenAIEmbeddings(
                    model=embedding_model or "models/embedding-001",
                    google_api_key=api_key
                )
            except ImportError:
                logger.warning("langchain-google-genai not installed. Embeddings will not be available.")
        else:
            # Fallback to OpenAI if installed
            try:
                from langchain_openai import OpenAIEmbeddings
                fallback_key = os.environ.get("OPENAI_API_KEY")
                if fallback_key:
                    fallback_key = fallback_key.strip()
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=fallback_key
                )
                logger.info("Using OpenAI fallback for embeddings.")
            except ImportError:
                pass

    def invoke(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Invoke LLM with retry logic. Returns the content string."""
        llm = self.llm
        if temperature is not None and hasattr(llm, "bind"):
            llm = llm.bind(temperature=temperature)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = llm.invoke(prompt)
                global_tracker.track_call(self.model, prompt, response)
                logger.debug("LLM invoke succeeded (attempt %d)", attempt)
                return str(response.content)
            except Exception as e:
                logger.warning(
                    "LLM invoke failed (attempt %d/%d): %s: %s",
                    attempt,
                    self.max_retries,
                    type(e).__name__,
                    e,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay * attempt)
        raise RuntimeError("Retries exhausted")

    def invoke_with_image(self, prompt: str, image_b64: str) -> str:
        """Invoke LLM with an image input."""
        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm.invoke([msg])
                global_tracker.track_call(self.model, prompt, response)
                logger.debug("LLM vision invoke succeeded (attempt %d)", attempt)
                return str(response.content)
            except Exception as e:
                logger.warning(
                    "LLM vision invoke failed (attempt %d/%d): %s: %s",
                    attempt,
                    self.max_retries,
                    type(e).__name__,
                    e,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay * attempt)
        raise RuntimeError("Retries exhausted")

    @staticmethod
    def parse_json_response(raw: str) -> Any:
        """Sanitize and parse raw LLM output string into JSON."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Fix LaTeX backslashes before letters (e.g. \text, \times, \frac, \beta, \lambda)
        cleaned = re.sub(r'\\([a-zA-Z]+)', lambda m: '\\\\' + m.group(1), cleaned)
        # Fix remaining unescaped single backslashes
        cleaned = re.sub(r'\\(?![/"bfnrtu\\])', r'\\\\', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response: %s\nRaw:\n%s", e, raw)
            raise

    def invoke_json(self, prompt: str, temperature: Optional[float] = None) -> Any:
        """Invoke LLM and parse the response strictly as JSON with sanitization."""
        raw = self.invoke(prompt, temperature=temperature)
        return self.parse_json_response(raw)

    def invoke_with_image_json(self, prompt: str, image_b64: str) -> Any:
        """Invoke LLM with image and parse the response strictly as JSON with sanitization."""
        raw = self.invoke_with_image(prompt, image_b64)
        return self.parse_json_response(raw)

