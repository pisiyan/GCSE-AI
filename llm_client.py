"""Unified LLM client module for GCSE AI.

Supports dynamic provider switching (OpenAI, Anthropic, Google)
using LangChain's init_chat_model and unified embeddings.
"""

import os
import json
import logging
import re
import time
from typing import Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_classic.chains import RetrievalQA

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client wrapper supporting multiple backend model providers.

    Leverages LangChain's unified init_chat_model API to make switching between
    providers (OpenAI, Anthropic, Google, etc.) seamless.
    """

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
            model: Model name. Defaults to LLM_MODEL or 'gpt-4o-mini'.
            temperature: Sampling temperature.
            max_retries: Number of invoke retries on failure.
            retry_delay: Delay multiplier for retries.
            embedding_model: Optional override for the provider's embedding model.
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
                result = llm.invoke(prompt).content
                logger.debug("LLM invoke succeeded (attempt %d)", attempt)
                return str(result)
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
                result = self.llm.invoke([msg]).content
                logger.debug("Image LLM invoke succeeded (attempt %d)", attempt)
                return str(result)
            except Exception as e:
                logger.warning(
                    "Image LLM invoke failed (attempt %d/%d): %s: %s",
                    attempt,
                    self.max_retries,
                    type(e).__name__,
                    e,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay * attempt)
        raise RuntimeError("Retries exhausted")

    def invoke_qa(self, qa_chain: RetrievalQA, query: str) -> str:
        """Invoke a QA chain with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                result = qa_chain.invoke({"query": query})["result"]
                logger.debug("QA chain invoke succeeded (attempt %d)", attempt)
                return str(result)
            except Exception as e:
                logger.warning(
                    "QA chain invoke failed (attempt %d/%d): %s: %s",
                    attempt,
                    self.max_retries,
                    type(e).__name__,
                    e,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay * attempt)
        raise RuntimeError("Retries exhausted")

    def invoke_json(self, prompt: str, temperature: Optional[float] = None) -> Any:
        """Invoke LLM and parse JSON response with error handling.

        Strips markdown code fences if present before parsing.
        Retries if response is not valid JSON.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.invoke(prompt, temperature=temperature)
                cleaned = self._strip_code_fences(raw)
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s. Raw: %.200s",
                    attempt,
                    self.max_retries,
                    e,
                    raw,
                )
                if attempt == self.max_retries:
                    raise ValueError(
                        f"Failed to parse JSON after {self.max_retries} attempts. Last response: {raw[:500]}"
                    ) from e
                time.sleep(self.retry_delay * attempt)
        raise RuntimeError("Retries exhausted")

    def get_embeddings(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        """Get embeddings for a list of texts in a single batched call."""
        if not self.embeddings:
            raise ValueError(
                f"Embeddings client is not configured for provider: '{self.provider}'. "
                "Ensure langchain-openai (for OpenAI fallback) or the provider's specific "
                "embeddings library is installed."
            )

        # Optional: dynamic model override for OpenAIEmbeddings
        emb_client = self.embeddings
        if model and hasattr(emb_client, "model"):
            try:
                # Create a copy or clone with new model if possible, or just set it temporarily
                emb_client.model = model
            except Exception:
                pass

        for attempt in range(1, self.max_retries + 1):
            try:
                return emb_client.embed_documents(texts)
            except Exception as e:
                logger.warning(
                    "Embeddings call failed (attempt %d/%d): %s: %s",
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
    def _strip_code_fences(text: str) -> str:
        """Strip markdown code fences and external text from LLM responses."""
        text = text.strip()
        
        # 1. Look for ```json ... ``` anywhere in the text
        pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # 2. If no code fences, but there is some leading/trailing text,
        # find the first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end+1].strip()
            
        return text
