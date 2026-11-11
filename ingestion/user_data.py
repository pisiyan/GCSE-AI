"""User data ingestion and extraction module for GCSE AI."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from pypdf import PdfReader

from llm.client import LLMClient
from llm.prompts import load_prompt

logger = logging.getLogger(__name__)


class UserDataIngestion:
    """Manages document conversion and item extraction for files in user_data/."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.cache_dir = "user_data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def convert_documents_to_text(self, user_data_dir: str = "user_data") -> List[Dict[str, str]]:
        """Scan flat user_data/ directory and convert all documents to text files."""
        if not os.path.exists(user_data_dir):
            logger.warning("Directory %s does not exist.", user_data_dir)
            return []

        doc_files = []
        for fn in os.listdir(user_data_dir):
            fp = os.path.join(user_data_dir, fn)
            if not os.path.isfile(fp):
                continue

            text_content = ""
            ext = os.path.splitext(fn)[1].lower()

            if ext == ".pdf":
                try:
                    reader = PdfReader(fp)
                    pages_text = []
                    start_page = 1 if len(reader.pages) > 1 else 0
                    for page in reader.pages[start_page:]:
                        pt = page.extract_text()
                        if pt:
                            pages_text.append(pt)
                    text_content = "\n\n".join(pages_text)
                except Exception as e:
                    logger.error("PDF conversion failed for %s: %s", fp, e)
            elif ext in (".txt", ".md"):
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        text_content = f.read()
                except Exception as e:
                    logger.error("Text file reading failed for %s: %s", fp, e)

            if text_content.strip():
                txt_fn = f"{os.path.splitext(fn)[0]}.txt"
                txt_fp = os.path.join(self.cache_dir, txt_fn)
                with open(txt_fp, "w", encoding="utf-8") as f:
                    f.write(text_content)

                doc_files.append({
                    "filename": fn,
                    "filepath": fp,
                    "cache_txt_path": txt_fp,
                    "text": text_content
                })

        logger.info("Converted %d document files from %s to text.", len(doc_files), user_data_dir)
        return doc_files

    def _split_text_into_chunks(self, text: str, max_chunk_len: int = 18000) -> List[str]:
        """Split a long text string into clean paragraph chunks."""
        if len(text) <= max_chunk_len:
            return [text]

        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p) + 2
            if current_len + p_len > max_chunk_len and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_len = p_len
            else:
                current_chunk.append(p)
                current_len += p_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def extract_document_items(self, doc_file: Dict[str, str]) -> Dict[str, Any]:
        """Send LLM calls to extract and categorize items from a text file."""
        full_text = doc_file["text"]
        text_chunks = self._split_text_into_chunks(full_text, max_chunk_len=18000)

        combined_doc_type = "unknown"
        all_extracted_items = []

        for chunk_idx, chunk_text in enumerate(text_chunks, start=1):
            prompt = load_prompt("extract_document_items", document_text=chunk_text)
            logger.info("Extracting items from %s (part %d/%d)...", doc_file["filename"], chunk_idx, len(text_chunks))

            try:
                chunk_data = self.llm_client.invoke_json(prompt)
                extracted_type = chunk_data.get("doc_type", "unknown")
                if combined_doc_type == "unknown" and extracted_type != "unknown":
                    combined_doc_type = extracted_type
                items = chunk_data.get("items", [])
                all_extracted_items.extend(items)
            except Exception as e:
                logger.error("Failed to parse items for %s chunk %d: %s", doc_file["filename"], chunk_idx, e)

        return {
            "doc_type": combined_doc_type,
            "filename": doc_file["filename"],
            "items": all_extracted_items
        }
