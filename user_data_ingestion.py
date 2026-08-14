import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
from pypdf import PdfReader
from llm_client import LLMClient

logger = logging.getLogger("user_data_ingestion")


class UserDataIngestion:
    """Manages document conversion, LLM item extraction, document classification,
    and canonical question code normalization for files in user_data/."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.cache_dir = "user_data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def convert_documents_to_text(self, user_data_dir: str = "user_data") -> List[Dict[str, str]]:
        """Scan flat user_data/ directory and convert all documents to text files.

        Args:
            user_data_dir: Path to user_data directory.

        Returns:
            List of dicts with 'filename', 'filepath', and 'text'.
        """
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
                    # Skip cover page (page 0) if multi-page PDF
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
                # Save converted text file to cache
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
        """Split a long text string into overlapping/clean paragraph chunks."""
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

    def _clean_json_str(self, json_str: str) -> str:
        """Sanitize raw LLM JSON response to fix invalid LaTeX backslash escapes."""
        cleaned = json_str.strip()
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
        return cleaned

    def extract_document_items(self, doc_file: Dict[str, str]) -> Dict[str, Any]:
        """Send LLM calls to extract and categorize items from a text file (chunked if long).

        Args:
            doc_file: Dict with 'filename' and 'text'.

        Returns:
            Dict containing 'doc_type' and combined 'items'.
        """
        full_text = doc_file["text"]
        text_chunks = self._split_text_into_chunks(full_text, max_chunk_len=18000)

        prompt_path = os.path.join("prompts", "extract_document_items.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        combined_doc_type = "unknown"
        all_extracted_items = []

        for chunk_idx, chunk_text in enumerate(text_chunks, start=1):
            prompt = prompt_template.replace("{document_text}", chunk_text)

            logger.info("Sending LLM extraction call for %s (chunk %d/%d)...", doc_file["filename"], chunk_idx, len(text_chunks))
            print(f"  [LLM Document Ingestion Call]: Extracting items from {doc_file['filename']} (part {chunk_idx}/{len(text_chunks)})...")

            raw_response = self.llm_client.invoke(prompt)
            cleaned_json = self._clean_json_str(raw_response)

            try:
                data = json.loads(cleaned_json, strict=False)
            except Exception as e:
                logger.warning("Standard json.loads failed for %s (chunk %d): %s. Attempting fallback backslash cleanup...", doc_file["filename"], chunk_idx, e)
                # Fallback: replace backslashes completely to guarantee valid JSON string parsing
                super_clean = raw_response.strip()
                if super_clean.startswith("```"):
                    lines = super_clean.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    super_clean = "\n".join(lines).strip()
                super_clean = super_clean.replace("\\", "\\\\")
                try:
                    data = json.loads(super_clean, strict=False)
                except Exception as e2:
                    logger.error("Failed to parse LLM JSON extraction for %s (chunk %d): %s", doc_file["filename"], chunk_idx, e2)
                    data = {}

            dt = data.get("doc_type", "").lower()
            if dt and dt != "unknown":
                combined_doc_type = dt

            items = data.get("items", [])
            all_extracted_items.extend(items)

        return {"doc_type": combined_doc_type, "items": all_extracted_items}

    def normalize_question_codes(self, raw_codes: List[str]) -> Dict[str, str]:
        """Send 1 LLM call to normalize all raw question codes across stores into canonical codes.

        Args:
            raw_codes: List of all raw code strings.

        Returns:
            Dict mapping raw_code -> canonical_code.
        """
        if not raw_codes:
            return {}

        prompt_path = os.path.join("prompts", "normalize_question_codes.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt = prompt_template.replace("{raw_codes_json}", json.dumps(raw_codes, indent=2))

        logger.info("Sending 1 LLM call to normalize %d question codes...", len(raw_codes))
        print(f"  [LLM Code Normalization Call]: Normalizing {len(raw_codes)} raw codes into canonical format...")

        raw_response = self.llm_client.invoke(prompt)
        cleaned_json = self._clean_json_str(raw_response)

        try:
            data = json.loads(cleaned_json, strict=False)
            mappings = data.get("code_mappings", {})
        except Exception as e:
            logger.error("Failed to parse LLM code normalization JSON: %s", e)
            mappings = {c: c for c in raw_codes}

        return mappings

    def ingest_user_data(self, user_data_dir: str = "user_data") -> Dict[str, Any]:
        """Full pipeline:
        1. Convert all user_data documents to text.
        2. 1 LLM call per file to extract items into QuestionStore, MarkSchemeStore, AnswerStore.
        3. 1 LLM call to normalize question codes across all stores.
        4. Return normalized stores ready for marking.

        Returns:
            Dict with 'QuestionStore', 'MarkSchemeStore', 'AnswerStore', 'CodeMappings'.
        """
        doc_files = self.convert_documents_to_text(user_data_dir)
        if not doc_files:
            return {"QuestionStore": {}, "MarkSchemeStore": {}, "AnswerStore": {}, "CodeMappings": {}}

        question_store_raw = []
        ms_store_raw = []
        answer_store_raw = []

        all_raw_codes = set()

        for doc_file in doc_files:
            extracted = self.extract_document_items(doc_file)
            doc_type = extracted.get("doc_type", "").lower()
            items = extracted.get("items", [])

            # Secondary fallback check if doc_type is ambiguous
            fn_lower = doc_file["filename"].lower()
            if "ms" in fn_lower or "markscheme" in fn_lower or "mark_scheme" in fn_lower:
                doc_type = "mark_scheme"
            elif "answer" in fn_lower:
                doc_type = "student_answer"
            elif "question" in fn_lower or "paper" in fn_lower:
                doc_type = "question_paper"

            for item in items:
                code = str(item.get("code") or "").strip()
                content = str(item.get("content") or "").strip()
                marks = item.get("marks")

                if code:
                    all_raw_codes.add(code)

                entry = {"raw_code": code, "content": content, "marks": marks, "file": doc_file["filename"]}

                if doc_type == "question_paper":
                    question_store_raw.append(entry)
                elif doc_type == "mark_scheme":
                    ms_store_raw.append(entry)
                elif doc_type == "student_answer":
                    answer_store_raw.append(entry)

        # Send 1 LLM call to normalize codes
        code_mappings = self.normalize_question_codes(list(all_raw_codes))

        # Re-index stores by cleaned canonical code key (appending content if split across chunks)
        question_store = {}
        for entry in question_store_raw:
            raw_c = entry["raw_code"]
            canon_c = code_mappings.get(raw_c, raw_c)
            entry["canonical_code"] = canon_c
            clean_k = str(canon_c).strip().replace(" ", "").lower()
            if clean_k in question_store:
                question_store[clean_k]["content"] = (question_store[clean_k]["content"] + "\n" + entry["content"]).strip()
            else:
                question_store[clean_k] = entry

        ms_store = {}
        for entry in ms_store_raw:
            raw_c = entry["raw_code"]
            canon_c = code_mappings.get(raw_c, raw_c)
            entry["canonical_code"] = canon_c
            clean_k = str(canon_c).strip().replace(" ", "").lower()
            if clean_k in ms_store:
                ms_store[clean_k]["content"] = (ms_store[clean_k]["content"] + "\n" + entry["content"]).strip()
            else:
                ms_store[clean_k] = entry

        answer_store = {}
        for entry in answer_store_raw:
            raw_c = entry["raw_code"]
            canon_c = code_mappings.get(raw_c, raw_c)
            entry["canonical_code"] = canon_c
            clean_k = str(canon_c).strip().replace(" ", "").lower()
            if clean_k in answer_store:
                answer_store[clean_k]["content"] = (answer_store[clean_k]["content"] + "\n" + entry["content"]).strip()
            else:
                answer_store[clean_k] = entry

        logger.info(
            "Ingestion complete: %d Questions, %d Mark Schemes, %d Answers indexed.",
            len(question_store), len(ms_store), len(answer_store)
        )

        return {
            "QuestionStore": question_store,
            "MarkSchemeStore": ms_store,
            "AnswerStore": answer_store,
            "CodeMappings": code_mappings
        }
