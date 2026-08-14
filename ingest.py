"""Ingestion script for GCSE AI.

Loads raw PDFs (Specifications, QuestionPapers, MarkSchemes) for all configured subjects
and processes them into FAISS vector databases.
"""

import logging
import os
import sys

from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"), override=True)

from config import SUBJECT_CONFIGS, load_subject_config
from load_and_store import DatabaseManager, VectorStore
from build_spec_summary import generate_specification_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_matching_folders(root_dir: str) -> list[str]:
    """Find all ingestable folders (Specification, QuestionPapers, MarkSchemes) recursively.

    Walks the full directory tree under ``root_dir``, including any
    ``Exam-Types/{ExamType}/`` subtrees, so folders such as
    ``Exam-Types/Christianity/questionPapers`` are discovered automatically.

    Args:
        root_dir: Root directory to search (e.g. ``data/ReligiousStudies/AQA``).

    Returns:
        List of absolute/relative folder paths that match the target names.
    """
    matching_folders = []
    target_names = {"specification", "questionpapers", "markschemes"}
    
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            if d.lower() in target_names:
                matching_folders.append(os.path.join(root, d))
                
    return matching_folders


def ingest_all_subjects(target_subject: str = None) -> None:
    """Read the configured subjects, run DatabaseManager, and store in FAISS vector stores."""
    for subject_key in SUBJECT_CONFIGS:
        if "-" not in subject_key:
            logger.warning("Skipping invalid subject key configuration: %s", subject_key)
            continue

        subject, examiner = subject_key.split("-", 1)
        if target_subject and subject.lower() != target_subject.lower():
            logger.info("Skipping subject %s as it does not match target %s", subject, target_subject)
            continue

        logger.info("Starting ingestion for subject: %s, examiner: %s", subject, examiner)

        try:
            load_subject_config(subject, examiner)
        except Exception as e:
            logger.warning("Failed to load configuration for %s (%s): %s, skipping.", subject, examiner, e)
            continue

        db_manager = DatabaseManager(subject, examiner)
        database_path = f"data/{subject}/{examiner}/{subject}-{examiner}-vectorDatabase"

        root_dir = f"data/{subject}/{examiner}"
        if not os.path.exists(root_dir):
            logger.warning("Root directory not found: %s", root_dir)
            continue

        folders_to_process = find_matching_folders(root_dir)

        if not folders_to_process:
            logger.warning("No ingestable folders found under %s", root_dir)
            continue

        # 1. Separate specification folders from question paper & mark scheme folders
        spec_folders = [f for f in folders_to_process if "specification" in f.lower()]
        other_folders = [f for f in folders_to_process if "specification" not in f.lower()]

        # 2. Ingest Specification FIRST
        if spec_folders:
            for folder_path in spec_folders:
                logger.info("Processing specification folder first: %s", folder_path)
                db_manager.add_folder_database(folder_path, database_path)

        # 3. Generate Specification Content Summary immediately after specification ingestion (if not already existing)
        try:
            generate_specification_summary(subject, examiner, skip_if_exists=True)
        except Exception as e:
            logger.error("Failed to generate specification content summary for %s (%s): %s", subject, examiner, e)

        # 4. Ingest QuestionPapers and MarkSchemes AFTER specification
        for folder_path in other_folders:
            logger.info("Processing paper folder: %s", folder_path)
            db_manager.add_folder_database(folder_path, database_path)

        # 5. Generate a plain text string output dump of the ingested vector database
        vdb = VectorStore(database_path)
        dump_text = vdb.dump_database_to_string(subject, examiner)
        out_file = f"data/{subject}/{examiner}/{subject}-{examiner}-vectorDatabase_contents.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(dump_text)
        logger.info("Saved vector database text dump to: %s", out_file)
        print(f"\nSaved readable vector database string dump to: {out_file}\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    ingest_all_subjects(target)
