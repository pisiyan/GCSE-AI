"""Ingestion script for GCSE AI.

Loads raw PDFs (Specifications, QuestionPapers, MarkSchemes) for all configured subjects
and processes them into FAISS vector databases.
"""

import logging
import os
import sys
from config import SUBJECT_CONFIGS, load_subject_config
from load_and_store import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_matching_folders(root_dir: str) -> list[str]:
    """Find all matching folders (Specification, QuestionPapers, MarkSchemes) recursively."""
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

        for folder_path in folders_to_process:
            logger.info("Processing folder: %s", folder_path)
            db_manager.add_folder_database(folder_path, database_path)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    ingest_all_subjects(target)
