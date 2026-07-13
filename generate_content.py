"""GCSE AI — Main application entry point.

Orchestrates exam generation, marking, and revision material creation
by wiring together the config, LLM, similarity, generator, and marker modules.
"""

import logging
import os
import json
from typing import Any

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import load_subject_config, SubjectConfig
from llm_client import LLMClient
from similarity import SimilarityEngine
from exam_generator import ExamStructureBuilder, QuestionGenerator
from exam_marker import ExamMarker


# Load environment variables relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"), override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key.strip()


class GcseAssistant:
    """Main GCSE AI assistant that orchestrates all functionality.

    Wires together configuration, LLM client, similarity engine,
    exam generation, and exam marking modules.
    """

    def __init__(self, subject: str, examiner: str) -> None:
        """Initialize the GCSE Assistant.

        Args:
            subject: Subject name (e.g., "Biology").
            examiner: Exam board name (e.g., "Edexcel").
        """
        logger.info("Initializing GCSE Assistant for %s (%s)...", subject, examiner)

        self.subject = subject
        self.examiner = examiner
        self.config = load_subject_config(subject, examiner)

        # Shared embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector store and retrievers
        vdb_path = f"data/{subject}/{examiner}/{subject}-{examiner}-vectorDatabase"
        if not os.path.exists(vdb_path):
            root_dir = os.path.dirname(os.path.abspath(__file__))
            vdb_path = os.path.join(root_dir, "data", subject, examiner, f"{subject}-{examiner}-vectorDatabase")
        self.vectorstore = FAISS.load_local(
            vdb_path, self.embedding_model, allow_dangerous_deserialization=True
        )

        self.spec_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.config.spec_search_kwargs_k,
                "filter": {"type": "Specification"},
            },
        )
        self.ms_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config.ms_search_kwargs_k,
                "filter": {"type": "MarkScheme"},
            },
        )

        # Single shared LLM client (replaces 3 identical instances)
        self.llm_client = LLMClient(model="gpt-4o", temperature=0)

        # QA chains
        self.spec_qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_client.llm, retriever=self.spec_retriever
        )
        self.ms_qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_client.llm, retriever=self.ms_retriever
        )

        # Load question data and mark schemes from vector store
        self.questions, self.mark_schemes = self._load_from_vectorstore()

        # Load full specification text
        self.specification_text = self._load_full_specification()

        # Load prompt and query templates
        root_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_path = "prompts" if os.path.exists("prompts") else os.path.join(root_dir, "prompts")
        queries_path = "queries" if os.path.exists("queries") else os.path.join(root_dir, "queries")
        self.prompts = self._load_templates(prompts_path)
        self.queries = self._load_templates(queries_path)

        # Similarity engine (batched API calls)
        self.similarity = SimilarityEngine(self.llm_client, self.embedding_model)

        # Exam structure builder
        self.structure_builder = ExamStructureBuilder(
            config=self.config,
            questions=self.questions,
        )

        # Question generator
        self.question_generator = QuestionGenerator(
            config=self.config,
            llm_client=self.llm_client,
            similarity_engine=self.similarity,
            questions=self.questions,
            prompts=self.prompts,
            queries=self.queries,
            spec_qa_chain=self.spec_qa_chain,
            embedding_model=self.embedding_model,
            specification_text=self.specification_text,
        )

        # Exam marker
        self.exam_marker = ExamMarker(
            config=self.config,
            llm_client=self.llm_client,
            similarity_engine=self.similarity,
            questions=self.questions,
            mark_schemes=self.mark_schemes,
            prompts=self.prompts,
            queries=self.queries,
            spec_qa_chain=self.spec_qa_chain,
            ms_qa_chain=self.ms_qa_chain,
            subject=subject,
            examiner=examiner,
            embedding_model=self.embedding_model,
            specification_text=self.specification_text,
        )

        logger.info("Initialization complete")

    def _load_from_vectorstore(self) -> tuple[list[dict], list[dict]]:
        """Load questions and mark schemes directly from the FAISS database docstore.

        Returns:
            A tuple of (questions_list, mark_schemes_list).
        """
        questions = []
        mark_schemes = []
        
        # FAISS docstore stores all documents in a local dict
        all_docs = list(self.vectorstore.docstore._dict.values())
        logger.info("Retrieved %d total documents from vector database", len(all_docs))

        for doc in all_docs:
            meta = doc.metadata.copy()
            doc_type = meta.get("type", "")

            if doc_type == "QuestionPaper":
                # Reconstruct question dict
                q_dict = meta.copy()
                q_dict["question_content"] = doc.page_content
                # Restore 'type' from 'q_type' to match expected schema downstream
                if "q_type" in q_dict:
                    q_dict["type"] = q_dict.pop("q_type")
                
                # De-serialize parent question structure
                if "parent_question_structure" in q_dict and isinstance(q_dict["parent_question_structure"], str):
                    try:
                        q_dict["parent_question_structure"] = json.loads(q_dict["parent_question_structure"])
                    except json.JSONDecodeError:
                        logger.error("Failed to decode parent_question_structure: %s", q_dict["parent_question_structure"])
                
                questions.append(q_dict)

            elif doc_type == "MarkScheme" or meta.get("doc_class") == "mark_scheme":
                # Reconstruct mark scheme dict
                ms_dict = meta.copy()
                ms_dict["question_content"] = doc.page_content
                mark_schemes.append(ms_dict)

        logger.info("Loaded %d questions and %d mark schemes from vector database", len(questions), len(mark_schemes))
        return questions, mark_schemes

    @staticmethod
    def _load_templates(folder_path: str) -> dict[str, str]:
        """Load all prompt/query template files from a folder.

        Strips .txt extensions from filenames to create dictionary keys.

        Args:
            folder_path: Path to the templates folder.

        Returns:
            Dict mapping template names to their content.
        """
        templates: dict[str, str] = {}
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            if os.path.isfile(full_path):
                with open(full_path, "r") as file:
                    content = file.read()
                key = filename.replace(".txt", "")
                templates[key] = content
        logger.debug("Loaded %d templates from %s", len(templates), folder_path)
        return templates

    # --- Convenience methods that delegate to sub-modules ---

    def make_exam(
        self, exam_topic: str, total_marks: int, topics: list[str]
    ) -> dict[str, Any]:
        """Generate a complete exam.

        Args:
            exam_topic: Broad exam category (e.g., "Higher").
            total_marks: Total marks for the exam.
            topics: List of topic areas to cover.

        Returns:
            Dict with exam structure and generated questions.
        """
        return self.question_generator.generate_exam(
            exam_topic=exam_topic,
            total_marks=total_marks,
            topics=topics,
            subject=self.subject,
            structure_builder=self.structure_builder,
        )

    def mark_question(self) -> str:
        """Mark a question from user-uploaded files.

        Returns:
            Marking feedback string.
        """
        return self.exam_marker.mark_question_from_files()

    def mark_exam(self, questions: list[dict], topic: str) -> list[dict]:
        """Mark a full exam.

        Args:
            questions: List of question dicts with answers.
            topic: The exam topic.

        Returns:
            List of marking result dicts.
        """
        return self.exam_marker.mark_exam(questions, topic)

    def revision_materials(self, topic: str) -> str:
        """Generate revision materials for a topic.

        Args:
            topic: The topic to generate materials for.

        Returns:
            Revision materials text.
        """
        return self.question_generator.get_revision_materials(
            topic=topic,
            subject=self.subject,
            spec_qa_chain=self.spec_qa_chain,
        )



    def _load_full_specification(self) -> str:
        """Find the specification PDF, extract text using pypdf if not cached, and return it."""
        spec_dir = f"data/{self.subject}/{self.examiner}/Specification"
        if not os.path.exists(spec_dir):
            root_dir = os.path.dirname(os.path.abspath(__file__))
            spec_dir = os.path.join(root_dir, "data", self.subject, self.examiner, "Specification")
            
        spec_dir = os.path.normpath(spec_dir)
        if not os.path.exists(spec_dir):
            logger.warning("Specification directory not found under %s", spec_dir)
            return "No specification details found."
            
        # Find PDF file
        pdf_file = None
        for filename in os.listdir(spec_dir):
            if filename.lower().endswith(".pdf"):
                pdf_file = os.path.normpath(os.path.join(spec_dir, filename))
                break
                
        if not pdf_file:
            logger.warning("No specification PDF file found in %s", spec_dir)
            return "No specification details found."
            
        # Check if text cache exists
        txt_file = os.path.normpath(pdf_file.rsplit(".", 1)[0] + ".txt")
        if os.path.exists(txt_file):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    logger.info("Loaded specification text from cache: %s", txt_file)
                    return f.read()
            except Exception as e:
                logger.error("Failed to read cached spec text: %s", e)
                
        # If cache doesn't exist, extract using pypdf
        logger.info("Extracting text from specification PDF: %s", pdf_file)
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_file)
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            full_text = "\n\n".join(text_parts)
            
            # Save cache
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            logger.info("Saved specification text cache to: %s", txt_file)
            return full_text
        except Exception as e:
            logger.error("Failed to extract text from specification PDF %s: %s", pdf_file, e)
            return "No specification details found."

def format_exam_as_markdown(subject: str, examiner: str, exam_topic: str, exam_data: dict) -> str:
    """Format a generated exam dictionary into a clean, readable markdown document.
    
    Args:
        subject: Subject name.
        examiner: Exam board name.
        exam_topic: Exam topic.
        exam_data: Dict with exam structure and generated questions.
        
    Returns:
        Formatted markdown string.
    """
    md = []
    md.append(f"# GCSE {subject} ({examiner}) - {exam_topic} Exam")
    md.append("")
    
    # Calculate total marks from questions
    total_marks = 0
    questions_by_topic = exam_data.get("questions", {})
    for topic, q_list in questions_by_topic.items():
        for q in q_list:
            if "sub_questions" in q:
                for sq in q["sub_questions"]:
                    if "sub_parts" in sq:
                        for gq in sq["sub_parts"]:
                            total_marks += gq.get("marks", 0)
                    else:
                        total_marks += sq.get("marks", 0)
            else:
                total_marks += q.get("marks", 0)
                
    md.append(f"**Total Marks:** {total_marks}")
    md.append("")
    
    for topic, q_list in questions_by_topic.items():
        md.append(f"## Topic: {topic}")
        md.append("")
        for q in q_list:
            q_num = q.get("number", "")
            subtopic = q.get("subtopic", "")
            
            # Check if it's a parent question with sub-parts
            if "sub_questions" in q:
                parent_desc = q.get("parent_description", "")
                md.append(f"### Question {q_num} (Subtopic: {subtopic})")
                if parent_desc:
                    md.append(parent_desc)
                    md.append("")
                
                for sq in q["sub_questions"]:
                    label = sq.get("label", "")
                    sq_subtopic = sq.get("subtopic", "")
                    sq_subtopic_str = f" (Subtopic: {sq_subtopic})" if sq_subtopic else ""
                    
                    if "sub_parts" in sq:
                        context = sq.get("context", "")
                        md.append(f"**{label}**{sq_subtopic_str} {context}")
                        md.append("")
                        for gq in sq["sub_parts"]:
                            g_label = gq.get("label", "")
                            g_text = gq.get("text", "")
                            g_marks = gq.get("marks", 0)
                            g_subtopic = gq.get("subtopic", "")
                            g_subtopic_str = f" (Subtopic: {g_subtopic})" if g_subtopic else ""
                            md.append(f"  * **{g_label}** {g_text}{g_subtopic_str} *({g_marks} mark{'s' if g_marks > 1 else ''})*")
                            md.append("")
                    else:
                        sq_text = sq.get("text", "")
                        sq_marks = sq.get("marks", 0)
                        md.append(f"* **{label}** {sq_text}{sq_subtopic_str} *({sq_marks} mark{'s' if sq_marks > 1 else ''})*")
                        md.append("")
            else:
                # Standalone question
                q_text = q.get("text", "")
                q_marks = q.get("marks", 0)
                md.append(f"### Question {q_num} (Subtopic: {subtopic}) *({q_marks} mark{'s' if q_marks > 1 else ''})*")
                md.append(q_text)
                md.append("")
                
    return "\n".join(md)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. Biology (Edexcel)
    print("\n==================================================")
    print("GENERATING BIOLOGY EXAM")
    print("==================================================")
    assistant_bio = GcseAssistant("Biology", "Edexcel")
    marks_bio = 100
    exam_topic_bio = "Higher"
    # topics_bio = [
    #     "Topic 6 – Plant structures and their functions",
    # ]
    # result_bio = assistant_bio.make_exam(exam_topic_bio, marks_bio, topics_bio)
    # print("Generated Exam (Bio):")
    # print(json.dumps(result_bio, indent=2))
    # print("\n--- Running Bio Exam Quality Analysis ---")
    # report_bio = assistant_bio.analyze_exam(result_bio)
    # report_bio_md = assistant_bio.quality_analyzer.generate_markdown_report(report_bio)
    # print(report_bio_md)
    #
    # # Format and save Biology Exam & Report as a markdown file
    # exam_bio_md = format_exam_as_markdown("Biology", "Edexcel", exam_topic_bio, result_bio)
    # full_bio_content = f"{exam_bio_md}\n\n---\n\n{report_bio_md}"
    # bio_filename = "Biology_Edexcel_exam.md"
    # with open(bio_filename, "w", encoding="utf-8") as f:
    #     f.write(full_bio_content)
    # print(f"\n[SUCCESS] Saved Biology exam and quality report to: {bio_filename}")

    # 2. Religious Studies (AQA)
    # print("\n==================================================")
    # print("GENERATING RELIGIOUS STUDIES EXAM")
    # print("==================================================")
    # assistant_rs = GcseAssistant("ReligiousStudies", "AQA")
    # marks_rs = 100
    # exam_topic_rs = "Christianity"
    # topics_rs = [
    #     "Beliefs",
    # ]
    # result_rs = assistant_rs.make_exam(exam_topic_rs, marks_rs, topics_rs)
    # print("Generated Exam (RS):")
    # print(json.dumps(result_rs, indent=2))
    # print("\n--- Running RS Exam Quality Analysis ---")
    # report_rs = assistant_rs.analyze_exam(result_rs)
    # report_rs_md = assistant_rs.quality_analyzer.generate_markdown_report(report_rs)
    # print(report_rs_md)
    #
    # # Format and save RS Exam & Report as a markdown file
    # exam_rs_md = format_exam_as_markdown("ReligiousStudies", "AQA", exam_topic_rs, result_rs)
    # full_rs_content = f"{exam_rs_md}\n\n---\n\n{report_rs_md}"
    # rs_filename = "ReligiousStudies_AQA_exam.md"
    # with open(rs_filename, "w", encoding="utf-8") as f:
    #     f.write(full_rs_content)
    # print(f"\n[SUCCESS] Saved Religious Studies exam and quality report to: {rs_filename}")
    #

    # 3. Feedback Quality Analysis
    print("\n==================================================")
    print("RUNNING FEEDBACK QUALITY ANALYSIS")
    print("==================================================")
    question_file = "user_data/question/question.txt"
    ms_file = "user_data/ms/ms.txt"
    answer_file = "user_data/answer/answer.txt"

    if os.path.exists(question_file) and os.path.exists(ms_file) and os.path.exists(answer_file):
        with open(question_file, "r", encoding="utf-8") as f:
            question_text = f.read().strip()
        with open(ms_file, "r", encoding="utf-8") as f:
            mark_scheme_text = f.read().strip()
        with open(answer_file, "r", encoding="utf-8") as f:
            answer_text = f.read().strip()

        try:
            marks = assistant_bio.exam_marker.get_marks_from_question(question_text)
        except Exception:
            marks = 6

        print("Marking student answer...")
        feedback_str = assistant_bio.exam_marker.mark_answer(
            answer=answer_text,
            mark_scheme=mark_scheme_text,
            question=question_text,
            marks=marks
        )
        print(f"Generated Feedback:\n{feedback_str}\n")


    else:
        print("[WARNING] Sample files in user_data/ not found. Skipping feedback quality evaluation.")

