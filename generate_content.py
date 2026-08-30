"""GCSE AI — Main application entry point.

Orchestrates exam generation, marking, and revision material creation
by wiring together the config, LLM, similarity, generator, and marker modules.
"""

import logging
import os
import json
from typing import Any, Optional, List, Dict

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import load_subject_config, SubjectConfig
from llm_client import LLMClient
from similarity import SimilarityEngine
from exam_generator import ExamStructureBuilder, QuestionGenerator, generate_answer_lines
from exam_marker import ExamMarker
from topic_tester import TopicTester


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

        # Single shared LLM client (model and provider determined by LLMClient defaults / env)
        self.llm_client = LLMClient()

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

        # Topic tester
        self.topic_tester = TopicTester(
            config=self.config,
            llm_client=self.llm_client,
            specification_text=self.specification_text,
            spec_qa_chain=self.spec_qa_chain,
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
                
                # De-serialize sub_questions
                if "sub_questions" in q_dict and isinstance(q_dict["sub_questions"], str):
                    try:
                        q_dict["sub_questions"] = json.loads(q_dict["sub_questions"])
                    except json.JSONDecodeError:
                        logger.error("Failed to decode sub_questions: %s", q_dict["sub_questions"])

                # De-serialize image_paths
                if "image_paths" in q_dict and isinstance(q_dict["image_paths"], str):
                    try:
                        q_dict["image_paths"] = json.loads(q_dict["image_paths"])
                    except json.JSONDecodeError:
                        logger.error("Failed to decode image_paths: %s", q_dict["image_paths"])
                
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

    @staticmethod
    def get_valid_exam_types(subject: str, examiner: str) -> list[str]:
        """Discover valid exam types for a given subject and examiner.

        Scans the `data/{subject}/{examiner}/Exam-Types/` directory and returns
        the names of all immediate subdirectories, each representing one exam type
        (e.g. 'Higher', 'Foundation', 'Christianity').

        Args:
            subject: Subject name (e.g., "Biology").
            examiner: Exam board name (e.g., "Edexcel").

        Returns:
            Sorted list of valid exam type names, or an empty list if the
            Exam-Types directory does not exist.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Support both relative and absolute paths
        for base in ("data", os.path.join(script_dir, "data")):
            exam_types_dir = os.path.join(base, subject, examiner, "Exam-Types")
            if os.path.isdir(exam_types_dir):
                return sorted(
                    name
                    for name in os.listdir(exam_types_dir)
                    if os.path.isdir(os.path.join(exam_types_dir, name))
                )
        logger.warning(
            "Exam-Types directory not found for %s/%s", subject, examiner
        )
        return []

    def make_exam(
        self,
        exam_type: str,
        total_marks: int,
        topics: list[str],
        user_preferences: Optional[Any] = None,
        num_questions: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate a complete exam.

        Args:
            exam_type: Exam type/tier (e.g., "Higher", "Christianity").
            total_marks: Total marks for the exam.
            topics: List of topic areas to cover.
            user_preferences: Custom preferences dict/string for exam generation.
            num_questions: Optional requested number of questions in the exam.

        Returns:
            Dict with exam structure and generated questions.
        """
        return self.question_generator.generate_exam(
            exam_type=exam_type,
            total_marks=total_marks,
            topics=topics,
            subject=self.subject,
            structure_builder=self.structure_builder,
            user_preferences=user_preferences,
            num_questions=num_questions,
        )

    def export_exam_pdf(
        self,
        exam_output: dict[str, Any],
        output_pdf_path: str,
    ) -> str:
        """Export a generated exam output dict into a past paper PDF document.

        Args:
            exam_output: Exam dict returned by make_exam().
            output_pdf_path: Target output path for the PDF.

        Returns:
            Path to the generated PDF file.
        """
        from exam_generator import render_exam_pdf
        return render_exam_pdf(
            exam_output=exam_output,
            output_pdf_path=output_pdf_path,
            subject=self.subject,
            examiner=self.examiner,
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

    def create_topic_test_session(self, desired_area: str) -> tuple[dict, list[dict]]:
        """Start a topic content test session.

        Uses 1 LLM call to compile target subtopics from specification summary,
        and 1 LLM call to generate study questions and perfect reference answers.

        Args:
            desired_area: The study area requested by student.

        Returns:
            Tuple of (compiled_scope_dict, questions_and_answers_list).
        """
        compiled_scope = self.topic_tester.compile_target_subtopics(
            desired_area=desired_area,
            subject=self.subject,
            examiner=self.examiner,
        )
        qa_list = self.topic_tester.generate_questions_and_answers(
            compiled_scope=compiled_scope,
            subject=self.subject,
            examiner=self.examiner,
        )
        return compiled_scope, qa_list

    def evaluate_topic_test_answer(
        self, question: str, perfect_answer: str, student_answer: str, subtopic: str = ""
    ) -> dict:
        """Evaluate a student's answer against the exemplar perfect answer.

        Args:
            question: Question text.
            perfect_answer: Model answer.
            student_answer: Student response.
            subtopic: Subtopic name.

        Returns:
            Evaluation result dict.
        """
        return self.topic_tester.evaluate_student_answer(
            question=question,
            perfect_answer=perfect_answer,
            student_answer=student_answer,
            subtopic=subtopic,
        )

    def save_topic_test_report(
        self, desired_area: str, results: list[dict], output_dir: str = "test_outputs"
    ) -> str:
        """Save a markdown performance report for a topic test session.

        Args:
            desired_area: Target topic area.
            results: List of question testing result dicts.
            output_dir: Output folder.

        Returns:
            File path of saved report.
        """
        return self.topic_tester.generate_performance_report(
            subject=self.subject,
            examiner=self.examiner,
            desired_area=desired_area,
            test_results=results,
            output_dir=output_dir,
        )

    def generate_single_question(
        self,
        topic: str,
        marks: int = 4,
        subtopic: str = "",
        q_type: str = "basic",
        exam_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate a single standalone or parent question for a target topic and mark count."""
        et = exam_type or self.examiner
        mark_struct = [marks, q_type] if q_type == "parent" else marks
        task = {
            "mark_structure": mark_struct,
            "exam_type": et,
            "subtopic": subtopic or topic,
            "subject": self.subject,
            "number": 1
        }
        return self.question_generator._execute_generation_task(task)

    def generate_mark_scheme(
        self,
        question: str,
        marks: int = 4,
        topic: str = "General",
    ) -> str:
        """Create a mark scheme for a single question."""
        return self.exam_marker.create_mark_scheme(question=question, marks=marks, topic=topic)

    def generate_model_answer(
        self,
        question: str,
        mark_scheme: str = "",
        marks: int = 4,
        topic: str = "General",
    ) -> str:
        """Generate a full-mark exemplar model answer for a question."""
        if not mark_scheme or not mark_scheme.strip():
            mark_scheme = self.generate_mark_scheme(question=question, marks=marks, topic=topic)
        return self.exam_marker.generate_model_answer(question=question, mark_scheme=mark_scheme)

    def get_spec_breakdown(self, topic: Optional[str] = None) -> dict[str, Any]:
        """Retrieve specification codes, topics, and subtopics breakdown."""
        if topic and topic.strip():
            tree = self.exam_marker._get_spec_tree_cached(topic, self.examiner)
            return {
                "subject": self.subject,
                "examiner": self.examiner,
                "topic": topic,
                "spec_code": tree.get("spec_code", topic),
                "subtopics": tree.get("subtopics", [topic])
            }

        sum_path = f"data/{self.subject}/{self.examiner}/{self.subject}-{self.examiner}-specification_summary.json"
        if not os.path.exists(sum_path):
            root_dir = os.path.dirname(os.path.abspath(__file__))
            sum_path = os.path.join(root_dir, "data", self.subject, self.examiner, f"{self.subject}-{self.examiner}-specification_summary.json")

        summary_data = {}
        if os.path.exists(sum_path):
            try:
                with open(sum_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
            except Exception as e:
                logger.warning("Could not read spec summary: %s", e)

        return {
            "subject": self.subject,
            "examiner": self.examiner,
            "summary": summary_data
        }

    def explain_command_word(self, command_word: str = "", question: str = "") -> str:
        """Explain GCSE command word requirements, mark criteria, and structure."""
        cw = command_word.strip()
        if not cw and question:
            try:
                cw_prompt = self.prompts.get("get_command_word", "Extract command word: {question}").format(question=question)
                cw = self.llm_client.invoke(cw_prompt).strip()
            except Exception as e:
                logger.warning("Failed to extract command word: %s", e)

        if not cw:
            cw = "Explain"

        prompt = (
            f"As an expert GCSE {self.subject} ({self.examiner}) examiner, provide clear guidance on the GCSE command word '{cw}'.\n"
            f"Include:\n"
            f"1. **Official Definition & Examiner Expectation**: What this command word demands.\n"
            f"2. **How to Structure Full-Mark Answers**: Key steps and phrases to use.\n"
            f"3. **Common Student Mistakes & Pitfalls**: What causes lost marks.\n"
            f"4. **Example Demonstration**: A quick 2-line mini example showing correct vs incorrect response style.\n\n"
            f"Format clearly with bold headers and bullet points (•)."
        )
        return self.llm_client.invoke(prompt)





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

def _safe_marks_str(marks: Any) -> str:
    """Format marks string safely, handling None or non-integer values gracefully."""
    if marks is None:
        return ""
    try:
        m = int(marks)
        return f" *({m} mark{'s' if m > 1 else ''})*"
    except (ValueError, TypeError):
        return ""


def format_single_question_as_markdown(subject: str, examiner: str, q: dict) -> str:
    """Format a single generated question dictionary into clean markdown."""
    md = []
    q_num = q.get("number", "1)")
    marks = q.get("marks")
    subtopic = q.get("subtopic", "")
    matched_subtopic = q.get("matched_subtopic", "")
    sim_score = q.get("similarity_score")

    marks_str = _safe_marks_str(marks)
    md.append(f"# 📝 GCSE {subject} ({examiner}) - Single Question{marks_str}")
    md.append("")
    md.append(f"**Subject:** `{subject}` | **Exam Board:** `{examiner}`")

    if subtopic:
        md.append(f"* **Target Subtopic:** `{subtopic}`")
    if matched_subtopic:
        md.append(f"* **Matched Question Subtopic:** `{matched_subtopic}`")
    if sim_score is not None:
        sim_pct = f"{sim_score * 100:.1f}%" if isinstance(sim_score, float) and sim_score <= 1.0 else f"{sim_score}"
        md.append(f"* **Semantic Similarity:** `{sim_pct}`")
    md.append("")

    if "sub_questions" in q:
        p_desc = q.get("parent_description", "")
        if p_desc:
            md.append(f"### Question Context")
            md.append(p_desc)
            md.append("")
        md.append("### Sub-Questions")
        for sq in q.get("sub_questions", []):
            label = sq.get("label", "")
            sq_text = sq.get("text", "")
            sq_marks = sq.get("marks")
            md.append(f"* **{label}** {sq_text}{_safe_marks_str(sq_marks)}")
            md.append("")
            md.append(generate_answer_lines(sq_marks))
            md.append("")
    else:
        q_text = q.get("text", "")
        md.append("### Question")
        md.append(q_text)
        md.append("")
        md.append(generate_answer_lines(marks))
        md.append("")

    return "\n".join(md)


def format_exam_as_markdown(subject: str, examiner: str, exam_type: str, exam_data: dict) -> str:
    """Format a generated exam dictionary into a clean, readable markdown document.
    
    Args:
        subject: Subject name.
        examiner: Exam board name.
        exam_type: Exam type/tier (e.g., 'Higher', 'Christianity').
        exam_data: Dict with exam structure and generated questions.
        
    Returns:
        Formatted markdown string.
    """
    md = []
    md.append(f"# GCSE {subject} ({examiner}) - {exam_type} Exam")
    md.append("")
    
    # Calculate total marks from questions
    total_marks = 0
    questions_by_topic = exam_data.get("questions", {})
    for topic, q_list in questions_by_topic.items():
        for q in q_list:
            if "sub_questions" in q:
                if q.get("marks") is not None:
                    total_marks += (q.get("marks") or 0)
                else:
                    for sq in q.get("sub_questions", []):
                        if "sub_parts" in sq:
                            for gq in sq.get("sub_parts", []):
                                total_marks += (gq.get("marks") or 0)
                        else:
                            total_marks += (sq.get("marks") or 0)
            else:
                total_marks += (q.get("marks") or 0)
                
    md.append(f"**Total Marks:** {total_marks}")
    md.append("")
    
    spec_trees = exam_data.get("spec_trees", {})
    if spec_trees:
        md.append("## Topic & Specification Breakdown")
        md.append("")
        for topic, tree_data in spec_trees.items():
            if isinstance(tree_data, dict):
                spec_code = tree_data.get("spec_code", "")
                subtopic_names = tree_data.get("subtopics", [])
                if not subtopic_names and "subtopics" not in tree_data:
                    subtopic_names = [k for k in tree_data.keys() if not k.startswith("_")]
            elif isinstance(tree_data, list):
                spec_code = ""
                subtopic_names = tree_data
            else:
                spec_code = ""
                subtopic_names = []

            code_str = f" (Specification Code: {spec_code})" if spec_code else ""
            md.append(f"### Topic: {topic}{code_str}")
            for st_name in subtopic_names:
                md.append(f"- **Subtopic:** {st_name}")
            md.append("")
        md.append("---")
        md.append("")

    for topic, q_list in questions_by_topic.items():
        md.append(f"## Topic: {topic}")
        md.append("")
        for q in q_list:
            q_num = q.get("number", "")
            subtopic = q.get("subtopic", "")
            matched_subtopic = q.get("matched_subtopic", "")
            similarity_score = q.get("similarity_score")
            
            subtopic_info = []
            if subtopic:
                subtopic_info.append(f"* **Target Subtopic:** {subtopic}")
            if matched_subtopic:
                subtopic_info.append(f"* **Matched Question Subtopic:** {matched_subtopic}")
            if similarity_score is not None:
                sim_pct = f"{similarity_score * 100:.1f}%" if isinstance(similarity_score, float) and similarity_score <= 1.0 else f"{similarity_score}"
                subtopic_info.append(f"* **Semantic Similarity:** {sim_pct}")

            subtopic_block = "\n".join(subtopic_info) if subtopic_info else ""
            
            # Check if it's a parent question with sub-parts
            if "sub_questions" in q:
                parent_desc = q.get("parent_description", "")
                marks_str = _safe_marks_str(q.get("marks"))
                md.append(f"### Question {q_num}{marks_str}")
                if subtopic_block:
                    md.append(subtopic_block)
                    md.append("")
                if parent_desc:
                    md.append(parent_desc)
                    md.append("")
                
                for sq in q.get("sub_questions", []):
                    label = sq.get("label", "")
                    sq_subtopic = sq.get("subtopic", "")
                    sq_subtopic_str = f" _(Subtopic: {sq_subtopic})_" if sq_subtopic and sq_subtopic != matched_subtopic else ""
                    
                    if "sub_parts" in sq:
                        context = sq.get("context", "")
                        if context:
                            md.append(f"**{label}**{sq_subtopic_str} {context}")
                            md.append("")
                        else:
                            md.append(f"**{label}**{sq_subtopic_str}")
                            md.append("")
                        for gq in sq.get("sub_parts", []):
                            g_label = gq.get("label", "")
                            g_text = gq.get("text", "")
                            g_marks = gq.get("marks")
                            md.append(f"  * **{g_label}** {g_text}{_safe_marks_str(g_marks)}")
                            md.append("")
                            md.append(generate_answer_lines(g_marks))
                            md.append("")
                    else:
                        sq_text = sq.get("text", "")
                        sq_marks = sq.get("marks")
                        md.append(f"* **{label}**{sq_subtopic_str} {sq_text}{_safe_marks_str(sq_marks)}")
                        md.append("")
                        md.append(generate_answer_lines(sq_marks))
                        md.append("")
            else:
                # Standalone question
                q_text = q.get("text", "")
                q_marks = q.get("marks")
                md.append(f"### Question {q_num}{_safe_marks_str(q_marks)}")
                if subtopic_block:
                    md.append(subtopic_block)
                    md.append("")
                md.append(q_text)
                md.append("")
                md.append(generate_answer_lines(q_marks))
                md.append("")
                
    return "\n".join(md)


def save_marking_report_md(
    subject: str,
    examiner: str,
    topic: str,
    marking_results: List[Dict[str, Any]],
    output_dir: str = "test_outputs",
    llm_client: Optional[LLMClient] = None
) -> str:
    """Format and save exam marking results into markdown files.

    Args:
        subject: Subject name.
        examiner: Exam board name.
        topic: Topic or exam tier name.
        marking_results: List of dicts returned by ExamMarker.mark_exam.
        output_dir: Directory to save the markdown file.
        llm_client: Optional LLMClient instance for generating performance summary.

    Returns:
        Path to the saved markdown file.
    """
    import time
    os.makedirs(output_dir, exist_ok=True)
    report_md = format_marking_as_markdown(subject, examiner, topic, marking_results, llm_client=llm_client)

    # Save to canonical fixed filepath for easy access
    canonical_path = os.path.join(output_dir, "marking_report.md")
    with open(canonical_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Save to timestamped file for archival
    timestamp = int(time.time())
    archive_path = os.path.join(output_dir, f"{subject}_{examiner}_marking_{timestamp}.md")
    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("Saved marking markdown report to %s and %s", canonical_path, archive_path)
    print(f"\033[92m  [Report Saved]: {canonical_path}\033[0m")
    return canonical_path


def generate_performance_summary(
    subject: str,
    examiner: str,
    topic: str,
    marking_results: List[Dict[str, Any]],
    llm_client: Optional[LLMClient] = None
) -> str:
    """Generate an executive performance summary (What Went Well & Areas for Improvement) via LLM call.

    Args:
        subject: Subject name.
        examiner: Exam board.
        topic: Paper/Topic.
        marking_results: Full list of marking items.
        llm_client: LLMClient instance.

    Returns:
        Markdown string with performance summary.
    """
    if not llm_client:
        try:
            llm_client = LLMClient()
        except Exception:
            return ""

    digest_lines = []
    for idx, item in enumerate(marking_results, start=1):
        label = item.get("label") or f"Question {idx}"
        q_text = (item.get("question") or "").replace("\n", " ")[:150]
        ans = (item.get("student_answer") or "No answer provided").replace("\n", " ")[:200]
        marks = item.get("marks", 1)
        awarded = item.get("awarded_marks", 0)
        feedback = (item.get("result") or "").replace("\n", " ")[:200]
        digest_lines.append(f"- [{label}] Marks: {awarded}/{marks} | Q: {q_text} | Student Answer: {ans} | Feedback: {feedback}")

    results_digest = "\n".join(digest_lines)

    prompt_path = os.path.join("prompts", "generate_performance_summary.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        prompt_template = (
            "Review these GCSE {subject} ({examiner}) exam marking results:\n{results_digest}\n\n"
            "Provide two markdown sections: ### 🌟 What Went Well and ### 🎯 Areas for Improvement."
        )

    prompt = prompt_template.format(
        subject=subject,
        examiner=examiner,
        topic=topic,
        results_digest=results_digest
    )

    print("  [LLM Performance Analysis Call]: Analyzing exam report to generate executive summary...")
    try:
        summary_text = llm_client.invoke(prompt)
        return summary_text.strip()
    except Exception as e:
        logger.warning("Failed to generate performance summary via LLM: %s", e)
        return ""


def format_marking_as_markdown(
    subject: str,
    examiner: str,
    topic: str,
    marking_results: List[Dict[str, Any]],
    llm_client: Optional[LLMClient] = None
) -> str:
    """Format exam marking results into a rich, structured markdown document.

    Args:
        subject: Subject name.
        examiner: Exam board name.
        topic: Topic or exam tier name.
        marking_results: List of dicts returned by ExamMarker.mark_exam.
        llm_client: Optional LLMClient instance to generate performance summary.

    Returns:
        Formatted markdown string.
    """
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# 📝 GCSE {subject} ({examiner}) - {topic} Marking Report")
    md.append("")
    md.append(f"**Date:** `{now_str}` | **Subject:** `{subject}` | **Exam Board:** `{examiner}` | **Paper/Topic:** `{topic}`")
    md.append("")

    total_marks_available = 0
    total_marks_awarded = 0
    has_scores = False

    for item in marking_results:
        m_avail = item.get("marks", 0)
        m_award = item.get("awarded_marks")
        if isinstance(m_avail, int):
            total_marks_available += m_avail
        if isinstance(m_award, int):
            total_marks_awarded += m_award
            has_scores = True

    if has_scores and total_marks_available > 0:
        pct = (total_marks_awarded / total_marks_available) * 100
        
        # Estimate performance band
        if pct >= 80:
            performance = "Grade 9 / Exceptional Performance 🌟"
        elif pct >= 70:
            performance = "Grade 8 / Strong Performance 🚀"
        elif pct >= 60:
            performance = "Grade 7 / Good Performance 👍"
        elif pct >= 50:
            performance = "Grade 6 / Solid Pass ✅"
        elif pct >= 40:
            performance = "Grade 4-5 / Standard Pass 📈"
        else:
            performance = "Grade 1-3 / Revision Required ⚠️"

        md.append("## 📊 Executive Score Summary")
        md.append("")
        md.append("| Total Available Marks | Total Awarded Marks | Overall Percentage | Estimated Performance |")
        md.append("| :---: | :---: | :---: | :---: |")
        md.append(f"| **{total_marks_available}** | **{total_marks_awarded}** | **{pct:.1f}%** | **{performance}** |")
        md.append("")
        
        md.append("> [!TIP]")
        md.append(f"> **Performance Summary**: Achieved **{total_marks_awarded}/{total_marks_available}** marks ({pct:.1f}%). Focus on reviewing questions where partial credit was awarded to secure remaining marks.")
        md.append("")

        # Generate Executive Performance Analysis via LLM call
        if llm_client:
            summary = generate_performance_summary(subject, examiner, topic, marking_results, llm_client)
            if summary:
                md.append("## 💡 Executive Performance Analysis")
                md.append("")
                md.append(summary)
                md.append("")

        md.append("---")
        md.append("")

    md.append("## 📄 Question Feedback & Marking Detail")
    md.append("")

    for idx, item in enumerate(marking_results, start=1):
        label = item.get("label") or f"Question {idx}"
        q_title = item.get("question", "").strip()
        student_ans = item.get("student_answer", "").strip()
        ms = item.get("mark_scheme", "").strip()
        res = item.get("result", "").strip()
        marks = item.get("marks", "")
        awarded = item.get("awarded_marks")

        marks_badge = f"({awarded}/{marks} Marks)" if (awarded is not None and marks) else (f"({marks} Marks)" if marks else "")

        md.append(f"### ❓ {label} {marks_badge}")
        md.append("")
        
        if q_title:
            md.append("#### Question Prompt")
            md.append(f"```text\n{q_title}\n```")
            md.append("")

        md.append("#### Student Answer")
        if student_ans:
            md.append(f"> {student_ans.replace(chr(10), chr(10) + '> ')}")
        else:
            md.append("*No student answer provided.*")
        md.append("")

        ms_source = item.get("mark_scheme_source") or "Official Mark Scheme (user_data)"
        ans_fallback = item.get("answer_fallback_used", False)
        ms_fallback = item.get("ms_fallback_used", False)

        if ms:
            md.append(f"#### 📋 Mark Scheme (Source: `{ms_source}`)")
            md.append(f"```text\n{ms}\n```")
            md.append("")

        if ans_fallback or ms_fallback:
            md.append("> [!WARNING]")
            if ans_fallback:
                md.append("> **LLM Fallback Search Triggered**: Exact canonical code match failed for student answer; used LLM context search to retrieve answer.")
            if ms_fallback:
                md.append("> **LLM Fallback Search Triggered**: Exact canonical code match failed for mark scheme; used LLM context search to retrieve mark scheme.")
            md.append("")

        md.append("#### Examiner Feedback & Score")
        md.append("> [!NOTE]")
        if res:
            res_lines = res.splitlines()
            for line in res_lines:
                md.append(f"> {line}")
        else:
            md.append("> *No feedback available.*")
        md.append("")
        md.append("---")
        md.append("")

    return "\n".join(md)


if __name__ == "__main__":
    import sys
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="GCSE AI Exam Generator and Marking Assistant")
    parser.add_argument("--subject", type=str, default="Physics", help="Subject name (e.g. Physics, ReligiousStudies, Biology)")
    parser.add_argument("--examiner", type=str, default="Edexcel", help="Exam board (e.g. Edexcel, AQA)")
    args = parser.parse_args()

    subject = args.subject
    examiner = args.examiner

    print("\n==================================================")
    print(f"INITIALIZING GCSE ASSISTANT FOR {subject.upper()} ({examiner.upper()})")
    print("==================================================")
    assistant = GcseAssistant(subject, examiner)
    
    print("\nMarking files in user_data/ directory...")
    feedback_str = assistant.mark_question()
    
    if feedback_str:
        os.makedirs("test_outputs", exist_ok=True)
        report_file = "test_outputs/marking_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(feedback_str)
        print(f"\n[SUCCESS] Generated Marking Feedback Report saved to:\n  -> {report_file}\n")
    else:
        print("\n[INFO] Required question/answer files in user_data/ not found.")



