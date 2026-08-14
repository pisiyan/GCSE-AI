"""GCSE AI Interactive Chatbot Script.

Provides a terminal-based conversational interface to chat with the GCSE AI
assistant, search specification databases (RAG), generate exams, mark answers,
and save custom preferences.
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv

# Load env variables relative to the script location first with override=True
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"), override=True)
if "OPENAI_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()

# Ensure logs don't clutter the stdout during interactive chat
logging.getLogger().setLevel(logging.WARNING)

from generate_content import GcseAssistant, format_exam_as_markdown, format_marking_as_markdown, format_single_question_as_markdown

# ANSI Color Escape Codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def print_header(msg: str):
        print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")

    @staticmethod
    def print_blue(msg: str):
        print(f"{Colors.BLUE}{msg}{Colors.ENDC}")

    @staticmethod
    def print_cyan(msg: str):
        print(f"{Colors.CYAN}{msg}{Colors.ENDC}")

    @staticmethod
    def print_green(msg: str):
        print(f"{Colors.GREEN}{msg}{Colors.ENDC}")

    @staticmethod
    def print_warning(msg: str):
        print(f"{Colors.WARNING}{msg}{Colors.ENDC}")

    @staticmethod
    def print_fail(msg: str):
        print(f"{Colors.FAIL}{msg}{Colors.ENDC}")


# Diagnostic check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    Colors.print_fail("[Diagnostic] OPENAI_API_KEY is not set in environment!")
else:
    redacted = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 10 else "too short"
    Colors.print_cyan(f"[Diagnostic] Loaded API Key: {redacted} (length={len(api_key)})")

# Print other env variables that could override or redirect the API calls
env_diagnostics = {k: v for k, v in os.environ.items() if any(x in k.upper() for x in ["OPENAI", "PROXY", "BASE", "URL"])}
for k, v in env_diagnostics.items():
    if k != "OPENAI_API_KEY":  # already printed safely
        Colors.print_cyan(f"[Diagnostic] Env Var: {k} = {v}")


def scan_available_subjects() -> List[Tuple[str, str]]:
    """Scan the data directory for existing FAISS databases."""
    available = []
    data_dir = "data"
    if not os.path.exists(data_dir):
        return [("Physics", "Edexcel"), ("Biology", "Edexcel"), ("ReligiousStudies", "AQA")]
    
    for subj in os.listdir(data_dir):
        subj_path = os.path.join(data_dir, subj)
        if os.path.isdir(subj_path):
            for exam in os.listdir(subj_path):
                exam_path = os.path.join(subj_path, exam)
                if os.path.isdir(exam_path):
                    # Check if vector database exists inside
                    vdb_dir = os.path.join(exam_path, f"{subj}-{exam}-vectorDatabase")
                    if os.path.exists(vdb_dir):
                        available.append((subj, exam))
    
    # Fallback to defaults if none found
    if not available:
        return [("Physics", "Edexcel"), ("Biology", "Edexcel"), ("ReligiousStudies", "AQA")]
    return list(set(available))


class ChatbotAgent:
    """Manages the agent prompt, history, preferences, and action executions."""

    def __init__(self, subject: str, examiner: str):
        self.subject = subject
        self.examiner = examiner
        Colors.print_cyan(f"\nInitializing GCSE Assistant for {subject} ({examiner}). Please wait...")
        
        t0 = time.time()
        self.assistant = GcseAssistant(subject, examiner)
        Colors.print_green(f"Successfully loaded database in {time.time() - t0:.1f}s.")

        # Discover valid exam types for this subject/examiner from the data directory
        self.valid_exam_types: List[str] = GcseAssistant.get_valid_exam_types(subject, examiner)
        if self.valid_exam_types:
            Colors.print_cyan(f"Valid exam types for {subject} ({examiner}): {', '.join(self.valid_exam_types)}")
        else:
            Colors.print_warning(f"No Exam-Types directory found for {subject} ({examiner}). Exam type validation will be skipped.")
        
        self.history: List[Dict[str, str]] = []
        self.preferences: List[str] = []

    def get_preferences_text(self) -> str:
        if not self.preferences:
            return "No custom preferences set yet."
        return "\n".join(f"- {pref}" for pref in self.preferences)

    def run_agent_loop(self, user_msg: str):
        """Run the main conversational agent loop."""
        # 1. Add user message to history
        self.history.append({"role": "user", "content": user_msg})

        # Keep history length within reasonable bounds
        trimmed_history = self.history[-12:]

        # 2. Construct Prompt
        # Build exam-type constraint section for the prompt
        if self.valid_exam_types:
            exam_type_constraint = (
                f"Valid exam types for this subject: {', '.join(self.valid_exam_types)}.\n"
                f"You MUST use one of these exact strings for the \"exam_type\" parameter."
            )
        else:
            exam_type_constraint = "No exam type list available — ask the user to specify the exam type."

        system_instruction = f"""You are the GCSE AI Chatbot Assistant for {self.subject} ({self.examiner}).
Your goal is to help the user study, answer questions about syllabus/specification content, mark answers, and generate exams.

Current Active Subject: {self.subject} ({self.examiner})
User Preferences & Custom Constraints:
{self.get_preferences_text()}

Official GCSE Syllabus/Specification context for {self.subject} ({self.examiner}):
{self.assistant.specification_text}

You must respond strictly in JSON format matching the schema below.
If you need to perform an action, specify the action and its arguments in "action" and "params".
For complex queries requesting multiple granular tasks together, specify the list of action objects in "actions".
If you need to ask a question, tell the user something, or answer directly, specify action "none" and write your response in the "message" field.

AVAILABLE ACTIONS:
1. "none":
   Use this for regular conversation, explaining content using the specification context above, or when asking for missing details.
   Params: None.
2. "query_past_papers":
   Use this when searching past exam papers, questions, or mark schemes in the database.
   Params: `query` (str)
3. "generate_exam":
   Use this ONLY when the user requests generating a new full exam paper.
   REQUIRED PARAMETERS: `exam_type` (e.g. {self.valid_exam_types[0] if self.valid_exam_types else 'Higher'}), `total_marks` (int), and `topics` (list of strings).
   OPTIONAL PARAMETER: `num_questions` (int, optional) - Number of questions in the exam.
   {exam_type_constraint}
   Params: `exam_type` (str), `total_marks` (int), `topics` (list of str), `num_questions` (int or null, optional)
4. "mark_answer":
   Use this ONLY when the user wants to mark a student's answer.
   VALIDATION REQUIREMENT: Must provide all 4 parameters: `question` (str), `marks` (int), `student_answer` (str), and `mark_scheme` (str).
   Params: `question` (str), `marks` (int), `student_answer` (str), `mark_scheme` (str)
5. "revision_materials":
   Use this when the user wants structured revision notes/materials for a topic.
   Params: `topic` (str)
6. "save_preferences":
   Use this when saving custom preferences or notes for future exams.
   Params: `preferences` (str)
7. "topic_test":
   Use this when starting an interactive multi-question topic test session.
   Params: `desired_area` (str)
8. "generate_single_question":
   Use this when the user requests generating a single standalone or parent question.
   Params: `topic` (str, required), `marks` (int, optional, default 4), `subtopic` (str, optional), `q_type` ("basic" or "parent", optional)
9. "generate_mark_scheme":
   Use this when the user requests a mark scheme for a single question or topic.
   Params: `question` (str, optional), `marks` (int, optional, default 4), `topic` (str, optional)
10. "generate_model_answer":
    Use this when the user requests a full-mark model/exemplar answer for a question.
    Params: `question` (str, required), `mark_scheme` (str, optional), `marks` (int, optional), `topic` (str, optional)
11. "get_spec_breakdown":
    Use this when the user asks for specification code(s), topic breakdown, or subtopics list.
    Params: `topic` (str, optional)
12. "explain_command_word":
    Use this when the user asks to explain a GCSE command word (e.g. Describe, Explain, Evaluate) or identify/explain examiner criteria for a question's command word.
    Params: `command_word` (str, optional), `question` (str, optional)

RESPONSE SCHEMA:
{{
  "thought": "Your internal thinking process about what the user wants and parameter validation check.",
  "action": "none" | "query_past_papers" | "generate_exam" | "mark_answer" | "revision_materials" | "save_preferences" | "topic_test" | "generate_single_question" | "generate_mark_scheme" | "generate_model_answer" | "get_spec_breakdown" | "explain_command_word",
  "actions": [
    {{ "action": "generate_single_question", "params": {{ "topic": "Photosynthesis", "marks": 4 }} }},
    {{ "action": "generate_mark_scheme", "params": {{ "topic": "Photosynthesis", "marks": 4 }} }}
  ],
  "params": {{ ... }},
  "message": "Your conversational reply to the user (required if action is 'none', otherwise a brief placeholder)."
}}

COMPLEX QUERIES & MULTI-ACTION RULES:
- For standard single requests, set "action" to the single action name and "params" to its parameters.
- ONLY when the user's prompt explicitly asks for MULTIPLE granular tasks (from generate_single_question, generate_mark_scheme, generate_model_answer, get_spec_breakdown, explain_command_word) in a single query (e.g. asking for a question AND a mark scheme AND a model answer together), provide the list of action objects in the "actions" field.
- CRITICAL ISOLATION RULE: If the user requests generating a full exam ("generate_exam") or marking an answer ("mark_answer"), you MUST ONLY set action to "generate_exam" or "mark_answer". Do NOT include any secondary actions in "actions". Full exam generation and answer marking are isolated standalone operations and must never be combined with single-item actions.
"""

        # Format conversation history
        history_lines = []
        for h in trimmed_history:
            role = "User" if h["role"] == "user" else "Assistant"
            content = h["content"]
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        full_prompt = f"{system_instruction}\n\n--- Conversation History ---\n{history_text}\n\nAssistant Response (JSON):"

        try:
            res_dict = self.assistant.llm_client.invoke_json(full_prompt)
        except Exception as e:
            Colors.print_fail(f"Error communicating with LLM: {e}")
            return

        thought = res_dict.get("thought", "")
        primary_action = res_dict.get("action", "none")
        raw_actions = res_dict.get("actions", [])
        params = res_dict.get("params", {})
        message = res_dict.get("message", "")

        # Strict isolation: generate_exam and mark_answer can NEVER run with secondary actions
        if primary_action in ("generate_exam", "mark_answer"):
            actions_list = [{"action": primary_action, "params": params}]
        elif isinstance(raw_actions, list) and len(raw_actions) > 0:
            valid_actions = [
                act for act in raw_actions
                if isinstance(act, dict) and act.get("action") not in ("generate_exam", "mark_answer")
            ]
            actions_list = valid_actions if valid_actions else [{"action": primary_action, "params": params}]
        else:
            actions_list = [{"action": primary_action, "params": params}]

        context_data = {}
        executed_any = False

        for act_item in actions_list:
            action = act_item.get("action", "none")
            act_params = act_item.get("params", {})
            if not isinstance(act_params, dict):
                act_params = {}

            if action == "none":
                Colors.print_green(f"\n{message}")
                self.history.append({"role": "assistant", "content": message})

            elif action == "query_past_papers":
                query = act_params.get("query", "")
                Colors.print_blue(f"\n[Database Search] Searching past papers & mark schemes for: '{query}'...")
                try:
                    rag_result = self.assistant.llm_client.invoke_qa(self.assistant.ms_qa_chain, query)
                    obs_content = f"Database search results for query '{query}':\n{rag_result}"
                    self.history.append({"role": "system", "content": obs_content})
                    executed_any = True
                except Exception as e:
                    self.handle_action_error("query_past_papers", e)

            elif action == "generate_exam":
                exam_type = act_params.get("exam_type")
                total_marks = act_params.get("total_marks")
                topics = act_params.get("topics")
                num_questions = act_params.get("num_questions")

                if num_questions is not None:
                    try:
                        num_questions = int(num_questions)
                    except (ValueError, TypeError):
                        num_questions = None
                
                # Validation: required fields
                missing = []
                if not exam_type: missing.append("exam_type")
                if not total_marks: missing.append("total_marks")
                if not topics: missing.append("topics")
                
                if missing:
                    err_msg = f"Validation Error: Cannot generate exam. Missing required parameter(s): {', '.join(missing)}."
                    Colors.print_fail(f"\n{err_msg}")
                    self.history.append({"role": "system", "content": err_msg})
                    self.run_agent_loop_followup()
                    return

                # Validation: exam type must be in the known-valid list
                if self.valid_exam_types and exam_type not in self.valid_exam_types:
                    err_msg = (
                        f"Validation Error: '{exam_type}' is not a valid exam type for "
                        f"{self.subject} ({self.examiner}). "
                        f"Valid exam types are: {', '.join(self.valid_exam_types)}."
                    )
                    Colors.print_fail(f"\n{err_msg}")
                    self.history.append({"role": "system", "content": err_msg})
                    self.run_agent_loop_followup()
                    return

                try:
                    total_marks = int(total_marks)
                except (ValueError, TypeError):
                    err_msg = "Validation Error: 'total_marks' must be a valid integer."
                    Colors.print_fail(f"\n{err_msg}")
                    self.history.append({"role": "system", "content": err_msg})
                    self.run_agent_loop_followup()
                    return

                if not isinstance(topics, list) or len(topics) == 0:
                    err_msg = "Validation Error: 'topics' must be a non-empty list of strings."
                    Colors.print_fail(f"\n{err_msg}")
                    self.history.append({"role": "system", "content": err_msg})
                    self.run_agent_loop_followup()
                    return

                Colors.print_blue(f"\n[Exam Generator] Triggering parallel exam generation...")
                Colors.print_blue(f"  - Exam Type: {exam_type}")
                Colors.print_blue(f"  - Total Marks: {total_marks}")
                Colors.print_blue(f"  - Topic Areas: {topics}")
                if num_questions:
                    Colors.print_blue(f"  - Requested Question Count: {num_questions}")
                
                user_prefs = act_params.get("user_preferences", {})
                if isinstance(user_prefs, str):
                    user_prefs = {"custom_instructions": user_prefs}
                elif not isinstance(user_prefs, dict):
                    user_prefs = {}
                if self.preferences:
                    user_prefs["stored_preferences"] = self.preferences

                try:
                    exam_data = self.assistant.make_exam(
                        exam_type, total_marks, topics,
                        user_preferences=user_prefs,
                        num_questions=num_questions
                    )
                    exam_md = format_exam_as_markdown(self.subject, self.examiner, exam_type, exam_data)
                    
                    os.makedirs("test_outputs", exist_ok=True)
                    timestamp = int(time.time())
                    filename = f"test_outputs/{self.subject}_{self.examiner}_exam_{timestamp}.md"
                    with open(filename, "w", encoding="utf-8") as f:
                         f.write(exam_md)
                    Colors.print_green(f"  -> Generated exam saved to: {filename}")

                    spec_trees = exam_data.get("spec_trees", {})
                    if spec_trees:
                        Colors.print_blue("\n[Specification Codes & Subtopics Extracted/Used]:")
                        for top, tree_data in spec_trees.items():
                            if isinstance(tree_data, dict):
                                spec_code = tree_data.get("spec_code", "")
                                subtopic_names = tree_data.get("subtopics", [])
                                if not subtopic_names and "subtopics" not in tree_data:
                                    subtopic_names = [k for k in tree_data.keys() if not k.startswith("_")]
                            else:
                                spec_code = ""
                                subtopic_names = tree_data if isinstance(tree_data, list) else []

                            code_str = f" [{spec_code}]" if spec_code else ""
                            Colors.print_cyan(f"  Topic: {top}{code_str}")
                            for st_name in subtopic_names:
                                Colors.print_cyan(f"    • Subtopic: {st_name}")

                    obs_content = (
                        f"Exam generated successfully and saved to file '{filename}'. "
                        f"Specification subtopics breakdown included in file and outputted during generation."
                    )
                    self.history.append({"role": "system", "content": obs_content})
                    executed_any = True
                except Exception as e:
                    self.handle_action_error("generate_exam", e)

            elif action == "mark_answer":
                question = act_params.get("question")
                marks = act_params.get("marks")
                student_answer = act_params.get("student_answer")
                mark_scheme = act_params.get("mark_scheme")

                missing = []
                if not question: missing.append("question")
                if not marks: missing.append("marks")
                if not student_answer: missing.append("student_answer")
                if not mark_scheme: missing.append("mark_scheme")

                if missing:
                    err_msg = f"Validation Error: Cannot mark answer. Missing required parameter(s): {', '.join(missing)}."
                    Colors.print_fail(f"\n{err_msg}")
                    self.history.append({"role": "system", "content": err_msg})
                    self.run_agent_loop_followup()
                    return

                try:
                    marks = int(marks)
                except (ValueError, TypeError):
                    err_msg = "Validation Error: 'marks' must be a valid integer."
                    Colors.print_fail(f"\n{err_msg}")
                    self.history.append({"role": "system", "content": err_msg})
                    self.run_agent_loop_followup()
                    return
                
                Colors.print_blue(f"\n[Marker] Grading answer against mark scheme...")
                try:
                    feedback = self.assistant.exam_marker.mark_answer(student_answer, mark_scheme, question, marks)
                    
                    os.makedirs("test_outputs", exist_ok=True)
                    timestamp = int(time.time())
                    filename = f"test_outputs/{self.subject}_{self.examiner}_feedback_{timestamp}.md"
                    
                    single_res = [{
                        "question": question,
                        "marks": marks,
                        "student_answer": student_answer,
                        "mark_scheme": mark_scheme,
                        "result": feedback
                    }]
                    report_md = format_marking_as_markdown(self.subject, self.examiner, "Single Answer", single_res)
                    
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(report_md)
                    Colors.print_green(f"  -> Marking report saved to: {filename}")

                    obs_content = (
                        f"Answer marked. Feedback saved to '{filename}'. "
                        f"Awarded score detail: {feedback}."
                    )
                    self.history.append({"role": "system", "content": obs_content})
                    executed_any = True
                except Exception as e:
                    self.handle_action_error("mark_answer", e)

            elif action == "revision_materials":
                topic = act_params.get("topic")
                Colors.print_blue(f"\n[Revision] Generating revision materials for '{topic}'...")
                try:
                    revision_text = self.assistant.revision_materials(topic)
                    obs_content = f"Revision materials for '{topic}':\n{revision_text}"
                    self.history.append({"role": "system", "content": obs_content})
                    executed_any = True
                except Exception as e:
                    self.handle_action_error("revision_materials", e)

            elif action == "save_preferences":
                pref = act_params.get("preferences")
                Colors.print_blue(f"\n[Preferences] Saving custom instruction: '{pref}'")
                self.preferences.append(pref)
                obs_content = f"Instruction saved: '{pref}'"
                self.history.append({"role": "system", "content": obs_content})
                executed_any = True

            elif action == "topic_test":
                desired_area = act_params.get("desired_area") or act_params.get("topic") or "General"
                Colors.print_header("\n==================================================")
                Colors.print_header(f"  STARTING INTERACTIVE TOPIC TEST: '{desired_area}'")
                Colors.print_header("==================================================")
                Colors.print_blue("\n[1/2] Compiling target subtopics from specification summary (1 LLM call)...")
                Colors.print_blue("[2/2] Generating content questions & exemplar answers (1 LLM call)...")

                try:
                    compiled_scope, qa_list = self.assistant.create_topic_test_session(desired_area)
                    scope_desc = compiled_scope.get("scope_description", desired_area)
                    Colors.print_cyan(f"\n[Target Scope]: {scope_desc}")
                    Colors.print_cyan(f"Prepared {len(qa_list)} content questions for testing.\n")

                    test_results = []

                    for idx, item in enumerate(qa_list, start=1):
                        subtopic = item.get("subtopic", f"Question {idx}")
                        q_text = item.get("question", "")
                        perf_ans = item.get("perfect_answer", "")

                        Colors.print_header(f"\n--- Question {idx} of {len(qa_list)} [{subtopic}] ---")
                        Colors.print_header(q_text)
                        print()

                        try:
                            student_ans = input(f"{Colors.BOLD}Your Answer (or type 'skip'/'quit'): {Colors.ENDC}").strip()
                        except (EOFError, KeyboardInterrupt):
                            Colors.print_warning("\nTesting session interrupted.")
                            break

                        if student_ans.lower() in ("quit", "exit", "stop"):
                            Colors.print_warning("Testing session ended early by student.")
                            break

                        Colors.print_blue("\n[Evaluating Answer against Exemplar...]")
                        eval_result = self.assistant.evaluate_topic_test_answer(
                            question=q_text,
                            perfect_answer=perf_ans,
                            student_answer=student_ans,
                            subtopic=subtopic
                        )

                        score = eval_result.get("score", 0)
                        max_score = eval_result.get("max_score", 5)
                        feedback = eval_result.get("feedback", "")
                        corrects = eval_result.get("correct_points", [])
                        missings = eval_result.get("missing_points", [])

                        Colors.print_header(f"\n==================================================")
                        Colors.print_header(f"  EVALUATION FEEDBACK (Score: {score}/{max_score})")
                        Colors.print_header(f"==================================================")

                        if corrects:
                            Colors.print_green("✅ What You Got Right:")
                            for c in corrects:
                                Colors.print_green(f"  • {c}")

                        if missings:
                            Colors.print_warning("\n⚠️ Key Points Missed / To Revise:")
                            for m in missings:
                                Colors.print_warning(f"  • {m}")

                        if feedback:
                            Colors.print_blue("\n💡 Feedback & Guidance:")
                            for line in feedback.splitlines():
                                if line.strip():
                                    print(f"  {line.strip()}")

                        Colors.print_cyan(f"\n📋 Exemplar Perfect Answer:\n{perf_ans}\n")

                        test_results.append({
                            "subtopic": subtopic,
                            "question": q_text,
                            "perfect_answer": perf_ans,
                            "student_answer": student_ans,
                            "evaluation": eval_result
                        })

                        while True:
                            try:
                                followup = input(f"{Colors.WARNING}Do you have any questions about this topic/feedback? (Type your question, or press Enter to move to next question): {Colors.ENDC}").strip()
                            except (EOFError, KeyboardInterrupt):
                                break

                            if not followup or followup.lower() in ("no", "n", "next", "continue", "skip", "none"):
                                break

                            q_prompt = (
                                f"The student is taking an interactive test on GCSE {self.subject} ({self.examiner}) "
                                f"topic '{subtopic}'.\n"
                                f"Question: {q_text}\n"
                                f"Student's Answer: {student_ans}\n"
                                f"Feedback: {feedback}\n\n"
                                f"The student asks a follow-up question: '{followup}'\n\n"
                                f"Provide a helpful, concise, and educational response answering their question. "
                                f"IMPORTANT FORMATTING RULE: Format your response clearly using bullet points (•) and bold text for key terms."
                            )
                            try:
                                reply = self.assistant.llm_client.invoke(q_prompt)
                                Colors.print_green(f"\n[Tutor Explanation]:\n{reply}\n")
                            except Exception as ex:
                                Colors.print_warning(f"Could not answer follow-up question: {ex}")
                                break

                    if test_results:
                        Colors.print_blue("\nCompiling final performance report...")
                        report_path = self.assistant.save_topic_test_report(desired_area, test_results)
                        Colors.print_green(f"\n[Test Complete] Report saved to: {report_path}")

                        obs_content = f"Interactive topic test completed for '{desired_area}'. Performance report saved to '{report_path}'."
                        self.history.append({"role": "system", "content": obs_content})
                        executed_any = True

                except Exception as e:
                    self.handle_action_error("topic_test", e)

            elif action == "generate_single_question":
                topic = act_params.get("topic") or context_data.get("last_topic") or self.subject
                marks = act_params.get("marks", 4)
                try:
                    marks = int(marks)
                except (ValueError, TypeError):
                    marks = 4
                subtopic = act_params.get("subtopic", "")
                q_type = act_params.get("q_type", "basic")

                Colors.print_blue(f"\n[Single Question Generator] Generating {marks}-mark {q_type} question for '{topic}'...")
                try:
                    q_data = self.assistant.generate_single_question(
                        topic=topic,
                        marks=marks,
                        subtopic=subtopic,
                        q_type=q_type,
                        exam_type=self.examiner
                    )
                    q_md = format_single_question_as_markdown(self.subject, self.examiner, q_data)

                    q_text = q_data.get("text") or q_data.get("parent_description") or ""
                    context_data["last_question"] = q_text
                    context_data["last_topic"] = topic
                    context_data["last_marks"] = marks

                    obs_content = f"Single question generated:\n{q_md}"
                    self.history.append({"role": "system", "content": obs_content})
                    executed_any = True
                except Exception as e:
                    self.handle_action_error("generate_single_question", e)

            elif action == "generate_mark_scheme":
                question = act_params.get("question") or context_data.get("last_question")
                topic = act_params.get("topic") or context_data.get("last_topic") or self.subject
                marks = act_params.get("marks") or context_data.get("last_marks") or 4
                try:
                    marks = int(marks)
                except (ValueError, TypeError):
                    marks = 4

                if not question:
                    q_data = self.assistant.generate_single_question(topic=topic, marks=marks, exam_type=self.examiner)
                    question = q_data.get("text") or q_data.get("parent_description") or f"Explain key concepts of {topic}."
                    context_data["last_question"] = question

                Colors.print_blue(f"\n[Mark Scheme Generator] Creating mark scheme ({marks} marks) for topic '{topic}'...")
                try:
                    ms_text = self.assistant.generate_mark_scheme(question=question, marks=marks, topic=topic)

                    context_data["last_mark_scheme"] = ms_text
                    context_data["last_topic"] = topic
                    context_data["last_marks"] = marks

                    obs_content = f"Mark scheme generated for question '{question}' ({marks} marks):\n{ms_text}"
                    self.history.append({"role": "system", "content": obs_content})
                    executed_any = True
                except Exception as e:
                    self.handle_action_error("generate_mark_scheme", e)

            elif action == "generate_model_answer":
                question = act_params.get("question") or context_data.get("last_question")
                mark_scheme = act_params.get("mark_scheme") or context_data.get("last_mark_scheme") or ""
                topic = act_params.get("topic") or context_data.get("last_topic") or self.subject
                marks = act_params.get("marks") or context_data.get("last_marks") or 4
                try:
                    marks = int(marks)
                except (ValueError, TypeError):
                    marks = 4

                if not question:
                    q_data = self.assistant.generate_single_question(topic=topic, marks=marks, exam_type=self.examiner)
                    question = q_data.get("text") or q_data.get("parent_description") or f"Explain key concepts of {topic}."
                    context_data["last_question"] = question

                Colors.print_blue(f"\n[Model Answer Generator] Writing full-mark exemplar answer...")
                try:
                    ma_text = self.assistant.generate_model_answer(question=question, mark_scheme=mark_scheme, marks=marks, topic=topic)

                    context_data["last_model_answer"] = ma_text

                    obs_content = f"Model answer (full marks) generated for question '{question}':\n{ma_text}"
                    self.history.append({"role": "system", "content": obs_content})
                    executed_any = True
                except Exception as e:
                    self.handle_action_error("generate_model_answer", e)

            elif action == "get_spec_breakdown":
                topic = act_params.get("topic") or context_data.get("last_topic")
                Colors.print_blue(f"\n[Specification Breakdown] Retrieving syllabus hierarchy...")
                try:
                    bd = self.assistant.get_spec_breakdown(topic=topic)
                    md_lines = [f"# Specification Breakdown — {self.subject} ({self.examiner})"]
                    if bd.get("topic"):
                        md_lines.append(f"### Topic: {bd['topic']}")
                        md_lines.append(f"**Specification Code:** `{bd.get('spec_code', 'N/A')}`")
                        md_lines.append("**Subtopics:**")
                        for sub in bd.get("subtopics", []):
                            md_lines.append(f"  - {sub}")
                    else:
                        md_lines.append("### Full Syllabus Topics & Subtopics Summary")
                        summary = bd.get("summary", {})
                        for et in summary.get("exam_types", []):
                            md_lines.append(f"#### Tier/Type: {et.get('exam_type')}")
                            for t in et.get("topics", []):
                                code = f" [{t.get('spec_code')}]" if t.get('spec_code') else ""
                                md_lines.append(f"- **{t.get('topic')}**{code}")
                                for s in t.get("subtopics", []):
                                    sub_name = s if isinstance(s, str) else s.get("name", "")
                                    md_lines.append(f"  - {sub_name}")

                    bd_md = "\n".join(md_lines)

                    obs_content = f"Specification breakdown:\n{bd_md}"
                    self.history.append({"role": "system", "content": obs_content})
                    executed_any = True
                except Exception as e:
                    self.handle_action_error("get_spec_breakdown", e)

            elif action == "explain_command_word":
                command_word = act_params.get("command_word", "")
                question = act_params.get("question") or context_data.get("last_question") or ""
                Colors.print_blue(f"\n[Command Word Examiner Guidance] Analyzing command word requirements...")
                try:
                    cw_text = self.assistant.explain_command_word(command_word=command_word, question=question)

                    obs_content = f"Command word guidance:\n{cw_text}"
                    self.history.append({"role": "system", "content": obs_content})
                    executed_any = True
                except Exception as e:
                    self.handle_action_error("explain_command_word", e)

        if executed_any:
            self.run_agent_loop_followup()

    def run_agent_loop_followup(self):
        """Follow-up agent query after action execution completes."""
        trimmed_history = self.history[-12:]
        history_lines = []
        for h in trimmed_history:
            role = "User" if h["role"] == "user" else ("Assistant" if h["role"] == "assistant" else "System")
            content = h["content"]
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        prompt = (
            f"You are the GCSE AI Chatbot Assistant for {self.subject} ({self.examiner}).\n"
            f"A background tool/action has just completed. Respond to the user explaining the result "
            f"and providing study guidance.\n\n"
            f"--- Conversation History ---\n{history_text}\n\n"
            f"Assistant Response (explain the result or answer the user directly in markdown format):"
        )
        try:
            message = self.assistant.llm_client.invoke(prompt)
            Colors.print_green(f"\n{message}")
            self.history.append({"role": "assistant", "content": message})
        except Exception as e:
            Colors.print_fail(f"Error during follow-up response: {e}")

    def handle_action_error(self, action_name: str, exception: Exception):
        """Handle execution errors cleanly by informing the agent."""
        Colors.print_fail(f"Error running {action_name}: {exception}")
        obs_content = f"Error executing {action_name}: {str(exception)}"
        self.history.append({"role": "system", "content": obs_content})
        self.run_agent_loop_followup()


def show_welcome_banner():
    Colors.print_header("=" * 65)
    Colors.print_header("            GCSE AI CO-PILOT STUDY ASSISTANT CHATBOT            ")
    Colors.print_header("=" * 65)
    print("Welcome! I am your agentic study helper. You can:")
    print(" - Talk with me to ask questions on specification content (I use RAG)")
    print(" - Ask me to generate custom exams")
    print(" - Grade/mark student answers against a mark scheme")
    print(" - Create revision notes on topics")
    print(" - Instruct me to remember notes or preferences for exam generation")
    Colors.print_blue("\nType direct messages to chat, or use slash commands:")
    print("  /help                     Show list of direct commands")
    print("  /subject <subj> <board>   Switch active subject database")
    print("  /preferences              Show custom active rules & notes")
    print("  /clear                    Reset chat conversation history")
    print("  /exit                     Exit the program")
    Colors.print_header("=" * 65)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Scan available databases
    available_subjects = scan_available_subjects()
    
    Colors.print_header("Available Databases Found:")
    for idx, (subj, exam) in enumerate(available_subjects, start=1):
        print(f"  {idx}. {subj} ({exam})")
    
    # Pick active database
    selected_idx = 0
    while True:
        try:
            sel = input(f"\nSelect database number [1-{len(available_subjects)}] (default 1): ").strip()
            if not sel:
                selected_idx = 0
                break
            selected_idx = int(sel) - 1
            if 0 <= selected_idx < len(available_subjects):
                break
            else:
                Colors.print_warning(f"Please select a number between 1 and {len(available_subjects)}")
        except ValueError:
            Colors.print_warning("Invalid input. Please enter a valid number.")

    active_subj, active_exam = available_subjects[selected_idx]
    
    # Initialize Agent
    agent = ChatbotAgent(active_subj, active_exam)
    
    show_welcome_banner()

    # Chat Loop
    while True:
        try:
            prompt_str = f"\n{Colors.BOLD}GCSE AI ({agent.subject}-{agent.examiner}) > {Colors.ENDC}"
            user_input = input(prompt_str).strip()
            
            if not user_input:
                continue

            # Check Slash Commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()

                if cmd in ("/exit", "/quit"):
                    Colors.print_blue("\nGoodbye! Happy studying!")
                    break

                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /help                     Show list of commands")
                    print("  /subject <subj> <board>   Switch active subject database (e.g. /subject Physics Edexcel)")
                    print("  /preferences              Display active custom preferences/rules")
                    print("  /clear                    Reset current conversation history")
                    print("  /exit or /quit            Exit chatbot")
                    
                elif cmd == "/preferences":
                    print(f"\nActive preferences for {agent.subject}-{agent.examiner}:")
                    print(agent.get_preferences_text())

                elif cmd == "/clear":
                    agent.history.clear()
                    Colors.print_green("\nConversation history cleared.")

                elif cmd == "/subject":
                    if len(cmd_parts) < 3:
                        Colors.print_warning("\nUsage: /subject <subject> <examiner>  (e.g., /subject Physics Edexcel)")
                        continue
                    new_subj = cmd_parts[1]
                    new_exam = cmd_parts[2]
                    try:
                        # Try to initialize first before changing
                        agent = ChatbotAgent(new_subj, new_exam)
                        Colors.print_green(f"\nSwitched subject to {new_subj} ({new_exam})")
                    except Exception as e:
                        Colors.print_fail(f"\nFailed to load database for {new_subj}-{new_exam}: {e}")

                else:
                    Colors.print_warning(f"\nUnknown command: {cmd}. Type /help for assistance.")

            else:
                # Regular message
                agent.run_agent_loop(user_input)

        except KeyboardInterrupt:
            Colors.print_blue("\n\nSession ended by user (Ctrl+C). Goodbye!")
            break
        except Exception as e:
            Colors.print_fail(f"\nAn unexpected error occurred in chat loop: {e}")


if __name__ == "__main__":
    main()
