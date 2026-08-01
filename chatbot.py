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

from generate_content import GcseAssistant, format_exam_as_markdown

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
If you need to ask a question, tell the user something, or answer directly, specify action "none" and write your response in the "message" field.

AVAILABLE ACTIONS:
1. "none":
   Use this for regular conversation, explaining content using the specification context above, or when you need to ask the user for missing details.
   Params: None.
2. "query_past_papers":
   Use this when the user asks to search past exam papers, questions, or mark schemes in the database (e.g. to see example questions or mark schemes).
   Do NOT use this for syllabus content lookups, as you already have the full syllabus specification context above.
   Params:
     - "query": (str) The past paper search query.
3. "generate_exam":
   Use this ONLY when the user requests generating a new exam.
   REQUIRED PARAMETERS: `exam_type` (e.g. {self.valid_exam_types[0] if self.valid_exam_types else 'Higher'}), `total_marks` (int), and `topics` (list of strings).
   OPTIONAL PARAMETER: `num_questions` (int, optional) - Number of questions in the exam.
   {exam_type_constraint}
   QUESTION COUNT WORKFLOW:
   - `num_questions` is optional.
   - If the user specifies a question count (e.g. "make a 30 mark exam with 4 questions"), pass `num_questions`: 4.
   - If the user HAS NOT specified or confirmed a question count, use action "none" and ask the user if they would like a specific number of questions or if they are happy with automatic question count selection.
   - Once the user confirms (e.g. specifies "4 questions", or says "auto", "default", "I'm fine with auto"), trigger `generate_exam`. (Omit `num_questions` or set to null if auto/default).
   Params:
     - "exam_type": (str)
     - "total_marks": (int)
     - "topics": (list of str)
     - "num_questions": (int or null, optional)
4. "mark_answer":
   Use this ONLY when the user wants to mark a student's answer.
   VALIDATION REQUIREMENT: You MUST have all 4 parameters provided: `question` (str), `marks` (int), `student_answer` (str), and `mark_scheme` (str).
   If any parameter is missing, you must NOT trigger this action. Instead, use action "none" and ask the user to provide the missing parameter(s) (e.g. ask for the mark scheme or the question).
   Params:
     - "question": (str)
     - "marks": (int)
     - "student_answer": (str)
     - "mark_scheme": (str)
5. "revision_materials":
   Use this when the user wants structured revision notes/materials for a topic.
   Params:
     - "topic": (str)
6. "save_preferences":
   Use this when the user gives custom preferences, instructions, or notes for future exams/content (e.g. "make exams 20 marks", "ignore topic X").
   Params:
     - "preferences": (str) The custom preference.

RESPONSE SCHEMA:
{{
  "thought": "Your internal thinking process about what the user wants and parameter validation check.",
  "action": "none" | "query_past_papers" | "generate_exam" | "mark_answer" | "revision_materials" | "save_preferences",
  "params": {{ ... }},
  "message": "Your conversational reply to the user (required if action is 'none', otherwise a brief placeholder)."
}}

IMPORTANT:
- Respond ONLY with the JSON block. Do not add any text before or after the JSON.
- Be very strict about validation requirements! Never generate an exam or mark an answer without all the parameters.
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
        action = res_dict.get("action", "none")
        params = res_dict.get("params", {})
        message = res_dict.get("message", "")

        # Process the action
        if action == "none":
            Colors.print_green(f"\n{message}")
            self.history.append({"role": "assistant", "content": message})

        elif action == "query_past_papers":
            query = params.get("query", "")
            Colors.print_blue(f"\n[Database Search] Searching past papers & mark schemes for: '{query}'...")
            try:
                rag_result = self.assistant.llm_client.invoke_qa(self.assistant.ms_qa_chain, query)
                obs_content = f"Database search results for query '{query}':\n{rag_result}"
                self.history.append({"role": "system", "content": obs_content})
                # Call LLM again with the new observation
                self.run_agent_loop_followup()
            except Exception as e:
                self.handle_action_error("query_past_papers", e)

        elif action == "generate_exam":
            exam_type = params.get("exam_type")
            total_marks = params.get("total_marks")
            topics = params.get("topics")
            num_questions = params.get("num_questions")

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
            
            user_prefs = params.get("user_preferences", {})
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

                # Output specification tree & selected subtopics to terminal during generation
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
            except Exception as e:
                self.handle_action_error("generate_exam", e)

        elif action == "mark_answer":
            question = params.get("question")
            marks = params.get("marks")
            student_answer = params.get("student_answer")
            mark_scheme = params.get("mark_scheme")

            # Validation checks
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
                
                # Feedback Quality Analysis bypassed as per user request
                os.makedirs("test_outputs", exist_ok=True)
                timestamp = int(time.time())
                filename = f"test_outputs/{self.subject}_{self.examiner}_feedback_{timestamp}.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(
                        f"# GCSE Marking Feedback\n\n"
                        f"**Question:** {question}\n\n"
                        f"**Max Marks:** {marks}\n\n"
                        f"**Student Answer:** {student_answer}\n\n"
                        f"**Mark Scheme:**\n{mark_scheme}\n\n"
                        f"**Feedback & Grade:**\n{feedback}\n"
                    )
                Colors.print_green(f"  -> Marking evaluation saved to: {filename}")

                obs_content = (
                    f"Answer marked. Feedback saved to '{filename}'. "
                    f"Awarded score detail: {feedback}."
                )
                self.history.append({"role": "system", "content": obs_content})
            except Exception as e:
                self.handle_action_error("mark_answer", e)

        elif action == "revision_materials":
            topic = params.get("topic")
            Colors.print_blue(f"\n[Revision] Generating revision materials for '{topic}'...")
            try:
                revision_text = self.assistant.revision_materials(topic)
                os.makedirs("test_outputs", exist_ok=True)
                timestamp = int(time.time())
                safe_topic = topic.replace(" ", "_").replace("–", "").replace("-", "")
                filename = f"test_outputs/{self.subject}_{self.examiner}_revision_{safe_topic}_{timestamp}.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(revision_text)
                Colors.print_green(f"  -> Revision notes saved to: {filename}")

                obs_content = f"Revision materials generated and saved to '{filename}'."
                self.history.append({"role": "system", "content": obs_content})
                self.run_agent_loop_followup()
            except Exception as e:
                self.handle_action_error("revision_materials", e)

        elif action == "save_preferences":
            pref = params.get("preferences")
            Colors.print_blue(f"\n[Preferences] Saving custom instruction: '{pref}'")
            self.preferences.append(pref)
            obs_content = f"Instruction saved: '{pref}'"
            self.history.append({"role": "system", "content": obs_content})
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
