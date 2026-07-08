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
        system_instruction = f"""You are the GCSE AI Chatbot Assistant for {self.subject} ({self.examiner}).
Your goal is to help the user study, query syllabus/specification content, mark answers, and generate exams.

Current Active Subject: {self.subject} ({self.examiner})
User Preferences & Custom Constraints:
{self.get_preferences_text()}

You must respond strictly in JSON format matching the schema below.
If you need to perform an action, specify the action and its arguments in "action" and "params".
If you need to ask a question, tell the user something, or answer directly, specify action "none" and write your response in the "message" field.

AVAILABLE ACTIONS:
1. "none":
   Use this for regular conversation, explaining content, or when you need to ask the user for missing details.
   Params: None.
2. "query_content":
   Use this when the user asks a question about the syllabus or exam content, and you need to query the vector database (RAG) to retrieve accurate details.
   Params:
     - "query": (str) The specific search/syllabus query.
3. "generate_exam":
   Use this ONLY when the user requests generating a new exam.
   VALIDATION REQUIREMENT: You MUST have all 3 parameters provided: `exam_topic` (e.g. Higher/Foundation), `total_marks` (int), and `topics` (list of strings).
   If any parameter is missing, you must NOT trigger this action. Instead, use action "none" and ask the user to provide the missing parameter(s).
   Params:
     - "exam_topic": (str)
     - "total_marks": (int)
     - "topics": (list of str)
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
  "action": "none" | "query_content" | "generate_exam" | "mark_answer" | "revision_materials" | "save_preferences",
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

        elif action == "query_content":
            query = params.get("query", "")
            Colors.print_blue(f"\n[Syllabus Search] Searching vector database for: '{query}'...")
            try:
                rag_result = self.assistant.llm_client.invoke_qa(self.assistant.spec_qa_chain, query)
                obs_content = f"Database search results for query '{query}':\n{rag_result}"
                self.history.append({"role": "system", "content": obs_content})
                # Call LLM again with the new observation
                self.run_agent_loop_followup()
            except Exception as e:
                self.handle_action_error("query_content", e)

        elif action == "generate_exam":
            exam_topic = params.get("exam_topic")
            total_marks = params.get("total_marks")
            topics = params.get("topics")
            
            # Validation checks
            missing = []
            if not exam_topic: missing.append("exam_topic")
            if not total_marks: missing.append("total_marks")
            if not topics: missing.append("topics")
            
            if missing:
                err_msg = f"Validation Error: Cannot generate exam. Missing required parameter(s): {', '.join(missing)}."
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
            Colors.print_blue(f"  - Level/Topic: {exam_topic}")
            Colors.print_blue(f"  - Total Marks: {total_marks}")
            Colors.print_blue(f"  - Topic Areas: {topics}")
            
            try:
                exam_data = self.assistant.make_exam(exam_topic, total_marks, topics)
                exam_md = format_exam_as_markdown(self.subject, self.examiner, exam_topic, exam_data)
                
                os.makedirs("test_outputs", exist_ok=True)
                timestamp = int(time.time())
                filename = f"test_outputs/{self.subject}_{self.examiner}_exam_{timestamp}.md"
                with open(filename, "w", encoding="utf-8") as f:
                     f.write(exam_md)
                Colors.print_green(f"  -> Generated exam saved to: {filename}")

                # Analyze Exam Quality
                Colors.print_blue("  - Running quality analysis report...")
                quality_report = self.assistant.analyze_exam(exam_data)
                quality_md = self.assistant.quality_analyzer.generate_markdown_report(quality_report)
                
                filename_quality = f"test_outputs/{self.subject}_{self.examiner}_exam_quality_{timestamp}.md"
                with open(filename_quality, "w", encoding="utf-8") as f:
                    f.write(quality_md)
                Colors.print_green(f"  -> Quality report saved to: {filename_quality}")

                obs_content = (
                    f"Exam generated successfully and saved to file '{filename}'.\n"
                    f"Overall quality score is {quality_report.get('overall_score')}/100. "
                    f"Quality report is saved to '{filename_quality}'."
                )
                self.history.append({"role": "system", "content": obs_content})
                self.run_agent_loop_followup()
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
                
                # Feedback Quality Analysis
                fb_report = self.assistant.analyze_feedback(question, marks, mark_scheme, student_answer, feedback)
                
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
                        f"**Feedback & Grade:**\n{feedback}\n\n"
                        f"**Feedback Quality Score:** {fb_report.get('overall_score')}/100"
                    )
                Colors.print_green(f"  -> Marking evaluation saved to: {filename}")

                obs_content = (
                    f"Answer marked. Feedback saved to '{filename}'. "
                    f"Awarded score detail: {feedback}. Feedback Quality Score: {fb_report.get('overall_score')}/100."
                )
                self.history.append({"role": "system", "content": obs_content})
                self.run_agent_loop_followup()
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
