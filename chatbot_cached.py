"""GCSE AI Interactive Chatbot Script with Full Context & Prompt Caching.

Loads the full specification, all past papers, and all mark schemes for the chosen
subject and exam board into a unified, cached prompt context.
Removes dependencies on generate_content.py/RAG chains, allowing direct,
context-rich conversational assistance.
"""

import os
import sys
import time
import json
import logging
from typing import List, Tuple, Dict, Any, Optional

from dotenv import load_dotenv

# Load env variables relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"), override=True)
if "OPENAI_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()

# Suppress log clutter during interactive session
logging.getLogger().setLevel(logging.WARNING)

try:
    import pypdf
except ImportError:
    pypdf = None

from llm_client import LLMClient
from token_cost_tracker import global_tracker

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


def scan_available_subjects() -> List[Tuple[str, str]]:
    """Scan the data directory for available subject-examiner folders."""
    available = []
    data_dir = os.path.join(script_dir, "data")
    if not os.path.exists(data_dir):
        return [("Physics", "Edexcel"), ("Biology", "Edexcel"), ("ReligiousStudies", "AQA")]
    
    for subj in os.listdir(data_dir):
        subj_path = os.path.join(data_dir, subj)
        if os.path.isdir(subj_path):
            for exam in os.listdir(subj_path):
                exam_path = os.path.join(subj_path, exam)
                if os.path.isdir(exam_path):
                    available.append((subj, exam))
    
    if not available:
        return [("Physics", "Edexcel"), ("Biology", "Edexcel"), ("ReligiousStudies", "AQA")]
    return sorted(list(set(available)))


def load_full_subject_context(subject: str, examiner: str) -> Tuple[str, Dict[str, int]]:
    """Load specification, all past papers, and all markschemes into a unified text block.
    
    Caches the extracted text locally in .cache_context to ensure instant startup on subsequent runs.
    """
    cache_dir = os.path.join(script_dir, ".cache_context")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{subject}_{examiner}_full_context.txt")
    meta_file = os.path.join(cache_dir, f"{subject}_{examiner}_meta.json")

    if os.path.exists(cache_file) and os.path.exists(meta_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                context_text = f.read()
            with open(meta_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
            Colors.print_cyan(f"[Context Cache] Loaded cached context for {subject}-{examiner} ({stats.get('est_tokens', 0):,} tokens).")
            return context_text, stats
        except Exception as e:
            Colors.print_warning(f"[Context Cache] Could not read cache file ({e}). Rebuilding context...")

    Colors.print_cyan(f"[Context Loader] Building context for {subject}-{examiner}... (parsing specification, past papers, mark schemes)")
    t0 = time.time()
    subj_dir = os.path.join(script_dir, "data", subject, examiner)

    spec_text = ""
    qp_text = ""
    ms_text = ""
    spec_count = 0
    qp_count = 0
    ms_count = 0

    if os.path.exists(subj_dir):
        # 1. Load Specification
        spec_dir = os.path.join(subj_dir, "Specification")
        if os.path.exists(spec_dir):
            for file in os.listdir(spec_dir):
                fp = os.path.join(spec_dir, file)
                if file.lower().endswith(".txt"):
                    spec_count += 1
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        spec_text += f"\n--- Specification File: {file} ---\n" + f.read()
                elif file.lower().endswith(".pdf") and pypdf:
                    spec_count += 1
                    try:
                        reader = pypdf.PdfReader(fp)
                        txt = "\n".join([p.extract_text() or "" for p in reader.pages])
                        spec_text += f"\n--- Specification PDF: {file} ---\n" + txt
                    except Exception as e:
                        Colors.print_warning(f"Error reading PDF {file}: {e}")

        # 2. Load Past Papers & Mark Schemes from Exam-Types
        exam_types_dir = os.path.join(subj_dir, "Exam-Types")
        if os.path.exists(exam_types_dir):
            for root, _, files in os.walk(exam_types_dir):
                for file in files:
                    fp = os.path.join(root, file)
                    rel_path = os.path.relpath(fp, subj_dir)
                    
                    is_qp = any(k in root.lower() or k in file.lower() for k in ["questionpaper", "question_paper", "qp"])
                    is_ms = any(k in root.lower() or k in file.lower() for k in ["markscheme", "mark_scheme", "ms"])

                    extracted = ""
                    if file.lower().endswith(".txt"):
                        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                            extracted = f.read()
                    elif file.lower().endswith(".pdf") and pypdf:
                        try:
                            reader = pypdf.PdfReader(fp)
                            extracted = "\n".join([p.extract_text() or "" for p in reader.pages])
                        except Exception:
                            pass

                    if extracted.strip():
                        if is_ms:
                            ms_count += 1
                            ms_text += f"\n\n========================================\nMARK SCHEME: {rel_path}\n========================================\n" + extracted
                        elif is_qp:
                            qp_count += 1
                            qp_text += f"\n\n========================================\nQUESTION PAPER: {rel_path}\n========================================\n" + extracted

    context_sections = []
    if spec_text.strip():
        context_sections.append(f"==================================================\nOFFICIAL SPECIFICATION ({subject} {examiner})\n==================================================\n" + spec_text.strip())
    if qp_text.strip():
        context_sections.append(f"==================================================\nALL PAST QUESTION PAPERS ({qp_count} files)\n==================================================\n" + qp_text.strip())
    if ms_text.strip():
        context_sections.append(f"==================================================\nALL MARK SCHEMES ({ms_count} files)\n==================================================\n" + ms_text.strip())

    full_context = "\n\n".join(context_sections)
    est_tokens = len(full_context) // 4
    elapsed = time.time() - t0

    stats = {
        "spec_files": spec_count,
        "qp_files": qp_count,
        "ms_files": ms_count,
        "total_chars": len(full_context),
        "est_tokens": est_tokens,
        "load_time_sec": round(elapsed, 2)
    }

    # Save cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(full_context)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        Colors.print_warning(f"[Context Cache] Failed to write cache: {e}")

    Colors.print_green(f"[Context Loader] Loaded {stats['qp_files']} QPs, {stats['ms_files']} MSs (~{est_tokens:,} tokens) in {elapsed:.1f}s.")
    return full_context, stats


class CachedChatbotAgent:
    """Chatbot agent utilizing static prompt caching with full past paper context."""

    def __init__(self, subject: str, examiner: str):
        self.subject = subject
        self.examiner = examiner
        
        global_tracker.start_session(f"Chatbot - {subject} ({examiner})")
        
        Colors.print_cyan(f"\nInitializing Full-Context GCSE Chatbot for {subject} ({examiner})...")
        self.context_text, self.stats = load_full_subject_context(subject, examiner)
        
        # Initialize unified LLM Client
        self.llm_client = LLMClient()
        self.history: List[Dict[str, str]] = []

        # System prompt with context positioned first to leverage Prompt Caching
        self.system_prefix = f"""You are the GCSE AI Chatbot Assistant for {self.subject} ({self.examiner}).
You have FULL, direct context of the official syllabus specification, ALL past question papers, and ALL mark schemes for {self.subject} ({self.examiner}).

=== FULL SUBJECT CONTEXT (SPECIFICATION, PAST PAPERS & MARK SCHEMES) ===
{self.context_text}
========================================================================

YOUR ROLE & CAPABILITIES:
1. **Syllabus & Spec Queries:** Explain concepts, learning objectives, and definitions directly from the specification above.
2. **Past Paper & Mark Scheme Analysis:** Answer questions about past exam patterns, exact mark scheme points, command words, and recurring exam themes using the past papers above.
3. **Answer Marking & Grading:** When given a student's answer and question, evaluate it against the relevant official mark schemes in your context and award precise marks with detailed feedback.
4. **Question Generation & Revision:** Generate realistic GCSE practice questions or structured revision notes formatted exactly like official exam papers.

RESPONSE FORMAT:
- Provide clear, accurate, and helpful answers formatted in GitHub Markdown.
- Quote or reference official mark scheme points or specification topics whenever relevant.
"""

    def chat(self, user_msg: str):
        """Send user message and receive AI response using prompt caching."""
        self.history.append({"role": "user", "content": user_msg})
        
        # Keep last 10 messages for conversation continuity
        recent_history = self.history[-10:]
        
        history_str = ""
        for msg in recent_history:
            role_name = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"\n{role_name}: {msg['content']}\n"

        # The full prompt maintains the exact static system_prefix at the start for prompt caching
        full_prompt = f"{self.system_prefix}\n\n--- CONVERSATION HISTORY ---{history_str}\nAssistant:"

        try:
            Colors.print_blue("\nThinking (using cached context)...")
            response = self.llm_client.invoke(full_prompt)
            Colors.print_green(f"\n{response}")
            self.history.append({"role": "assistant", "content": response})
        except Exception as e:
            Colors.print_fail(f"\nError communicating with LLM: {e}")


def show_welcome_banner(subject: str, examiner: str, stats: Dict[str, int]):
    Colors.print_header("=" * 68)
    Colors.print_header("   GCSE AI CO-PILOT CHATBOT (FULL CONTEXT & PROMPT CACHED)   ")
    Colors.print_header("=" * 68)
    print(f" Active Subject : {subject} ({examiner})")
    print(f" Context Scope  : Specification + {stats.get('qp_files', 0)} Past Papers + {stats.get('ms_files', 0)} Mark Schemes")
    print(f" Total Context  : ~{stats.get('est_tokens', 0):,} tokens (Prompt Caching active)")
    Colors.print_blue("\nYou can ask anything directly:")
    print(" - 'Explain topic X according to the spec'")
    print(" - 'What questions came up on radioactivity in recent past papers?'")
    print(" - 'Mark my answer: [Question] ... [Answer] ...'")
    print(" - 'Create a 6-mark practice question on photosynthesis with mark scheme'")
    Colors.print_cyan("\nSlash Commands:")
    print("  /help                     Show list of direct commands")
    print("  /subject <subj> <board>   Switch active subject database")
    print("  /stats                    Display current context token statistics")
    print("  /clear                    Reset chat conversation history")
    print("  /exit                     Exit the program")
    Colors.print_header("=" * 68)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    available_subjects = scan_available_subjects()
    
    Colors.print_header("Available Subjects Found:")
    for idx, (subj, exam) in enumerate(available_subjects, start=1):
        print(f"  {idx}. {subj} ({exam})")
    
    selected_idx = 0
    while True:
        try:
            sel = input(f"\nSelect subject number [1-{len(available_subjects)}] (default 1): ").strip()
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
    
    # Initialize Agent with Full Context & Prompt Caching
    agent = CachedChatbotAgent(active_subj, active_exam)
    show_welcome_banner(agent.subject, agent.examiner, agent.stats)

    # Chat Loop
    while True:
        try:
            prompt_str = f"\n{Colors.BOLD}GCSE AI Full-Context ({agent.subject}-{agent.examiner}) > {Colors.ENDC}"
            user_input = input(prompt_str).strip()
            
            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()

                if cmd in ("/exit", "/quit"):
                    Colors.print_blue("\nGoodbye! Happy studying!")
                    break

                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /help                     Show command list")
                    print("  /subject <subj> <board>   Switch active subject context (e.g. /subject Physics Edexcel)")
                    print("  /stats                    View context size and token stats")
                    print("  /clear                    Reset conversation history")
                    print("  /exit or /quit            Exit chatbot")

                elif cmd == "/stats":
                    print(f"\nContext Stats for {agent.subject}-{agent.examiner}:")
                    print(f"  Specification Files: {agent.stats.get('spec_files', 0)}")
                    print(f"  Question Papers    : {agent.stats.get('qp_files', 0)}")
                    print(f"  Mark Schemes       : {agent.stats.get('ms_files', 0)}")
                    print(f"  Total Characters   : {agent.stats.get('total_chars', 0):,}")
                    print(f"  Estimated Tokens   : ~{agent.stats.get('est_tokens', 0):,}")

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
                        agent = CachedChatbotAgent(new_subj, new_exam)
                        show_welcome_banner(agent.subject, agent.examiner, agent.stats)
                    except Exception as e:
                        Colors.print_fail(f"\nFailed to load context for {new_subj}-{new_exam}: {e}")

                else:
                    Colors.print_warning(f"\nUnknown command: {cmd}. Type /help for assistance.")

            else:
                agent.chat(user_input)

        except KeyboardInterrupt:
            Colors.print_blue("\n\nSession ended by user (Ctrl+C). Goodbye!")
            break
        except Exception as e:
            Colors.print_fail(f"\nAn unexpected error occurred in chat loop: {e}")


if __name__ == "__main__":
    main()
