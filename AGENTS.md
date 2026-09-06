# GCSE AI - Developer & Agent Instructions

This document provides architectural context, coding rules, and conventions for any AI agent or developer maintaining and extending the **GCSE AI** system.

---

## 1. Project Overview & Architecture

GCSE AI is an AI-powered revision, exam generation, and answer-marking platform tailored for GCSE examinations across multiple subjects (e.g. Physics, Biology, Religious Studies) and exam boards (e.g. Edexcel, AQA, OCR).

### Directory & Package Structure

```
GCSE AI/
├── config.py                 # Central SubjectConfig dataclass & board-specific definitions
├── assistant.py              # High-level orchestrator facade (GcseAssistant)
├── chatbot.py                # Interactive terminal-based agent chatbot
├── ingest.py                 # Entry point for PDF ingestion & vectorstore indexing
├── prompts/                  # Externalized prompt templates (.txt files with {placeholders})
│   ├── compile_target_subtopics.txt
│   ├── generate_study_questions.txt
│   ├── evaluate_topic_answer.txt
│   ├── topic_test_report.txt
│   ├── extract_spec_tree.txt
│   ├── generate_parent_description.txt
│   ├── make_parent_question.txt
│   ├── improve_question_coherence.txt
│   ├── classify_question_basic.txt
│   ├── classify_question_parent.txt
│   ├── build_spec_summary.txt
│   ├── command_word_guidance.txt
│   ├── generate_marking_policy.txt
│   ├── evaluate_handwritten_script.txt
│   └── ground_annotations.txt
├── llm/                      # LLM utilities, tracking, and embeddings
│   ├── client.py             # Multi-provider LLM client with retries & JSON parsing
│   ├── tracker.py            # TokenCostTracker for named session cost tracking
│   ├── prompts.py            # Centralized prompt template loader (load_prompt)
│   └── similarity.py         # Vector similarity and local diversity engines
├── exam_generation/          # Exam generation components
│   ├── generator.py          # QuestionGenerator (parallel exam question authoring)
│   ├── structure.py          # ExamStructureBuilder (pattern solver & mark distribution)
│   ├── formatter.py          # Markdown rendering for exams & individual questions
│   └── pdf_exporter.py       # PyMuPDF-based printable PDF export
├── marking/                  # Grading & feedback components
│   ├── marker.py             # ExamMarker (mark scheme generation, grading, model answers)
│   ├── visual_marker.py      # VisualExamMarker (3-stage vision marking & spatial grounding)
│   ├── visual_renderer.py    # Teacher annotation renderer (highlights, ticks, underlines, margin notes)
│   ├── segmenter.py          # ExamSegmenter (question paper, answer & mark scheme parsing)
│   └── report.py             # Markdown feedback reports & performance analytics
├── content/                  # Interactive revision & pedagogical tools
│   ├── topic_tester.py       # Interactive terminal-based quiz tester
│   ├── revision.py           # Structured revision note generation
│   └── guidance.py           # Command word explanations & specification breakdowns
├── ingestion/                # Data loading & indexing
│   ├── load_and_store.py     # PDF parsing, diagram screenshot cropping, FAISS vectorstore
│   ├── user_data.py          # User exam data conversion to JSON format
│   └── spec_summary.py       # Automated specification hierarchy indexing
├── data/                     # Raw exam PDFs, vectorstores, and extracted images
│   └── {Subject}/{Examiner}/
├── test_outputs/             # Generated exam papers and marking reports
└── tests/                    # Comprehensive unit test suite (85 tests)
```

---

## 2. Core Development Rules for AI Agents

### Rule 1: Multi-Subject & Exam Board Extensibility
- **NEVER hardcode subject-specific logic** (e.g. `if subject == 'Physics'`) in core logic files.
- All subject-specific rules, regex patterns (e.g. `mark_pattern`, `sub_question_pattern`, `question_split_pattern`), exam types, and model parameters must reside in `config.py` under `SubjectConfig` and `SUBJECT_CONFIGS`.
- The system must dynamically discover and handle any new subject or exam board added to `data/{Subject}/{Examiner}/` and `config.py`.

### Rule 2: Keep Prompts in `prompts/` (Externalized Templates)
- **Do NOT embed multi-line LLM prompts directly inside Python code files.**
- Save prompt templates as `.txt` files in the `prompts/` directory using `{placeholder}` formatting.
- Load them using `llm.prompts.load_prompt("template_name", **kwargs)`.
- If a prompt changes or a new LLM task is added, create a new template in `prompts/` and register it.

### Rule 3: Discrete Unit Ingestion (No Arbitrary Chunking)
- Do **NOT** use arbitrary fixed-character splitters or chunking fallbacks.
- GCSE question papers and mark schemes are parsed as discrete individual questions and mark scheme items using subject regexes.
- Specifications are processed in full via PyMuPDF/pypdf and indexed into structured specification trees (`specification_summary.json`).

### Rule 4: Named Session Token Cost Tracking
- Every LLM operation or chatbot action must be tracked under human-readable session names using `global_tracker.start_session(name)` and `global_tracker.end_session()`.
- Cumulative token usage and estimated costs are automatically logged to `session_token_costs.md`.

### Rule 5: Thorough Logging & Error Handling
- Use Python's standard `logging.getLogger(__name__)` across all modules.
- Ensure informative INFO logs for major operations and clear WARNING/ERROR logs with full contextual details.

### Rule 6: Testing & Verification
- All code changes must maintain a 100% test pass rate.
- Run tests using:
  ```powershell
  .\venv\Scripts\python -m unittest discover tests
  ```
- Any new feature or module should have matching unit tests under `tests/`.

---

## 3. How to Add a New Subject or Exam Board

1. **Add configuration in `config.py`**:
   ```python
   SUBJECT_CONFIGS["Chemistry-AQA"] = SubjectConfig(
       subject="Chemistry",
       examiner="AQA",
       mark_pattern=r"\[(\d{1,2})\s*marks?\]",
       question_pattern=r"(?i)Question\s*\d+",
       sub_question_pattern=r"\(\s*[a-h]\s*\)",
       exam_types=["Higher", "Foundation"],
   )
   ```
2. **Add PDFs in directory structure**:
   ```
   data/Chemistry/AQA/
   ├── Specification/Chemistry-AQA-Specification.pdf
   └── Exam-Types/Higher/
       ├── QuestionPapers/Chemistry-AQA-Paper1.pdf
       └── MarkSchemes/Chemistry-AQA-MarkScheme1.pdf
   ```
3. **Ingest data**:
   ```powershell
   .\venv\Scripts\python ingest.py Chemistry
   ```
4. **Launch chatbot**:
   ```powershell
   .\venv\Scripts\python chatbot.py
   ```

---

## 4. What to Do When Code Is Changed

Whenever any section of code is modified, added, or refactored, perform the following steps:

1. **Update Architecture in `AGENTS.md`**:
   - If files, directories, or modules are created, deleted, renamed, or relocated, update the directory and package structure tree in [Section 1](#1-project-overview--architecture).
   - Keep module descriptions and responsibilities accurate and aligned with the codebase.

2. **Update `README.md` (if necessary)**:
   - If changes affect architecture, CLI usage, public interfaces, features, dependencies, or setup instructions, update [`README.md`](README.md).

3. **Add and Commit to Git**:
   - Ensure all unit tests pass before committing:
     ```powershell
     .\venv\Scripts\python -m unittest discover tests
     ```
   - Stage modified and newly created files:
     ```powershell
     git add <files>
     ```
   - Commit with a clear, concise, and descriptive commit message:
     ```powershell
     git commit -m "feat/fix: descriptive commit summary"
     ```
