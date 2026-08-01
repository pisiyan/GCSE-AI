# GCSE-AI

> **An AI-powered Retrieval-Augmented Generation (RAG) framework designed for the automated synthesis, structural layout, and marking of UK GCSE examination materials.**

---

## Overview

**GCSE-AI** is a specialized framework that automates the creation and grading of UK GCSE examination papers and mark schemes. By processing official curriculum specifications and historical past papers from major exam boards (e.g., **Edexcel**, **AQA**), the system ensures generated content strictly adheres to curriculum standards, topic weightings, mark allocations, and authentic exam paper formatting.

The core platform utilizes **semantic vector search** to map requested topics to official syllabus specifications, while structural layout engines replicate authentic exam paper hierarchies (lead-in descriptions, multi-part sub-questions, and mark distributions). Furthermore, GCSE-AI includes automated marking pipelines capable of evaluating handwritten student answers using Vision OCR LLMs.

---

## Key Features

- **Specification-Grounded RAG Engine**: Uses local FAISS vector databases and embeddings to map questions directly to official syllabus points, eliminating hallucinations and ensuring syllabus alignment.
- **Structural Exam Synthesis**: Replicates authentic exam paper layouts including parent questions, sub-questions (e.g., `(a)`, `(i)`), mark distributions, and lead-in scenario descriptions.
- **Automatic Mark Scheme Generation**: Synthesizes detailed, point-by-point mark schemes matching official marking criteria when original mark schemes are unavailable.
- **Automated OCR & Student Response Grading**: Processes images of handwritten student answers using vision-capable LLMs, evaluating work against official or generated mark schemes to award marks and provide model answers.
- **Subject & Exam Board Configurable**: Easily extensible regex patterns and chunking strategies tailored to specific exam boards (e.g., Edexcel Biology, Edexcel Physics, AQA Religious Studies).
- **Multi-LLM Provider Support**: Integrated via LangChain with support for OpenAI (GPT-4o/GPT-4), Google Gemini, and Anthropic Claude.

---

## Tech Stack

- **Core & Logic**: Python 3.10+
- **RAG & Orchestration**: [LangChain](https://github.com/langchain-ai/langchain) (`langchain-openai`, `langchain-huggingface`, `langchain-google-genai`, `langchain-anthropic`)
- **Vector Storage**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- **Embeddings**: HuggingFace SentenceTransformers & OpenAI Embeddings
- **PDF Extraction & Parsing**: `pypdf`, custom Regex structural parsers
- **Environment & Config**: `python-dotenv`, PyYAML, dataclasses

---

## System Architecture

```
GCSE AI/
├── data/                       # Ingested PDF specs, past papers, mark schemes
├── user_data/                  # FAISS vector indexes & processed metadata
├── prompts/                    # System prompts & generation templates
├── config.py                   # Subject-examiner regex patterns & RAG configs
├── load_and_store.py           # PDF extraction & FAISS vector store management
├── ingest.py                   # Data ingestion & vectorization script
├── exam_generator.py           # Core examination synthesis engine
├── generate_content.py         # Content generation orchestrator
├── exam_marker.py              # Vision OCR & automated grading engine
├── chatbot.py                  # Interactive query & assessment interface
├── similarity.py               # Semantic similarity & topic matching utilities
└── llm_client.py               # Multi-provider LLM client wrappers
```

## Usage

### 1. Ingest Curriculum & Past Papers
Process PDF specifications and past mark schemes into local FAISS vector stores:
```bash
python ingest.py
```

### 2. Generate GCSE Exam Papers
Synthesize a full exam paper based on subject configuration and topic selection:
```bash
python exam_generator.py
```

### 3. Automated Marking & Vision OCR
Grade student submissions (including image/handwritten responses):
```bash
python exam_marker.py
```

### 4. Interactive Chatbot Interface
Run the interactive console for custom topic queries and question generation:
```bash
python chatbot.py
```

---

## Supported Subject Configurations

Custom configurations and Regex parsing rules are defined in [config.py](file:///c:/Users/burak/IdeaProjects/GCSE%20AI/config.py):

| Subject | Exam Board |
| :--- | :--- |
| **Biology** | Edexcel |
| **Physics** | Edexcel |
| **Religious Studies** | AQA |

