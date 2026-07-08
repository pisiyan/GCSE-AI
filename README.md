# GCSE-AI

## Overview

**GCSE-AI** is a specialised **Retrieval-Augmented Generation (RAG)** framework designed to automate the creation and assessment of **GCSE examination materials**. 

The system processes official curriculum specifications and historical mark schemes to ensure that generated content aligns with specific exam board requirements.

The core logic uses **semantic search** to map user requested topics to official specification points. By analysing the structural patterns of past papers - such as the distribution of marks and the hierarchy of sub-questions - the system can generate new exam papers that mirror the **complexity and format** of authentic assessments.

## Technical Structure
The project is divided into three primary functional areas:

### 1. Data Ingestion and Processing (load_and_store.py)
This module handles the transformation of PDF documents into searchable data structures.

**PdfFile Class**: Extracts text from PDFs and uses regular expressions to identify question numbers, mark allocations, and hierarchical markers (roman numerals and letters). It distinguishes between different document types such as Specifications and Mark Schemes.

**VectorStore Class**: Manages the embedding process using HuggingFace models and stores the resulting vectors in a FAISS index for local retrieval.

**DatabaseManager**: Executes the batch processing of entire folders, ensuring that metadata (subject, examiner, topic, and year) is correctly attached to each data chunk.

### 2. Examination Engine (generate_content.py)
The main controller responsible for synthesising exams.

**RAG Pipeline**: Employs LangChain to query the FAISS database, retrieving relevant context from specifications to inform the LLM during question generation.

**Structural Synthesis**: Analyses the "parent-child" relationship of questions in past papers to replicate specific exam layouts (e.g., a lead-in description followed by multiple related sub-parts).

**Semantic Similarity**: Uses OpenAI embeddings to compare topic strings and ensure that generated questions maintain a high degree of relevance to the syllabus while avoiding repetitive content.

### 3.  Evaluating student performance (generate_content.py)

The system includes tools for automated marking of student generated content

**OCR Integration**: Converts images of student answers into text using vision-capable models, allowing for the assessment of handwritten work.

**Mark Scheme Generation**: When an official mark scheme is unavailable for a generated question, the system synthesises one based on the specification and historical marking patterns.

**Evaluative Logic**: Compares student responses against synthesised or retrieved mark schemes to provide specific mark awards and model answers.
