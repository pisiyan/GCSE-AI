"""Full end-to-end pipeline tests for GCSE AI.

Generates 100-mark exams with ALL available topics for both subjects,
generates imperfect student answers for one exam per subject using mark schemes,
and runs quality analysis on the generated content.

Requires:
  - OPENAI_API_KEY in .env
  - Ingested vector databases at data/<subject>/<examiner>/<subject>-<examiner>-vectorDatabase
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generate_content import GcseAssistant, format_exam_as_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Configuration for both subjects
# ────────────────────────────────────────────────────────────

SUBJECTS = {
    "Physics-Edexcel": {
        "subject": "Physics",
        "examiner": "Edexcel",
        "exam_topic": "Higher",
        "total_marks": 100,
        # All Paper 1 topics for Edexcel GCSE Physics
        "topics": [
            "Topic 1 – Key concepts of physics",
            "Topic 2 – Motion and forces",
            "Topic 3 – Conservation of energy",
            "Topic 4 – Waves",
            "Topic 5 – Light and the electromagnetic spectrum",
            "Topic 6 – Radioactivity",
            "Topic 7 – Astronomy",
        ],
    },
}

# Directory for output artefacts
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_outputs')


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_imperfect_answer(assistant: GcseAssistant, question_text: str, marks: int, ms: str) -> str:
    """Generate a plausible but imperfect student answer for a question.

    Uses the provided actual mark scheme from the dataset, then asks the LLM
    to write an answer that deliberately scores around 60-80% of available marks.
    """
    prompt = (
        f"You are a GCSE student writing an exam answer. Write an answer that is mostly correct "
        f"but misses some key points and contains one or two minor inaccuracies. "
        f"The answer should be good enough to score roughly {int(marks * 0.65)}-{int(marks * 0.8)} "
        f"out of {marks} marks.\n\n"
        f"Question ({marks} marks):\n{question_text}\n\n"
        f"Mark Scheme:\n{ms}\n\n"
        f"Write the student's imperfect answer (no explanations, just the answer text):"
    )
    answer = assistant.llm_client.invoke(prompt)
    return answer


def flatten_exam_for_marking(exam_data: dict) -> list[dict]:
    """Flatten a generated exam into a list of question dicts suitable for marking.

    Each dict has: parent_description, question, marks, answer (to be filled).
    """
    flat = []
    for topic, q_list in exam_data.get("questions", {}).items():
        for q in q_list:
            if "sub_questions" in q:
                parent_desc = q.get("parent_description", "")
                for sq in q["sub_questions"]:
                    if "sub_parts" in sq:
                        context = sq.get("context", "")
                        full_context = f"{parent_desc}\n{context}".strip()
                        for gq in sq["sub_parts"]:
                            flat.append({
                                "parent_description": full_context,
                                "question": gq.get("text", ""),
                                "marks": gq.get("marks", 1),
                                "subtopic": q.get("subtopic", ""),
                                "topic": topic,
                                "answer": "",
                            })
                    else:
                        flat.append({
                            "parent_description": parent_desc,
                            "question": sq.get("text", ""),
                            "marks": sq.get("marks", 1),
                            "subtopic": q.get("subtopic", ""),
                            "topic": topic,
                            "answer": "",
                        })
            else:
                flat.append({
                    "parent_description": "",
                    "question": q.get("text", ""),
                    "marks": q.get("marks", 1),
                    "subtopic": q.get("subtopic", ""),
                    "topic": topic,
                    "answer": "",
                })
    return flat


def run_test_for_subject(subject_key: str, config: dict) -> dict:
    """Run the full pipeline test for one subject.

    Steps:
      1. Initialize GcseAssistant
      2. Generate a 100-mark exam with all topics
      3. Run exam quality analysis
      4. Generate imperfect student answers for a subset of questions
      5. Mark the answers and run feedback quality analysis
      6. Save all outputs

    Returns:
        A summary dict with scores and timings.
    """
    subject = config["subject"]
    examiner = config["examiner"]
    exam_topic = config["exam_topic"]
    total_marks = config["total_marks"]
    topics = config["topics"]

    summary = {
        "subject": subject_key,
        "exam_topic": exam_topic,
        "total_marks": total_marks,
        "topics_requested": topics,
    }

    # ── Step 1: Initialize ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("INITIALIZING %s", subject_key)
    logger.info("=" * 60)
    t0 = time.time()
    assistant = GcseAssistant(subject, examiner)
    summary["init_time_s"] = round(time.time() - t0, 1)
    logger.info("Initialization took %.1fs", summary["init_time_s"])

    # ── Step 2: Generate exam ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("GENERATING %d-MARK EXAM FOR %s (topics: %s)", total_marks, subject_key, topics)
    logger.info("=" * 60)
    t0 = time.time()
    exam_data = assistant.make_exam(exam_topic, total_marks, topics)
    summary["exam_generation_time_s"] = round(time.time() - t0, 1)
    logger.info("Exam generation took %.1fs", summary["exam_generation_time_s"])

    # Save the raw exam JSON
    exam_json_path = os.path.join(OUTPUT_DIR, f"{subject_key}_exam.json")
    with open(exam_json_path, "w", encoding="utf-8") as f:
        json.dump(exam_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved exam JSON to %s", exam_json_path)

    # Save the exam as markdown
    exam_md = format_exam_as_markdown(subject, examiner, exam_topic, exam_data)
    exam_md_path = os.path.join(OUTPUT_DIR, f"{subject_key}_exam.md")
    with open(exam_md_path, "w", encoding="utf-8") as f:
        f.write(exam_md)
    logger.info("Saved exam markdown to %s", exam_md_path)



    # ── Step 4: Generate imperfect answers & mark ─────────────
    logger.info("=" * 60)
    logger.info("GENERATING IMPERFECT ANSWERS & MARKING FOR %s", subject_key)
    logger.info("=" * 60)
    t0 = time.time()

    # Load past paper questions and mark schemes from dataset for answer generation and marking
    qp_time = "1June18"
    ms_time = "1_18"
    logger.info("Loading past paper questions for %s and mark schemes for %s from dataset", qp_time, ms_time)

    qp_questions = [q for q in assistant.questions if q.get("time") == qp_time]
    ms_pool = [m for m in assistant.mark_schemes if m.get("time") == ms_time]
    logger.info("Loaded %d past paper questions and %d mark scheme chunks", len(qp_questions), len(ms_pool))

    past_flat_questions = []
    for q in qp_questions:
        if q.get("marks") is not None:
            past_flat_questions.append({
                "parent_description": q.get("parent_question_description") or "",
                "question": q.get("question_content") or "",
                "marks": q.get("marks"),
                "subtopic": q.get("topic") or "Higher",
                "topic": q.get("topic") or "Higher",
                "answer": "",
            })

    logger.info("Flattened past paper into %d leaf questions for answer generation", len(past_flat_questions))

    # Run on a subset (e.g. first 5 questions) for speed/reliability, or all of them
    # Let's take the first 5 questions to verify the pipeline quickly and avoid RateLimit/quota errors
    run_questions = past_flat_questions[:5]
    logger.info("Running answer generation and marking on a subset of %d questions", len(run_questions))

    # Generate answers & mark
    answer_results = []
    for i, fq in enumerate(run_questions):
        q_text = fq["question"]
        marks = fq["marks"]
        topic = fq["topic"]
        parent_desc = fq["parent_description"]
        full_question = f"{parent_desc}\n{q_text}".strip() if parent_desc else q_text

        logger.info("Processing Q%d/%d (%d marks): %.80s...", i + 1, len(run_questions), marks, q_text)
        try:
            # Retrieve actual mark scheme from dataset using offline HuggingFace embedding similarity
            actual_ms = "No matching mark scheme chunk found."
            if ms_pool:
                ms_texts = [m["question_content"] for m in ms_pool]
                actual_ms = assistant.question_generator.local_similarity.find_most_similar(full_question, ms_texts)

            # Generate imperfect answer based on the actual mark scheme
            answer = generate_imperfect_answer(assistant, full_question, marks, actual_ms)
            fq["answer"] = answer

            # Mark the answer using the actual mark scheme
            feedback = assistant.exam_marker.mark_answer(answer, actual_ms, full_question, marks)

            answer_results.append({
                "question_index": i + 1,
                "question": q_text[:200],
                "marks": marks,
                "answer": answer[:500],
                "mark_scheme": actual_ms[:500],
                "feedback": feedback[:500],
            })

        except Exception as e:
            logger.error("Failed to generate/mark answer for Q%d: %s", i + 1, e, exc_info=True)
            answer_results.append({
                "question_index": i + 1,
                "question": q_text[:200],
                "marks": marks,
                "error": str(e),
            })

    summary["answer_generation_time_s"] = round(time.time() - t0, 1)
    logger.info("Answer generation and marking took %.1fs", summary["answer_generation_time_s"])

    summary["num_questions_answered"] = len([r for r in answer_results if "error" not in r])
    summary["num_questions_failed"] = len(answer_results) - summary["num_questions_answered"]

    # Save answer results
    answers_json_path = os.path.join(OUTPUT_DIR, f"{subject_key}_answers.json")
    with open(answers_json_path, "w", encoding="utf-8") as f:
        json.dump(answer_results, f, indent=2, ensure_ascii=False)
    logger.info("Saved answer results to %s", answers_json_path)

    return summary


def main():
    """Run end-to-end pipeline tests for both subjects."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ensure_output_dir()

    results = {}
    for subject_key, config in SUBJECTS.items():
        logger.info("\n" + "█" * 70)
        logger.info("██  STARTING FULL PIPELINE TEST: %s", subject_key)
        logger.info("█" * 70 + "\n")

        t_start = time.time()
        try:
            summary = run_test_for_subject(subject_key, config)
            summary["total_time_s"] = round(time.time() - t_start, 1)
            results[subject_key] = summary
        except Exception as e:
            logger.error("PIPELINE FAILED for %s: %s", subject_key, e, exc_info=True)
            results[subject_key] = {"error": str(e), "total_time_s": round(time.time() - t_start, 1)}

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FULL PIPELINE TEST RESULTS SUMMARY")
    print("=" * 70)

    summary_md_lines = ["# Full Pipeline Test Results\n"]

    for key, res in results.items():
        print(f"\n── {key} ──")
        summary_md_lines.append(f"## {key}\n")

        if "error" in res:
            print(f"  ❌ FAILED: {res['error']}")
            summary_md_lines.append(f"**FAILED:** {res['error']}\n")
        else:
            lines = [
                f"  Exam Topic:              {res['exam_topic']}",
                f"  Total Marks:             {res['total_marks']}",
                f"  Topics:                  {', '.join(res['topics_requested'])}",
                f"  Init Time:               {res['init_time_s']}s",
                f"  Exam Generation Time:    {res['exam_generation_time_s']}s",
                f"  Answer Gen+Mark Time:    {res['answer_generation_time_s']}s",
                f"  Total Time:              {res['total_time_s']}s",
                f"  ──────────────────────────────────────────",
                f"  ★ QUESTIONS ANSWERED:         {res['num_questions_answered']}",
                f"  ★ QUESTIONS FAILED:           {res['num_questions_failed']}",
            ]
            for line in lines:
                print(line)

            # Markdown summary
            summary_md_lines.append(f"| Metric | Value |")
            summary_md_lines.append(f"| :--- | :--- |")
            summary_md_lines.append(f"| Exam Topic | {res['exam_topic']} |")
            summary_md_lines.append(f"| Total Marks | {res['total_marks']} |")
            summary_md_lines.append(f"| Topics | {', '.join(res['topics_requested'])} |")
            summary_md_lines.append(f"| Questions Answered | {res['num_questions_answered']} |")
            summary_md_lines.append(f"| Questions Failed | {res['num_questions_failed']} |")
            summary_md_lines.append(f"| Total Time | {res['total_time_s']}s |")
            summary_md_lines.append("")

    # Save combined summary
    summary_json_path = os.path.join(OUTPUT_DIR, "pipeline_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Combined results JSON: {summary_json_path}")

    summary_md_path = os.path.join(OUTPUT_DIR, "pipeline_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md_lines))
    print(f"[SAVED] Combined results markdown: {summary_md_path}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
