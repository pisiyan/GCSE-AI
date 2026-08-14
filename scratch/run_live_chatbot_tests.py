"""Live test harness for testing ChatbotAgent across all available subjects:
Physics (Edexcel), Biology (Edexcel), ReligiousStudies (AQA).
Tests granular features (1, 3, 4, 6, 8), RAG search, revision notes, preferences,
and complex multi-action queries without running expensive exam generation or full exam marking.
"""

import os
import sys
import time
import json

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from chatbot import ChatbotAgent, Colors

def run_tests_for_subject(subject: str, examiner: str) -> dict:
    """Run a suite of granular feature tests for a single subject/examiner pair."""
    print(f"\n=======================================================")
    print(f"   TESTING CHATBOT AGENT FOR: {subject} ({examiner})")
    print(f"=======================================================")

    t0 = time.time()
    try:
        agent = ChatbotAgent(subject, examiner)
    except Exception as e:
        print(f"FAILED to initialize ChatbotAgent for {subject} ({examiner}): {e}")
        return {"subject": subject, "examiner": examiner, "status": "FAIL_INIT", "error": str(e)}

    init_time = time.time() - t0
    test_results = []

    # Test cases to execute
    test_queries = [
        {
            "name": "1. Single Question Generation",
            "query": f"Give me a single 4-mark question on {subject} key concepts."
        },
        {
            "name": "3. Single Mark Scheme Generation",
            "query": f"Generate a mark scheme for a 4-mark question on {subject} topics."
        },
        {
            "name": "4. Model Answer Generation",
            "query": f"Write a model full-mark answer for a 4-mark question in {subject}."
        },
        {
            "name": "6. Specification Code & Subtopics Breakdown",
            "query": f"Show me the specification code and subtopics list for {subject}."
        },
        {
            "name": "8. Command Word Guidance",
            "query": "Explain what the command word 'Evaluate' means and how to structure a full mark response for it."
        },
        {
            "name": "RAG Database Search",
            "query": f"Search past papers for example questions on {subject} fundamentals."
        },
        {
            "name": "Revision Materials",
            "query": f"Generate revision notes for {subject} main topics."
        },
        {
            "name": "Save Custom Preference",
            "query": "Please remember that I prefer clear bullet point explanations."
        },
        {
            "name": "Complex Multi-Action Query (1, 3, 4, 6, 8 combined)",
            "query": (
                f"For {subject}, generate a 4-mark question, create its mark scheme, "
                f"write a model answer, explain the command word, and give me the spec breakdown."
            )
        }
    ]

    for test_case in test_queries:
        t_start = time.time()
        name = test_case["name"]
        query = test_case["query"]
        print(f"\n--- Running Test: [{name}] ---")
        print(f"Query: '{query}'")
        
        try:
            agent.run_agent_loop(query)
            elapsed = time.time() - t_start
            test_results.append({
                "name": name,
                "status": "PASS",
                "elapsed_sec": round(elapsed, 2),
                "history_length": len(agent.history)
            })
            print(f"✅ [{name}] PASSED in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"❌ [{name}] FAILED in {elapsed:.2f}s: {e}")
            test_results.append({
                "name": name,
                "status": "FAIL",
                "elapsed_sec": round(elapsed, 2),
                "error": str(e)
            })

    return {
        "subject": subject,
        "examiner": examiner,
        "status": "PASS",
        "init_time_sec": round(init_time, 2),
        "test_results": test_results
    }


def main():
    subjects = [
        ("Physics", "Edexcel"),
        ("Biology", "Edexcel"),
        ("ReligiousStudies", "AQA")
    ]

    summary_report = []
    overall_start = time.time()

    for subj, exam in subjects:
        report = run_tests_for_subject(subj, exam)
        summary_report.append(report)

    total_duration = time.time() - overall_start

    print("\n=======================================================")
    print("            LIVE CHATBOT TEST SUMMARY REPORT           ")
    print("=======================================================")
    print(f"Total Duration: {total_duration:.2f}s\n")

    for rep in summary_report:
        subj_str = f"{rep['subject']} ({rep['examiner']})"
        status = rep.get("status")
        init_t = rep.get("init_time_sec", 0)
        print(f"Subject: {subj_str} | Init Time: {init_t}s | Status: {status}")
        for tr in rep.get("test_results", []):
            st_icon = "✅" if tr["status"] == "PASS" else "❌"
            print(f"  {st_icon} {tr['name']}: {tr['status']} ({tr['elapsed_sec']}s)")
        print()

    # Save summary report to JSON for review
    report_file = os.path.join(project_root, "test_outputs", "live_chatbot_test_report.json")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_duration_sec": round(total_duration, 2),
            "subjects": summary_report
        }, f, indent=2)

    print(f"Saved live test report JSON to: {report_file}")

if __name__ == "__main__":
    main()
