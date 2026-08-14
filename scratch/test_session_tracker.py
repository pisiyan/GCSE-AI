import os
import sys

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_cost_tracker import TokenCostTracker, global_tracker

def test_session_tracking():
    test_md = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_session_output.md")
    if os.path.exists(test_md):
        os.remove(test_md)

    tracker = TokenCostTracker()
    
    # Session 1: Chatbot
    tracker.start_session("Chatbot - Biology")
    tracker.track_call("gpt-4o-mini", "Explain photosynthesis", "Photosynthesis is the process...", prompt_tokens=200, completion_tokens=80)
    tracker.track_call("gpt-4o-mini", "What is the equation?", "Light + CO2 + H2O...", prompt_tokens=150, completion_tokens=50)
    summary1 = tracker.end_session(md_filepath=test_md)

    assert summary1["session_name"] == "Chatbot - Biology"
    assert summary1["calls"] == 2
    assert summary1["input_tokens"] == 350
    assert summary1["output_tokens"] == 130
    assert summary1["total_tokens"] == 480

    # Session 2: Exam Generation
    tracker.start_session("Exam Generation - Physics")
    tracker.track_call("gpt-4o", "Generate 5 circuit questions", "Q1...", prompt_tokens=1200, completion_tokens=400)
    summary2 = tracker.end_session(md_filepath=test_md)

    assert summary2["session_name"] == "Exam Generation - Physics"
    assert summary2["calls"] == 1
    assert summary2["total_tokens"] == 1600

    # Session 3: Exam Marking
    tracker.start_session("Exam Marking - Chemistry")
    tracker.track_call("gpt-4o-mini", "Mark student answer on moles", "Award 3/4 marks...", prompt_tokens=500, completion_tokens=150)
    summary3 = tracker.end_session(md_filepath=test_md)

    assert summary3["session_name"] == "Exam Marking - Chemistry"

    # Verify Markdown file content
    with open(test_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    print("\n--- GENERATED MARKDOWN FILE CONTENT ---")
    print(md_text)
    print("---------------------------------------")

    assert "# LLM Token & Cost Usage - Session Log" in md_text
    assert "Chatbot - Biology" in md_text
    assert "Exam Generation - Physics" in md_text
    assert "Exam Marking - Chemistry" in md_text
    assert "Cumulative Summary Across All Sessions" in md_text
    assert "Total Sessions Logged**: 3" in md_text

    print("\nALL SESSION TRACKER TESTS PASSED SUCCESSFULLY!")

    # Clean up test output file
    if os.path.exists(test_md):
        os.remove(test_md)

if __name__ == "__main__":
    test_session_tracking()
