import os
import sys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from generate_content import GcseAssistant

def test_physics():
    print("Initializing GCSE Assistant for Physics-Edexcel...")
    assistant = GcseAssistant(subject="Physics", examiner="Edexcel")
    
    print(f"Loaded {len(assistant.questions)} past questions.")
    
    # Print some details of past questions to see if there are math questions
    math_examples = []
    for q in assistant.questions:
        content = q.get("question_content", "")
        # look for typical math keywords in physics questions
        if any(w in content.lower() for w in ["calculate", "equation", "formula", "use the equation"]):
            math_examples.append(q)
            
    print(f"Found {len(math_examples)} calculation/math-related past questions in database.")
    
    # Show 2 examples of past math questions
    print("\n--- Example Past Math Questions ---")
    for i, q in enumerate(math_examples[:2]):
        print(f"Example {i+1} ({q.get('marks')} marks, Topic: {q.get('topic')}):")
        print(q.get("question_content"))
        print("-" * 40)
        
    # Test generation of a physics question
    topic = "Topic 3 – Conservation of energy" # Let's try to find a subtopic
    print(f"\nGenerating a question for topic: {topic}...")
    
    # We will invoke the structure builder and generator for a 4-mark math question
    # We can retrieve the spec tree
    spec_tree = assistant.question_generator._get_spec_tree_cached(topic, "Higher")
    subtopics = list(spec_tree.keys())
    print("Available subtopics in specification:")
    for st in subtopics[:3]:
        print(f"- {st}: {spec_tree[st].get('description')[:100]}...")
        
    if subtopics:
        target_subtopic = subtopics[0]
        print(f"\nGenerating a 4-mark question on: {target_subtopic}")
        generated_q = assistant.question_generator.generate_question(
            marks=4,
            exam_topic="Higher",
            subtopic=target_subtopic,
            subtopic_info=spec_tree[target_subtopic]["description"],
            subject="Physics",
            examiner="Edexcel"
        )
        print("\n=== Generated Question ===")
        print(generated_q)
        print("==========================")
        
        # Also generate mark scheme for this question
        print("\nGenerating mark scheme for the generated question...")
        ms = assistant.exam_marker.create_mark_scheme(generated_q, 4, topic)
        print("\n=== Generated Mark Scheme ===")
        print(ms)
        print("=============================")

if __name__ == "__main__":
    test_physics()
