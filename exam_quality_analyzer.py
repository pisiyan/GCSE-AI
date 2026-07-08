"""GCSE AI — Exam Quality Analyzer.

Analyses a generated GCSE exam against historical past papers in the database
using both quantitative metrics and qualitative LLM evaluation.
"""

import logging
import re
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ExamQualityAnalyzer:
    """Evaluates generated exam quality against historical past papers."""

    def __init__(self, gcse_assistant) -> None:
        """Initialize the analyzer.

        Args:
            gcse_assistant: An initialized instance of GcseAssistant.
        """
        self.assistant = gcse_assistant
        self.llm = gcse_assistant.llm_client
        self.similarity_engine = gcse_assistant.similarity
        self.past_questions = gcse_assistant.questions
        self.prompts = gcse_assistant.prompts
        self.queries = gcse_assistant.queries
        self.spec_qa_chain = gcse_assistant.spec_qa_chain

    def extract_questions_flat(self, generated_exam: dict) -> list[dict]:
        """Flatten a generated exam structure into a list of individual (leaf) questions.

        Preserves question hierarchies, numbers, subtopics, contexts, and marks.
        """
        flat_questions = []
        questions_by_topic = generated_exam.get("questions", {})

        for topic, question_list in questions_by_topic.items():
            for q in question_list:
                q_num = q.get("number", "")
                subtopic = q.get("subtopic", "")

                # Check if it's a parent question with sub-parts
                if "sub_questions" in q:
                    parent_desc = q.get("parent_description", "") or ""
                    for sq in q["sub_questions"]:
                        label = sq.get("label", "")

                        # Sub-question might have further sub-parts (grandchild questions)
                        if "sub_parts" in sq:
                            child_context = sq.get("context", "") or ""
                            combined_parent_context = f"{parent_desc}\n{child_context}".strip()
                            for gq in sq["sub_parts"]:
                                g_label = gq.get("label", "")
                                flat_questions.append({
                                    "id": f"{q_num}_{label}_{g_label}".replace(")", "").replace(" ", ""),
                                    "text": gq.get("text", "") or "",
                                    "marks": gq.get("marks", 0),
                                    "parent_context": combined_parent_context,
                                    "subtopic": subtopic,
                                    "topic": topic,
                                    "full_text": f"{combined_parent_context}\n{g_label} {gq.get('text', '')}".strip()
                                })
                        else:
                            # Standard child question
                            flat_questions.append({
                                "id": f"{q_num}_{label}".replace(")", "").replace(" ", ""),
                                "text": sq.get("text", "") or "",
                                "marks": sq.get("marks", 0),
                                "parent_context": parent_desc,
                                "subtopic": subtopic,
                                "topic": topic,
                                "full_text": f"{parent_desc}\n{label} {sq.get('text', '')}".strip()
                            })
                else:
                    # Basic standalone question
                    flat_questions.append({
                        "id": q_num.replace(")", "").replace(" ", ""),
                        "text": q.get("text", "") or "",
                        "marks": q.get("marks", 0),
                        "parent_context": "",
                        "subtopic": subtopic,
                        "topic": topic,
                        "full_text": q.get("text", "") or ""
                    })
        return flat_questions

    def analyze_phrasing_similarity(self, flat_questions: list[dict]) -> tuple[float, list[dict]]:
        """Evaluate semantic similarity of the generated questions compared to actual past papers.

        Uses a sweet spot scoring function:
          - Cosine similarity between 0.60 and 0.85 maps to a perfect 100% score (good style, not direct copy).
          - Similarity >= 0.95 indicates plagiarism/duplication, penalizing the score.
          - Similarity < 0.60 scales down linearly to reflect phrasing divergence.
        """
        if not flat_questions:
            return 100.0, []

        details = []
        scores = []

        # Group past questions by mark for fairer style comparison
        past_by_mark = {}
        for pq in self.past_questions:
            m = pq.get("marks")
            if m is not None:
                past_by_mark.setdefault(m, []).append(pq)

        # Fallback list of all past questions that have content
        all_past_texts = [pq.get("question_content") for pq in self.past_questions if pq.get("question_content")]

        for fq in flat_questions:
            gen_text = fq["text"]
            marks = fq["marks"]

            # Select comparison pool from past papers (prefer same marks, fallback to all)
            comparison_pool = []
            if marks in past_by_mark:
                comparison_pool = [pq.get("question_content") for pq in past_by_mark[marks] if pq.get("question_content")]

            if not comparison_pool:
                comparison_pool = all_past_texts

            if not comparison_pool:
                # No past questions available at all
                scores.append(100.0)
                details.append({
                    "id": fq["id"],
                    "question": gen_text,
                    "max_similarity": 0.0,
                    "matched_past_question": "N/A",
                    "score": 100.0
                })
                continue

            # Batch compute similarities using SimilarityEngine
            sim_scores = self.similarity_engine.compute_similarity_scores([gen_text], comparison_pool)

            best_match, max_sim = sim_scores[0] if sim_scores else ("", 0.0)

            # Sweet spot scoring function
            if max_sim >= 0.95:
                # Plagiarism penalty
                q_score = 50.0 - (max_sim - 0.95) * 1000.0
                q_score = max(0.0, q_score)
            elif max_sim >= 0.60:
                # Perfect alignment in style
                q_score = 100.0
            else:
                # Phrasing divergence
                q_score = (max_sim / 0.60) * 100.0
                q_score = max(0.0, q_score)

            scores.append(q_score)
            details.append({
                "id": fq["id"],
                "question": gen_text,
                "max_similarity": round(max_sim, 3),
                "matched_past_question": best_match[:150] + "..." if len(best_match) > 150 else best_match,
                "score": round(q_score, 1)
            })

        avg_score = float(np.mean(scores)) if scores else 100.0
        return round(avg_score, 1), details

    def analyze_mark_distribution(self, flat_questions: list[dict]) -> float:
        """Compare the mark distribution of the generated exam against past papers.

        Uses cosine similarity of the relative frequency vectors.
        """
        past_marks = [q.get("marks") for q in self.past_questions if q.get("marks") is not None]
        if not past_marks or not flat_questions:
            return 100.0

        gen_marks = [fq["marks"] for fq in flat_questions]

        # Get all unique marks across both
        all_mark_values = sorted(list(set(past_marks + gen_marks)))

        # Count frequencies
        past_counts = {m: past_marks.count(m) for m in all_mark_values}
        gen_counts = {m: gen_marks.count(m) for m in all_mark_values}

        # Convert to relative frequencies (probability vectors)
        total_past = sum(past_counts.values())
        total_gen = sum(gen_counts.values())

        p_vec = np.array([past_counts[m] / total_past for m in all_mark_values])
        g_vec = np.array([gen_counts[m] / total_gen for m in all_mark_values])

        # Compute cosine similarity
        dot_product = np.dot(p_vec, g_vec)
        norm_p = np.linalg.norm(p_vec)
        norm_g = np.linalg.norm(g_vec)

        if norm_p == 0 or norm_g == 0:
            return 0.0

        cos_sim = dot_product / (norm_p * norm_g)
        score = cos_sim * 100.0
        return round(score, 1)

    def analyze_command_words(self, flat_questions: list[dict]) -> tuple[float, list[dict]]:
        """Verify if command words in generated questions match typical past paper mark allocations."""
        if not flat_questions:
            return 100.0, []

        past_word_marks = {}
        command_words_list = [
            "describe", "explain", "state", "identify", "give", "outline",
            "compare", "contrast", "evaluate", "assess", "discuss", "analyse",
            "calculate", "work out", "determine", "suggest", "justify", "define",
            "show", "interpret", "predict", "complete", "draw", "label", "plot", "name"
        ]

        def regex_extract_command_word(text: str) -> Optional[str]:
            if not text:
                return None
            cleaned = text.strip().lower()
            for word in command_words_list:
                if re.search(r'\b' + word + r'\b', cleaned[:50]):
                    return word.capitalize()
            return None

        # Build baseline from past papers
        for pq in self.past_questions:
            content = pq.get("question_content")
            marks = pq.get("marks")
            if content and marks is not None:
                word = regex_extract_command_word(content)
                if word:
                    past_word_marks.setdefault(word, []).append(marks)

        details = []
        scores = []

        for fq in flat_questions:
            gen_text = fq["text"]
            marks = fq["marks"]

            # Extract command word via LLM
            prompt = self.prompts["get_command_word"].format(question=gen_text)
            extracted_word = self.llm.invoke(prompt).strip()
            extracted_word = re.sub(r'[^a-zA-Z\s]', '', extracted_word).strip()

            typical_marks = past_word_marks.get(extracted_word, [])

            if not typical_marks:
                score = 80.0
                reason = f"Command word '{extracted_word}' not found in past papers baseline. Defaulting to general alignment."
            else:
                if marks in typical_marks:
                    score = 100.0
                    reason = f"Typical mark allocation. Real past papers use '{extracted_word}' for {marks} marks."
                else:
                    min_mark = min(typical_marks)
                    max_mark = max(typical_marks)
                    if min_mark <= marks <= max_mark:
                        score = 70.0
                        reason = f"Atypical but within range [{min_mark}-{max_mark}] marks for '{extracted_word}'. Past paper marks: {sorted(list(set(typical_marks)))}."
                    else:
                        score = 30.0
                        reason = f"Out of typical range [{min_mark}-{max_mark}] marks for '{extracted_word}'. Past paper marks: {sorted(list(set(typical_marks)))}."

            scores.append(score)
            details.append({
                "id": fq["id"],
                "question": gen_text,
                "command_word": extracted_word,
                "marks": marks,
                "score": score,
                "reason": reason
            })

        avg_score = float(np.mean(scores)) if scores else 100.0
        return round(avg_score, 1), details

    def analyze_specification_relevance(self, flat_questions: list[dict]) -> tuple[float, list[dict]]:
        """Verify generated questions align with specification content from the vector store."""
        if not flat_questions:
            return 100.0, []

        scores = []
        details = []

        for fq in flat_questions:
            gen_text = fq["text"]
            parent_ctx = fq["parent_context"]
            subtopic = fq["subtopic"]
            topic = fq["topic"]

            query_context = f"{parent_ctx}\n{gen_text}".strip()

            # Query vector database via specification retriever
            retrieved_docs = self.assistant.spec_retriever.invoke(query_context)
            spec_content = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Evaluate with LLM
            prompt = (
                f"You are an expert GCSE examiner checking if a generated question aligns with the specification.\n\n"
                f"GENERATED QUESTION:\n{query_context}\n\n"
                f"TOPIC/SUBTOPIC: {topic} / {subtopic}\n\n"
                f"RELEVANT SPECIFICATION CHUNKS:\n{spec_content}\n\n"
                f"Evaluate if the generated question directly tests the retrieved specification points.\n"
                f"Rate the relevance on a scale of 0 to 100, where:\n"
                f"- 100: Question perfectly aligns with the specification points.\n"
                f"- 70-99: Question aligns well but covers slightly broader/narrower scope.\n"
                f"- 40-69: Question is tangentially related but doesn't test the core specification point well.\n"
                f"- 0-39: Question tests topics completely outside the specification.\n\n"
                f"Respond ONLY with a JSON object in the following format:\n"
                f"{{\n"
                f"  \"relevance_score\": <int 0-100>,\n"
                f"  \"reasoning\": \"<brief explanation of why this score was given, pointing out any alignment or deviations>\"\n"
                f"}}"
            )

            try:
                result = self.llm.invoke_json(prompt)
                score = float(result.get("relevance_score", 80.0))
                reasoning = result.get("reasoning", "No details provided.")
            except Exception as e:
                logger.error("Failed to evaluate specification relevance: %s", e)
                score = 80.0
                reasoning = "Error during LLM evaluation. Defaulting to average score."

            scores.append(score)
            details.append({
                "id": fq["id"],
                "question": gen_text,
                "score": score,
                "reasoning": reasoning
            })

        avg_score = float(np.mean(scores)) if scores else 100.0
        return round(avg_score, 1), details

    def evaluate_qualitative_aspects(self, flat_questions: list[dict]) -> tuple[float, list[dict]]:
        """Perform qualitative review of each question (clarity, accuracy, difficulty, marks alignment)."""
        if not flat_questions:
            return 100.0, []

        scores = []
        details = []

        for fq in flat_questions:
            gen_text = fq["text"]
            marks = fq["marks"]
            subtopic = fq["subtopic"]
            parent_ctx = fq["parent_context"]

            prompt = self.prompts["evaluate_question_quality"].format(
                question_text=gen_text,
                marks=marks,
                subtopic=subtopic,
                parent_context=parent_ctx if parent_ctx else "None"
            )

            try:
                res = self.llm.invoke_json(prompt)

                clarity = float(res.get("clarity_score", 8)) * 10.0
                accuracy = float(res.get("accuracy_score", 8)) * 10.0
                difficulty = float(res.get("difficulty_score", 8)) * 10.0
                alignment = float(res.get("mark_alignment_score", 8)) * 10.0

                overall_q = float(res.get("overall_qualitative_score", 8.0)) * 10.0
                feedback = res.get("general_feedback", "")

                scores.append(overall_q)
                details.append({
                    "id": fq["id"],
                    "question": gen_text,
                    "scores": {
                        "clarity": clarity,
                        "accuracy": accuracy,
                        "difficulty": difficulty,
                        "mark_alignment": alignment
                    },
                    "overall_score": overall_q,
                    "feedback": feedback
                })
            except Exception as e:
                logger.error("Failed to qualitatively evaluate question: %s", e)
                scores.append(80.0)
                details.append({
                    "id": fq["id"],
                    "question": gen_text,
                    "scores": {"clarity": 80.0, "accuracy": 80.0, "difficulty": 80.0, "mark_alignment": 80.0},
                    "overall_score": 80.0,
                    "feedback": "Failed to parse qualitative evaluation."
                })

        avg_score = float(np.mean(scores)) if scores else 100.0
        return round(avg_score, 1), details

    def analyze_internal_coherence(self, generated_exam: dict) -> tuple[float, list[dict]]:
        """Evaluate the internal similarity, coherence, and redundancy of multi-part questions.

        - Sub-questions under the same parent must be semantically related (coherence).
        - They must refer to the same question context (similarity to parent_description).
        - They must NOT be too similar to each other (no redundancy or duplicate testing).
        """
        questions_by_topic = generated_exam.get("questions", {})
        parent_questions = []

        # Extract all parent questions with multiple parts
        for topic, q_list in questions_by_topic.items():
            for q in q_list:
                if "sub_questions" in q and len(q["sub_questions"]) > 1:
                    parent_questions.append(q)

        if not parent_questions:
            return 100.0, [{"info": "No multi-part questions found in the exam. Coherence is vacuously perfect."}]

        scores = []
        details = []

        for pq in parent_questions:
            parent_desc = pq.get("parent_description", "") or ""
            sub_questions = pq["sub_questions"]

            # Extract texts of all subquestions
            sub_texts = []
            sub_labels = []
            for sq in sub_questions:
                if "sub_parts" in sq:
                    for gq in sq["sub_parts"]:
                        sub_texts.append(gq.get("text", "") or "")
                        sub_labels.append(f"{sq.get('label', '')} {gq.get('label', '')}")
                else:
                    sub_texts.append(sq.get("text", "") or "")
                    sub_labels.append(sq.get("label", ""))

            if len(sub_texts) < 2:
                continue

            pq_scores = []
            pq_reasons = []

            # 1. Check similarity to parent context
            if parent_desc:
                all_texts = [parent_desc] + sub_texts
                embeddings = self.similarity_engine.llm_client.get_embeddings(all_texts)
                parent_emb = np.array(embeddings[0])
                sub_embs = [np.array(e) for e in embeddings[1:]]

                # Check alignment of each sub-question to parent context
                for i, sub_emb in enumerate(sub_embs):
                    norm_parent = np.linalg.norm(parent_emb)
                    norm_sub = np.linalg.norm(sub_emb)
                    if norm_parent > 0 and norm_sub > 0:
                        sim = float(np.dot(parent_emb, sub_emb) / (norm_parent * norm_sub))
                    else:
                        sim = 0.0

                    if sim >= 0.3:
                        align_score = 100.0
                    elif sim >= 0.15:
                        align_score = 70.0 + (sim - 0.15) / 0.15 * 30.0
                    else:
                        align_score = max(0.0, (sim / 0.15) * 70.0)
                    pq_scores.append(align_score)
                    if align_score < 80.0:
                        pq_reasons.append(f"Sub-part {sub_labels[i]} has low similarity ({round(sim, 2)}) to the parent description context.")
            else:
                embeddings = self.similarity_engine.llm_client.get_embeddings(sub_texts)
                sub_embs = [np.array(e) for e in embeddings]

            # 2. Check pairwise redundancy/overlap among sub-questions
            n_subs = len(sub_texts)
            for i in range(n_subs):
                for j in range(i + 1, n_subs):
                    emb_i = sub_embs[i]
                    emb_j = sub_embs[j]
                    norm_i = np.linalg.norm(emb_i)
                    norm_j = np.linalg.norm(emb_j)
                    if norm_i > 0 and norm_j > 0:
                        sim = float(np.dot(emb_i, emb_j) / (norm_i * norm_j))
                    else:
                        sim = 0.0

                    if sim > 0.80:
                        redundancy_score = max(0.0, 100.0 - (sim - 0.80) * 500.0)
                        pq_reasons.append(f"High redundancy warning: Sub-parts {sub_labels[i]} and {sub_labels[j]} are very similar ({round(sim, 2)}) and may test duplicate content.")
                    elif sim > 0.70:
                        redundancy_score = 100.0 - (sim - 0.70) * 100.0
                        pq_reasons.append(f"Caution: Sub-parts {sub_labels[i]} and {sub_labels[j]} have high similarity ({round(sim, 2)}). Ensure they do not require identical answers.")
                    elif sim < 0.15:
                        redundancy_score = max(0.0, (sim / 0.15) * 100.0)
                        pq_reasons.append(f"Low cohesion warning: Sub-parts {sub_labels[i]} and {sub_labels[j]} are quite unrelated ({round(sim, 2)}) under the same parent question.")
                    else:
                        redundancy_score = 100.0

                    pq_scores.append(redundancy_score)

            pq_avg = float(np.mean(pq_scores)) if pq_scores else 100.0
            scores.append(pq_avg)
            details.append({
                "parent_question": parent_desc[:100] + "..." if len(parent_desc) > 100 else parent_desc,
                "score": round(pq_avg, 1),
                "warnings": pq_reasons
            })

        avg_score = float(np.mean(scores)) if scores else 100.0
        return round(avg_score, 1), details

    def analyze_exam_quality(self, generated_exam: dict) -> dict:
        """Run full quality analysis on a generated exam."""
        logger.info("Extracting flat questions...")
        flat_qs = self.extract_questions_flat(generated_exam)

        logger.info("Analyzing phrasing similarity...")
        phrasing_score, phrasing_details = self.analyze_phrasing_similarity(flat_qs)

        logger.info("Analyzing mark distribution...")
        mark_dist_score = self.analyze_mark_distribution(flat_qs)

        logger.info("Analyzing command words...")
        cmd_score, cmd_details = self.analyze_command_words(flat_qs)

        logger.info("Analyzing specification relevance...")
        spec_score, spec_details = self.analyze_specification_relevance(flat_qs)

        logger.info("Evaluating internal multi-part question coherence...")
        coherence_score, coherence_details = self.analyze_internal_coherence(generated_exam)

        logger.info("Evaluating qualitative aspects...")
        qual_score, qual_details = self.evaluate_qualitative_aspects(flat_qs)

        # Configurable weights summing to 1.0
        weights = {
            "phrasing": 0.20,
            "mark_distribution": 0.15,
            "command_words": 0.15,
            "spec_relevance": 0.20,
            "internal_coherence": 0.15,
            "qualitative": 0.15
        }

        overall_score = (
            phrasing_score * weights["phrasing"] +
            mark_dist_score * weights["mark_distribution"] +
            cmd_score * weights["command_words"] +
            spec_score * weights["spec_relevance"] +
            coherence_score * weights["internal_coherence"] +
            qual_score * weights["qualitative"]
        )

        overall_score = round(overall_score, 1)

        report = {
            "overall_score": overall_score,
            "metrics": {
                "phrasing_similarity_score": phrasing_score,
                "mark_distribution_score": mark_dist_score,
                "command_word_alignment_score": cmd_score,
                "spec_relevance_score": spec_score,
                "internal_coherence_score": coherence_score,
                "qualitative_score": qual_score
            },
            "weights": weights,
            "details": {
                "phrasing_similarity": phrasing_details,
                "command_word_alignment": cmd_details,
                "spec_relevance": spec_details,
                "internal_coherence": coherence_details,
                "qualitative": qual_details
            }
        }

        return report

    def generate_markdown_report(self, report: dict) -> str:
        """Generate a formatted markdown report from the analysis results."""
        metrics = report["metrics"]
        details = report["details"]

        md = []
        md.append("# Exam Quality Analysis Report")
        md.append(f"### **Overall Exam Quality Score: {report['overall_score']}/100**\n")

        score = report['overall_score']
        if score >= 90:
            rating = "[EXCELLENT] (Ready for Use)"
        elif score >= 75:
            rating = "[GOOD] (Minor revisions recommended)"
        elif score >= 50:
            rating = "[FAIR] (Requires manual revision)"
        else:
            rating = "[POOR] (Do not use, regenerate)"

        md.append(f"**Rating:** {rating}\n")

        md.append("## Quantitative Metrics Breakdown")
        md.append("| Metric | Weight | Score |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| Phrasing & Style Similarity | {report['weights']['phrasing']*100}% | {metrics['phrasing_similarity_score']}/100 |")
        md.append(f"| Mark Distribution Match | {report['weights']['mark_distribution']*100}% | {metrics['mark_distribution_score']}/100 |")
        md.append(f"| Command Word & Mark Alignment | {report['weights']['command_words']*100}% | {metrics['command_word_alignment_score']}/100 |")
        md.append(f"| Specification Relevance & Coverage | {report['weights']['spec_relevance']*100}% | {metrics['spec_relevance_score']}/100 |")
        md.append(f"| Internal Multi-Part Coherence & Redundancy | {report['weights']['internal_coherence']*100}% | {metrics['internal_coherence_score']}/100 |")
        md.append(f"| Qualitative Academic Evaluation | {report['weights']['qualitative']*100}% | {metrics['qualitative_score']}/100 |")
        md.append("")

        md.append("## Multi-part Question Internal Coherence & Redundancy Analysis")
        coherence_list = details.get("internal_coherence", [])
        if not coherence_list or (len(coherence_list) == 1 and "info" in coherence_list[0]):
            md.append("No multi-part questions found in the exam. Coherence is vacuously perfect.")
        else:
            for item in coherence_list:
                md.append(f"### Question Context: *\"{item['parent_question']}\"*")
                md.append(f"- **Coherence & Uniqueness Score:** {item['score']}/100")
                if item.get("warnings"):
                    md.append("- **Observations/Warnings:**")
                    for warning in item["warnings"]:
                        md.append(f"  - *{warning}*")
                else:
                    md.append("- **Observations:** No issues. Sub-questions are cohesive, clear, and distinct.")
                md.append("")

        md.append("## Question-by-Question Evaluation Details")

        q_ids = [item["id"] for item in details["phrasing_similarity"]]

        for i, q_id in enumerate(q_ids):
            md.append(f"### Question {q_id}")
            phrasing_item = details["phrasing_similarity"][i]
            cmd_item = details["command_word_alignment"][i]
            spec_item = details["spec_relevance"][i]
            qual_item = details["qualitative"][i]

            md.append(f"**Question Text:** *\"{phrasing_item['question']}\"*")
            md.append(f"- **Allocated Marks:** {cmd_item['marks']}")
            md.append(f"- **Extracted Command Word:** `{cmd_item['command_word']}`")
            md.append(f"- **Command Word Alignment Score:** {cmd_item['score']}/100 (*{cmd_item['reason']}*)")
            md.append(f"- **Phrasing Cosine Similarity:** {phrasing_item['max_similarity']} (Matched Past Paper Question: *\"{phrasing_item['matched_past_question']}\"*)")
            md.append(f"- **Specification Relevance Score:** {spec_item['score']}/100")
            md.append(f"  - *Reasoning:* {spec_item['reasoning']}")
            md.append(f"- **Qualitative Evaluation Score:** {qual_item['overall_score']}/100")
            md.append(f"  - *Clarity:* {qual_item['scores']['clarity']}/100 | *Accuracy:* {qual_item['scores']['accuracy']}/100 | *Difficulty:* {qual_item['scores']['difficulty']}/100 | *Mark Alignment:* {qual_item['scores']['mark_alignment']}/100")
            md.append(f"  - *Feedback:* {qual_item['feedback']}")
            md.append("")

        return "\n".join(md)
