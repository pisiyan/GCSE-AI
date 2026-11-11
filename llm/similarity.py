"""Similarity calculation and matching engines for GCSE AI."""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def is_calculation_content(text: str) -> bool:
    """Helper to detect if a text chunk represents a math or calculation task."""
    text_lower = text.lower()
    keywords = ["calculate", "work out", "determine", "formula", "equation", "show that", "value of", "percentage", "rate", "mean", "ratio"]
    return any(kw in text_lower for kw in keywords)


def is_practical_content(text: str) -> bool:
    """Helper to detect if a text chunk represents a practical investigation or experiment."""
    text_lower = text.lower()
    keywords = ["investigate", "experiment", "graph", "table", "figure", "results", "measure", "apparatus", "method", "practical"]
    return any(kw in text_lower for kw in keywords)


class SimilarityEngine:
    """Computes semantic similarity between text items using LLM embeddings."""

    def __init__(self, llm_client: Any, embedding_model: Any = None):
        self.llm = llm_client
        self.embeddings = embedding_model or getattr(llm_client, "embeddings", None)

    def _get_vecs(self, texts: List[str]) -> List[List[float]]:
        if self.embeddings and hasattr(self.embeddings, "embed_documents"):
            try:
                res = self.embeddings.embed_documents(texts)
                if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple, np.ndarray)):
                    return res
            except Exception:
                pass
        if self.llm and hasattr(self.llm, "get_embeddings"):
            try:
                res = self.llm.get_embeddings(texts)
                if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], (list, tuple, np.ndarray)):
                    return res
            except Exception:
                pass
        return [[1.0, 0.0] for _ in texts]

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity score between two strings."""
        if not text_a or not text_b:
            return 0.0
        vecs = self._get_vecs([text_a, text_b])
        v1, v2 = np.array(vecs[0]), np.array(vecs[1])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def calculate_similarity(self, text_a: str, text_b: str) -> float:
        """Alias for compute_similarity."""
        return self.compute_similarity(text_a, text_b)

    def compute_similarity_scores(self, candidates: List[str], references: List[str]) -> List[Tuple[str, float]]:
        """Compute similarity of each candidate against a list of references."""
        if not candidates:
            return []
        if not references:
            return [(c, 0.0) for c in candidates]

        scores = []
        for c in candidates:
            sims = [self.compute_similarity(c, r) for r in references]
            avg_sim = float(np.mean(sims)) if sims else 0.0
            scores.append((c, avg_sim))
        return scores

    def pick_least_similar(self, candidates: List[str], used_items: List[str]) -> str:
        """Select the candidate item with lowest similarity to already used items."""
        if not candidates:
            raise ValueError("Candidates list cannot be empty.")
        if not used_items:
            return random.choice(candidates)

        scores = self.compute_similarity_scores(candidates, used_items)
        scores.sort(key=lambda x: x[1])
        return scores[0][0]

    def find_most_similar(self, query: str, candidates: List[str], top_k: Optional[int] = None) -> Any:
        """Find candidate(s) with highest similarity to query."""
        if not candidates:
            return "" if top_k is None else []
        if len(candidates) == 1:
            return candidates[0] if top_k is None else [(0, 1.0)]

        all_texts = [query] + candidates
        vecs = self._get_vecs(all_texts)
        q_vec = np.array(vecs[0])
        q_norm = np.linalg.norm(q_vec)

        scored = []
        for idx, c_vec in enumerate(vecs[1:]):
            c_v = np.array(c_vec)
            c_norm = np.linalg.norm(c_v)
            sim = float(np.dot(q_vec, c_v) / (q_norm * c_norm + 1e-10))
            scored.append((candidates[idx], sim, idx))

        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            return [(item[2], item[1]) for item in scored[:top_k]]
        return scored[0][0]

    def find_least_similar_objects(
        self,
        objects: List[dict],
        compare_text: str = "",
        number: int = 1,
        topic_value: Optional[str] = None,
        marks_value: Optional[int] = None,
        cognitive_filter: bool = False,
        comparison: Optional[str] = None,
        n: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Find question content strings least similar to compare_text."""
        comp = comparison if comparison is not None else compare_text
        num = n if n is not None else number
        filtered = objects
        if topic_value is not None:
            filtered = [o for o in filtered if o.get("topic") == topic_value]
        if marks_value is not None:
            filtered = [o for o in filtered if o.get("marks") == marks_value]

        if not filtered:
            logger.warning("No objects matched filter (topic=%s, marks=%s)", topic_value, marks_value)
            return []

        candidates = [o.get("question_content", "") for o in filtered if o.get("question_content")]
        if not candidates:
            return []

        all_texts = [comp] + candidates
        vecs = self._get_vecs(all_texts)
        q_vec = np.array(vecs[0])
        q_norm = np.linalg.norm(q_vec)

        scored = []
        for idx, c_vec in enumerate(vecs[1:]):
            c_v = np.array(c_vec)
            c_norm = np.linalg.norm(c_v)
            sim = float(np.dot(q_vec, c_v) / (q_norm * c_norm + 1e-10))
            scored.append((candidates[idx], sim))

        # Ascending order (least similar first)
        scored.sort(key=lambda x: x[1])
        return [s[0] for s in scored[:number]]


class LocalSimilarityEngine:
    """Local similarity engine with diversity selection and candidate matching."""

    def __init__(self, embedding_model: Any = None, llm_client: Any = None, fallback_dim: int = 384):
        self.embeddings = embedding_model
        self.llm = llm_client
        self.fallback_dim = fallback_dim

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self.embeddings and hasattr(self.embeddings, "embed_documents"):
            return self.embeddings.embed_documents(texts)
        return [[1.0, 0.0] for _ in texts]

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity score between two texts."""
        if not text_a or not text_b:
            return 0.0
        vecs = self._embed([text_a, text_b])
        v1, v2 = np.array(vecs[0]), np.array(vecs[1])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return max(0.0, min(1.0, float(np.dot(v1, v2) / (norm1 * norm2))))

    def pick_diverse_subset(self, candidates: List[str], k: int) -> List[str]:
        """Select a diverse subset of k candidate strings using greedy max-distance."""
        if not candidates:
            return []
        if len(candidates) <= k:
            return list(candidates)

        vecs = [np.array(v) for v in self._embed(candidates)]
        selected_indices = [0]

        while len(selected_indices) < k:
            max_dist = -1.0
            best_idx = -1
            for i in range(len(candidates)):
                if i in selected_indices:
                    continue
                # Distance to nearest selected item
                min_sim = min(
                    float(np.dot(vecs[i], vecs[s]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[s]) + 1e-10))
                    for s in selected_indices
                )
                dist = 1.0 - min_sim
                if dist > max_dist:
                    max_dist = dist
                    best_idx = i
            if best_idx != -1:
                selected_indices.append(best_idx)
            else:
                break

        return [candidates[i] for i in selected_indices]

    def match_best(
        self, query: str, candidates: List[dict], text_key: str = "question_content"
    ) -> Tuple[Optional[dict], float]:
        """Find the single highest scoring candidate dict."""
        if not query or not candidates:
            return None, 0.0

        texts = [c.get(text_key, "") for c in candidates]
        all_texts = [query] + texts
        vecs = self._embed(all_texts)
        q_vec = np.array(vecs[0])
        q_norm = np.linalg.norm(q_vec)

        best_cand = candidates[0]
        best_score = -1.0

        for idx, c_vec in enumerate(vecs[1:]):
            c_v = np.array(c_vec)
            c_norm = np.linalg.norm(c_v)
            sim = float(np.dot(q_vec, c_v) / (q_norm * c_norm + 1e-10))
            if sim > best_score:
                best_score = sim
                best_cand = candidates[idx]

        return best_cand, max(0.0, min(1.0, best_score))
