import logging
import random
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class SimilarityEngine:
    """Efficient semantic similarity engine with batched API calls.
    
    Instead of computing embeddings one pair at a time (O(n) API calls),
    this batches all texts into 1-2 API calls and uses numpy for fast
    cosine similarity computation.
    """
    
    def __init__(self, llm_client, embedding_model=None):
        """Initialize with an LLMClient instance for getting embeddings."""
        self.llm_client = llm_client
        self.embedding_model = embedding_model

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using local model if available, falling back to API client."""
        if self.embedding_model is not None:
            try:
                return self.embedding_model.embed_documents(texts)
            except Exception as e:
                logger.warning("Local embedding model failed, falling back to API: %s", e)
        return self.llm_client.get_embeddings(texts)
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def compute_similarity(self, text_a: str, text_b: str, 
                          model: str = "text-embedding-3-small") -> float:
        """Compute cosine similarity between two texts (single batched call)."""
        embeddings = self._get_embeddings([text_a, text_b])
        emb_a = np.array(embeddings[0])
        emb_b = np.array(embeddings[1])
        return self._cosine_similarity(emb_a, emb_b)
    
    def compute_similarity_scores(
        self, 
        candidates: list[str], 
        references: list[str], 
        model: str = "text-embedding-3-small"
    ) -> list[tuple[str, float]]:
        """Compute max similarity of each candidate against all references.
        
        Uses BATCHED API calls — one for candidates, one for references.
        Returns list of (candidate_text, max_similarity_score) sorted descending.
        """
        if not candidates or not references:
            return [(c, 0.0) for c in candidates]
        
        # Two batched API calls instead of len(candidates) * len(references) calls
        emb_candidates = np.array(self._get_embeddings(candidates))
        emb_references = np.array(self._get_embeddings(references))
        
        # Normalize for cosine similarity
        norms_c = np.linalg.norm(emb_candidates, axis=1, keepdims=True)
        norms_r = np.linalg.norm(emb_references, axis=1, keepdims=True)
        emb_candidates_norm = emb_candidates / norms_c
        emb_references_norm = emb_references / norms_r
        
        # Similarity matrix: (num_candidates x num_references)
        sim_matrix = emb_candidates_norm @ emb_references_norm.T
        
        # Max similarity per candidate
        max_sims = np.max(sim_matrix, axis=1)
        
        scores = [(candidates[i], float(max_sims[i])) for i in range(len(candidates))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def pick_least_similar(self, candidates: list[str], used: list[str]) -> str:
        """Pick a random item from the least similar half of candidates vs used items.
        
        This encourages topic diversity by preferring candidates that are
        semantically distant from already-used items.
        """
        if not candidates:
            raise ValueError("No candidates to pick from")
        if len(candidates) == 1:
            return candidates[0]
        if not used:
            return random.choice(candidates)
        
        scores = self.compute_similarity_scores(candidates, used)
        
        # Take the lower half (least similar to used)
        half_index = len(scores) // 2
        lower_half = scores[half_index:]
        
        if not lower_half:
            return scores[-1][0]
        
        chosen = random.choice(lower_half)
        return chosen[0]
    
    def find_most_similar(self, query: str, candidates: list[str],
                         model: str = "text-embedding-3-small") -> str:
        """Find the most similar candidate to a query string."""
        if not candidates:
            raise ValueError("No candidates to search")
        if len(candidates) == 1:
            return candidates[0]
        
        all_texts = [query] + candidates
        embeddings = self._get_embeddings(all_texts)
        
        query_emb = np.array(embeddings[0])
        candidate_embs = np.array(embeddings[1:])
        
        # Compute similarities
        norms = np.linalg.norm(candidate_embs, axis=1)
        query_norm = np.linalg.norm(query_emb)
        similarities = candidate_embs @ query_emb / (norms * query_norm)
        
        best_idx = int(np.argmax(similarities))
        return candidates[best_idx]
    
    def find_least_similar_objects(
        self,
        objects: list[dict],
        comparison: str,
        n: int,
        topic_key: str = "topic",
        topic_value: str = "",
        marks_key: str = "marks",  
        marks_value: Optional[int] = None,
        content_key: str = "question_content"
    ) -> list[str]:
        """Filter objects by topic/marks, score by similarity, return n LEAST similar.
        
        Replaces the original get_random_objects() which made individual API calls.
        """
        # Filter by topic and marks
        filtered = []
        shuffled = objects.copy()
        random.shuffle(shuffled)
        
        for obj in shuffled:
            matches_topic = (not topic_value) or obj.get(topic_key) == topic_value
            matches_marks = (marks_value is None) or obj.get(marks_key) == marks_value
            if matches_topic and matches_marks:
                filtered.append(obj)
            if len(filtered) >= n * 2:  # take a reasonable subset
                break
        
        if not filtered:
            logger.warning("No objects matched filter (topic=%s, marks=%s)", topic_value, marks_value)
            return []
        
        # Batch compute similarities
        contents = [obj[content_key] for obj in filtered]
        all_texts = [comparison] + contents
        embeddings = self._get_embeddings(all_texts)
        
        comp_emb = np.array(embeddings[0])
        content_embs = np.array(embeddings[1:])
        
        norms = np.linalg.norm(content_embs, axis=1)
        comp_norm = np.linalg.norm(comp_emb)
        similarities = content_embs @ comp_emb / (norms * comp_norm)
        
        # Sort ascending (least similar first) and take n
        scored = list(zip(contents, similarities))
        scored.sort(key=lambda x: x[1])
        
        return [item[0] for item in scored[:n]]
