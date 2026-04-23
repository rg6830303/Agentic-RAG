from __future__ import annotations

from src.utils.models import RetrievalHit
from src.utils.text import word_overlap_score


class HeuristicReranker:
    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        rescored: list[RetrievalHit] = []
        for hit in hits:
            overlap = word_overlap_score(query, hit.text)
            length_bonus = min(len(hit.text) / 1200, 1.0) * 0.05
            hit.score = round(hit.score * 0.65 + overlap * 0.3 + length_bonus, 6)
            hit.metadata["rerank_overlap"] = round(overlap, 4)
            rescored.append(hit)
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]
