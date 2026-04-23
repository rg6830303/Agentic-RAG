from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from src.config.settings import AppSettings
from src.utils.text import tokenize


@dataclass(slots=True)
class BM25SearchResult:
    chunk_id: str
    score: float


class BM25Store:
    def __init__(self, settings: AppSettings, k1: float = 1.5, b: float = 0.75) -> None:
        self.settings = settings
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.postings: dict[str, list[list[float | int]]] = {}

    def is_ready(self) -> bool:
        return self.settings.bm25_path.exists()

    def build(self, chunks: list[tuple[str, str]]) -> dict[str, object]:
        total_docs = len(chunks)
        if total_docs == 0:
            return {"ok": False, "message": "No chunks available for BM25 indexing."}
        self.chunk_ids = [chunk_id for chunk_id, _ in chunks]
        tokens_list = [tokenize(text) for _, text in chunks]
        self.doc_lengths = [len(tokens) for tokens in tokens_list]
        self.avg_doc_length = sum(self.doc_lengths) / max(total_docs, 1)

        document_frequency: Counter[str] = Counter()
        term_freqs: list[Counter[str]] = []
        for tokens in tokens_list:
            counts = Counter(tokens)
            term_freqs.append(counts)
            document_frequency.update(counts.keys())

        postings: dict[str, list[list[float | int]]] = defaultdict(list)
        for doc_index, counts in enumerate(term_freqs):
            for term, term_frequency in counts.items():
                postings[term].append([doc_index, term_frequency])

        payload = {
            "chunk_ids": self.chunk_ids,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "postings": postings,
            "document_frequency": dict(document_frequency),
            "k1": self.k1,
            "b": self.b,
        }
        self.settings.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.settings.bm25_path.write_text(json.dumps(payload), encoding="utf-8")
        self.postings = dict(postings)
        return {"ok": True, "message": f"Built BM25 index for {total_docs} chunks."}

    def load(self) -> None:
        if not self.is_ready():
            return
        payload = json.loads(self.settings.bm25_path.read_text(encoding="utf-8"))
        self.chunk_ids = payload.get("chunk_ids", [])
        self.doc_lengths = payload.get("doc_lengths", [])
        self.avg_doc_length = float(payload.get("avg_doc_length", 0.0))
        self.postings = payload.get("postings", {})
        self.k1 = float(payload.get("k1", self.k1))
        self.b = float(payload.get("b", self.b))

    def search(self, query: str, top_k: int = 6) -> list[BM25SearchResult]:
        if not self.postings and self.is_ready():
            self.load()
        if not self.postings:
            return []
        scores = defaultdict(float)
        total_docs = len(self.chunk_ids)
        query_terms = tokenize(query)
        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for doc_index, term_frequency in postings:
                doc_length = self.doc_lengths[int(doc_index)]
                numerator = float(term_frequency) * (self.k1 + 1)
                denominator = float(term_frequency) + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1.0)
                )
                scores[int(doc_index)] += idf * numerator / max(denominator, 1e-9)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            BM25SearchResult(chunk_id=self.chunk_ids[index], score=float(score))
            for index, score in ranked
        ]

    def delete(self) -> None:
        if self.settings.bm25_path.exists():
            self.settings.bm25_path.unlink()
        self.chunk_ids = []
        self.doc_lengths = []
        self.avg_doc_length = 0.0
        self.postings = {}
