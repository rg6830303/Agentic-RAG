from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config.settings import AppSettings


@dataclass(slots=True)
class FaissSearchResult:
    chunk_id: str
    score: float


class FaissStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._index = None
        self._chunk_ids: list[str] = []

    @staticmethod
    def faiss_available() -> bool:
        try:
            import faiss  # noqa: F401
        except ImportError:
            return False
        return True

    def is_ready(self) -> bool:
        return self.settings.faiss_index_path.exists() and self.settings.faiss_meta_path.exists()

    def build(self, chunk_ids: list[str], embeddings: list[list[float]]) -> dict[str, object]:
        if not self.faiss_available():
            return {"ok": False, "message": "faiss-cpu is not installed."}
        if not chunk_ids or not embeddings:
            return {"ok": False, "message": "No chunk embeddings were provided."}
        import faiss

        matrix = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self.settings.indexes_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.settings.faiss_index_path))
        self.settings.faiss_meta_path.write_text(
            json.dumps({"chunk_ids": chunk_ids, "dimension": int(matrix.shape[1])}, indent=2),
            encoding="utf-8",
        )
        self._index = index
        self._chunk_ids = list(chunk_ids)
        return {"ok": True, "message": f"Built FAISS index for {len(chunk_ids)} chunks."}

    def load(self) -> None:
        if not self.is_ready() or not self.faiss_available():
            return
        import faiss

        self._index = faiss.read_index(str(self.settings.faiss_index_path))
        metadata = json.loads(self.settings.faiss_meta_path.read_text(encoding="utf-8"))
        self._chunk_ids = metadata.get("chunk_ids", [])

    def search(self, query_embedding: list[float], top_k: int = 6) -> list[FaissSearchResult]:
        if self._index is None:
            self.load()
        if self._index is None:
            return []
        import faiss

        query = np.asarray([query_embedding], dtype="float32")
        faiss.normalize_L2(query)
        scores, indexes = self._index.search(query, top_k)
        results: list[FaissSearchResult] = []
        for score, index in zip(scores[0], indexes[0], strict=False):
            if index < 0 or index >= len(self._chunk_ids):
                continue
            results.append(FaissSearchResult(chunk_id=self._chunk_ids[index], score=float(score)))
        return results

    def delete(self) -> None:
        for path in (self.settings.faiss_index_path, self.settings.faiss_meta_path):
            if path.exists():
                path.unlink()
        self._index = None
        self._chunk_ids = []
