from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.config.settings import AppSettings
from src.docstore.sqlite_store import SQLiteDocstore
from src.indexing.bm25_store import BM25Store
from src.indexing.faiss_store import FaissStore
from src.providers.azure_openai import AzureOpenAIProvider, ProviderError
from src.utils.models import ChunkRecord


@dataclass(slots=True)
class IndexStatus:
    faiss_ready: bool
    bm25_ready: bool
    faiss_available: bool
    embeddings_available: bool
    chunk_count: int


class IndexManager:
    def __init__(
        self,
        settings: AppSettings,
        provider: AzureOpenAIProvider,
        docstore: SQLiteDocstore,
        faiss_store: FaissStore | None = None,
        bm25_store: BM25Store | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.docstore = docstore
        self.faiss_store = faiss_store or FaissStore(settings)
        self.bm25_store = bm25_store or BM25Store(settings)

    def status(self) -> IndexStatus:
        chunks = self.docstore.all_chunks()
        return IndexStatus(
            faiss_ready=self.faiss_store.is_ready(),
            bm25_ready=self.bm25_store.is_ready(),
            faiss_available=self.faiss_store.faiss_available(),
            embeddings_available=self.settings.embeddings_available,
            chunk_count=len(chunks),
        )

    def rebuild_all(
        self,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, object]:
        chunks = self.docstore.all_chunks()
        bm25_result = self.rebuild_bm25(chunks)
        if progress_callback:
            progress_callback(str(bm25_result["message"]))
        faiss_result = self.rebuild_faiss(chunks)
        if progress_callback:
            progress_callback(str(faiss_result["message"]))
        return {"bm25": bm25_result, "faiss": faiss_result}

    def rebuild_bm25(self, chunks: list[ChunkRecord] | None = None) -> dict[str, object]:
        chunk_records = chunks or self.docstore.all_chunks()
        pairs = [(chunk.chunk_id, chunk.text) for chunk in chunk_records]
        return self.bm25_store.build(pairs)

    def rebuild_faiss(
        self,
        chunks: list[ChunkRecord] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, object]:
        chunk_records = chunks or self.docstore.all_chunks()
        if not chunk_records:
            return {"ok": False, "message": "No chunks available for FAISS indexing."}
        if not self.settings.embeddings_available:
            return {
                "ok": False,
                "message": "Embeddings deployment is not configured, so FAISS cannot be rebuilt.",
            }
        if not self.faiss_store.faiss_available():
            return {"ok": False, "message": "faiss-cpu is not installed."}

        texts = [chunk.text for chunk in chunk_records]
        chunk_ids = [chunk.chunk_id for chunk in chunk_records]
        vectors: list[list[float]] = []
        batch_size = max(1, self.settings.embedding_batch_size)
        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                batch_vectors = self.provider.embed_texts(batch)
                vectors.extend(batch_vectors)
                if progress_callback:
                    progress_callback(min(start + batch_size, len(texts)), len(texts))
        except ProviderError as exc:
            return {"ok": False, "message": str(exc)}
        return self.faiss_store.build(chunk_ids, vectors)

    def search_vector(self, query: str, top_k: int = 6) -> list[tuple[ChunkRecord, float]]:
        if not self.settings.embeddings_available or not self.faiss_store.is_ready():
            return []
        try:
            embedding = self.provider.embed_texts([query])[0]
        except (IndexError, ProviderError):
            return []
        results = self.faiss_store.search(embedding, top_k=top_k)
        chunk_map = {chunk.chunk_id: chunk for chunk in self.docstore.get_chunks_by_ids([result.chunk_id for result in results])}
        return [
            (chunk_map[result.chunk_id], result.score)
            for result in results
            if result.chunk_id in chunk_map
        ]

    def search_bm25(self, query: str, top_k: int = 6) -> list[tuple[ChunkRecord, float]]:
        results = self.bm25_store.search(query, top_k=top_k)
        chunk_map = {chunk.chunk_id: chunk for chunk in self.docstore.get_chunks_by_ids([result.chunk_id for result in results])}
        return [
            (chunk_map[result.chunk_id], result.score)
            for result in results
            if result.chunk_id in chunk_map
        ]
