from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.docstore.sqlite_store import SQLiteDocstore
from src.indexing.manager import IndexManager
from src.reranking.heuristic import HeuristicReranker
from src.utils.models import QueryOptions, RetrievalHit, RetrievalMode
from src.utils.text import sentence_relevance


class RetrievalEngine:
    def __init__(
        self,
        docstore: SQLiteDocstore,
        index_manager: IndexManager,
        reranker: HeuristicReranker | None = None,
    ) -> None:
        self.docstore = docstore
        self.index_manager = index_manager
        self.reranker = reranker or HeuristicReranker()

    def retrieve(self, query: str, options: QueryOptions) -> list[RetrievalHit]:
        branches: dict[str, list[tuple]] = {}

        def run_vector() -> list[tuple]:
            return self.index_manager.search_vector(query, top_k=max(options.top_k * 2, 6))

        def run_bm25() -> list[tuple]:
            return self.index_manager.search_bm25(query, top_k=max(options.top_k * 2, 6))

        futures = {}
        if options.parallel_enabled and (options.use_vector or options.use_bm25):
            with ThreadPoolExecutor(max_workers=2) as executor:
                if options.use_vector:
                    futures["vector"] = executor.submit(run_vector)
                if options.use_bm25:
                    futures["bm25"] = executor.submit(run_bm25)
                for name, future in futures.items():
                    branches[name] = future.result()
        else:
            if options.use_vector:
                branches["vector"] = run_vector()
            if options.use_bm25:
                branches["bm25"] = run_bm25()

        merged = self._merge_results(branches)
        if options.retrieval_mode == RetrievalMode.HIERARCHICAL.value:
            merged = self._expand_hierarchy(merged)
        if options.sentence_attention:
            for hit in merged:
                hit.sentence_attention = sentence_relevance(query, hit.text, limit=5)
        if options.use_reranking and merged:
            merged = self.reranker.rerank(query, merged, options.top_k)
        else:
            merged = merged[: options.top_k]
        for rank, hit in enumerate(merged, start=1):
            hit.rank = rank
        return merged

    def _merge_results(self, branches: dict[str, list[tuple]]) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        for source, results in branches.items():
            if not results:
                continue
            max_score = max(score for _, score in results) or 1.0
            for rank, (chunk, score) in enumerate(results, start=1):
                normalized = float(score) / max_score
                weight = 0.6 if source == "vector" else 0.4
                blended = normalized * weight + (1 / (rank + 1)) * 0.05
                if chunk.chunk_id not in merged:
                    merged[chunk.chunk_id] = RetrievalHit(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        file_name=chunk.file_name,
                        file_path=chunk.file_path,
                        text=chunk.text,
                        score=round(blended, 6),
                        source=source,
                        rank=rank,
                        chunking_method=chunk.chunking_method,
                        page_number=chunk.page_number,
                        parent_chunk_id=chunk.parent_chunk_id,
                        level=chunk.level,
                        metadata=dict(chunk.metadata),
                    )
                    merged[chunk.chunk_id].metadata["source_branches"] = [source]
                else:
                    merged[chunk.chunk_id].score = round(merged[chunk.chunk_id].score + blended, 6)
                    merged[chunk.chunk_id].metadata.setdefault("source_branches", []).append(source)
        ordered = sorted(merged.values(), key=lambda hit: hit.score, reverse=True)
        return ordered

    def _expand_hierarchy(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        enriched: list[RetrievalHit] = []
        seen = set()
        for hit in hits:
            if hit.chunk_id not in seen:
                enriched.append(hit)
                seen.add(hit.chunk_id)
            if hit.parent_chunk_id and hit.parent_chunk_id not in seen:
                parent = self.docstore.get_chunk(hit.parent_chunk_id)
                if parent:
                    enriched.append(
                        RetrievalHit(
                            chunk_id=parent.chunk_id,
                            document_id=parent.document_id,
                            file_name=parent.file_name,
                            file_path=parent.file_path,
                            text=parent.text,
                            score=round(hit.score * 0.92, 6),
                            source="hierarchical_parent",
                            rank=hit.rank,
                            chunking_method=parent.chunking_method,
                            page_number=parent.page_number,
                            parent_chunk_id=parent.parent_chunk_id,
                            level=parent.level,
                            metadata={**parent.metadata, "expanded_from": hit.chunk_id},
                        )
                    )
                    seen.add(parent.chunk_id)
            if hit.level == 0:
                for child in self.docstore.get_children(hit.chunk_id)[:2]:
                    if child.chunk_id in seen:
                        continue
                    enriched.append(
                        RetrievalHit(
                            chunk_id=child.chunk_id,
                            document_id=child.document_id,
                            file_name=child.file_name,
                            file_path=child.file_path,
                            text=child.text,
                            score=round(hit.score * 0.88, 6),
                            source="hierarchical_child",
                            rank=hit.rank,
                            chunking_method=child.chunking_method,
                            page_number=child.page_number,
                            parent_chunk_id=child.parent_chunk_id,
                            level=child.level,
                            metadata={**child.metadata, "expanded_from": hit.chunk_id},
                        )
                    )
                    seen.add(child.chunk_id)
        enriched.sort(key=lambda item: item.score, reverse=True)
        return enriched
