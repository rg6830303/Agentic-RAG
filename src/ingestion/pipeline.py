from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.checkpoints.service import CheckpointManager
from src.chunking.strategies import ChunkingService, StrategySelection
from src.docstore.sqlite_store import SQLiteDocstore
from src.indexing.manager import IndexManager
from src.ingestion.loaders import DocumentLoader, IngestionError
from src.utils.concurrency import parallel_map
from src.utils.models import CheckpointRecord, ChunkRecord, LoadedDocument


@dataclass(slots=True)
class PreparedIngestion:
    documents: list[LoadedDocument]
    chunks: list[ChunkRecord]
    selections: dict[str, StrategySelection]
    checkpoints: list[CheckpointRecord]
    skipped: list[str]


class IngestionService:
    def __init__(
        self,
        docstore: SQLiteDocstore,
        chunking_service: ChunkingService,
        index_manager: IndexManager,
        checkpoint_manager: CheckpointManager,
        max_workers: int = 4,
    ) -> None:
        self.docstore = docstore
        self.loader = DocumentLoader()
        self.chunking_service = chunking_service
        self.index_manager = index_manager
        self.checkpoint_manager = checkpoint_manager
        self.max_workers = max_workers

    def prepare(
        self,
        paths: list[Path],
        strategy: str,
        auto_mode: bool,
        checkpoints_enabled: bool,
        parallel_enabled: bool,
        progress_callback: Callable[[str], None] | None = None,
    ) -> PreparedIngestion:
        skipped: list[str] = []

        def load_path(path: Path) -> LoadedDocument | None:
            try:
                return self.loader.load(path)
            except (IngestionError, OSError) as exc:
                skipped.append(f"{path.name}: {exc}")
                return None

        loaded = parallel_map(
            paths,
            load_path,
            max_workers=self.max_workers,
            enabled=parallel_enabled,
        )
        documents = [document for document in loaded if document is not None]
        if progress_callback:
            progress_callback(f"Loaded {len(documents)} documents. Skipped {len(skipped)} files.")
        load_checkpoint = self.checkpoint_manager.create(
            stage="post-load_pre-chunk",
            payload={
                "document_count": len(documents),
                "skipped_count": len(skipped),
                "files": [document.file_name for document in documents],
            },
            enabled=checkpoints_enabled,
            requires_human=False,
        )

        chunk_records: list[ChunkRecord] = []
        selections: dict[str, StrategySelection] = {}
        for document in documents:
            chunks, selection = self.chunking_service.chunk_document(
                document,
                requested_strategy=strategy,
                auto_mode=auto_mode,
            )
            chunk_records.extend(chunks)
            selections[document.document_id] = selection

        if progress_callback:
            progress_callback(f"Prepared {len(chunk_records)} chunks for review.")
        chunk_checkpoint = self.checkpoint_manager.create(
            stage="post-chunk_pre-index",
            payload={
                "chunk_count": len(chunk_records),
                "strategies": {document.file_name: selections[document.document_id].strategy for document in documents},
            },
            enabled=checkpoints_enabled,
            requires_human=False,
        )
        return PreparedIngestion(
            documents=documents,
            chunks=chunk_records,
            selections=selections,
            checkpoints=[load_checkpoint, chunk_checkpoint],
            skipped=skipped,
        )

    def commit(
        self,
        prepared: PreparedIngestion,
        rebuild_indexes: bool,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, object]:
        for document in prepared.documents:
            doc_chunks = [chunk for chunk in prepared.chunks if chunk.document_id == document.document_id]
            self.docstore.upsert_document(document, doc_chunks)
        index_results = {}
        if rebuild_indexes:
            index_results = self.index_manager.rebuild_all(progress_callback=progress_callback)
        return {
            "document_count": len(prepared.documents),
            "chunk_count": len(prepared.chunks),
            "index_results": index_results,
            "skipped": prepared.skipped,
        }
