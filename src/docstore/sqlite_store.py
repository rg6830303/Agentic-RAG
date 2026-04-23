from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.config.settings import AppSettings
from src.utils.models import CheckpointRecord, ChunkRecord, LoadedDocument


class SQLiteDocstore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.path = settings.sqlite_path
        self.settings.ensure_directories()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    chunking_method TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    text TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    chunking_method TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    page_number INTEGER,
                    parent_chunk_id TEXT,
                    level INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    notes TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_file_name ON chunks(file_name);
                CREATE INDEX IF NOT EXISTS idx_chunks_parent_chunk_id ON chunks(parent_chunk_id);
                """
            )

    def upsert_document(self, document: LoadedDocument, chunks: list[ChunkRecord]) -> None:
        chunking_method = chunks[0].chunking_method if chunks else "unknown"
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE file_path = ?", (document.file_path,))
            connection.execute("DELETE FROM chunks WHERE file_path = ?", (document.file_path,))
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, file_path, file_name, extension, checksum, ingested_at, chunking_method, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.file_path,
                    document.file_name,
                    document.extension,
                    document.checksum,
                    document.ingested_at,
                    chunking_method,
                    json.dumps(document.metadata),
                ),
            )
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, file_path, file_name, extension, text, ordinal, checksum,
                    created_at, chunking_method, token_count, char_count, page_number, parent_chunk_id,
                    level, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.file_path,
                        chunk.file_name,
                        chunk.extension,
                        chunk.text,
                        chunk.ordinal,
                        chunk.checksum,
                        chunk.created_at,
                        chunk.chunking_method,
                        chunk.token_count,
                        chunk.char_count,
                        chunk.page_number,
                        chunk.parent_chunk_id,
                        chunk.level,
                        json.dumps(chunk.metadata),
                    )
                    for chunk in chunks
                ],
            )
        self.export_chunks_jsonl()

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.*, COUNT(c.chunk_id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                GROUP BY d.document_id
                ORDER BY d.file_name
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self, file_name: str | None = None, limit: int | None = None) -> list[ChunkRecord]:
        query = "SELECT * FROM chunks"
        params: list[Any] = []
        if file_name:
            query += " WHERE file_name = ?"
            params.append(file_name)
        query += " ORDER BY file_name, level, ordinal"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                tuple(chunk_ids),
            ).fetchall()
        chunk_map = {row["chunk_id"]: self._row_to_chunk(row) for row in rows}
        return [chunk_map[chunk_id] for chunk_id in chunk_ids if chunk_id in chunk_map]

    def get_parent_chunk(self, chunk: ChunkRecord) -> ChunkRecord | None:
        if not chunk.parent_chunk_id:
            return None
        return self.get_chunk(chunk.parent_chunk_id)

    def get_children(self, parent_chunk_id: str) -> list[ChunkRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE parent_chunk_id = ? ORDER BY ordinal",
                (parent_chunk_id,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def all_chunks(self) -> list[ChunkRecord]:
        return self.list_chunks()

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            doc_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            parent_count = connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE level = 0"
            ).fetchone()[0]
            child_count = connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE level > 0"
            ).fetchone()[0]
            file_rows = connection.execute(
                """
                SELECT file_name, chunking_method, COUNT(*) AS chunk_count
                FROM chunks
                GROUP BY file_name, chunking_method
                ORDER BY file_name
                """
            ).fetchall()
        return {
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "parent_chunk_count": parent_count,
            "child_chunk_count": child_count,
            "files": [dict(row) for row in file_rows],
            "sqlite_path": str(self.path),
        }

    def remove_files(self, file_paths: list[str]) -> None:
        if not file_paths:
            return
        placeholders = ", ".join("?" for _ in file_paths)
        with self._connect() as connection:
            connection.execute(
                f"DELETE FROM chunks WHERE file_path IN ({placeholders})",
                tuple(file_paths),
            )
            connection.execute(
                f"DELETE FROM documents WHERE file_path IN ({placeholders})",
                tuple(file_paths),
            )
        self.export_chunks_jsonl()

    def clear_all(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM checkpoints")
        self.export_chunks_jsonl()

    def persist_checkpoint(self, checkpoint: CheckpointRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO checkpoints (
                    checkpoint_id, stage, status, created_at, payload_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.stage,
                    checkpoint.status,
                    checkpoint.created_at,
                    json.dumps(checkpoint.payload),
                    checkpoint.notes,
                ),
            )

    def update_checkpoint_status(self, checkpoint_id: str, status: str, notes: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE checkpoints SET status = ?, notes = COALESCE(?, notes) WHERE checkpoint_id = ?",
                (status, notes, checkpoint_id),
            )

    def list_checkpoints(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM checkpoints ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def export_chunks_jsonl(self) -> None:
        chunks = [asdict(chunk) for chunk in self.all_chunks()]
        with self.settings.chunk_artifact_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=True) + "\n")

    def _row_to_chunk(self, row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            file_path=row["file_path"],
            file_name=row["file_name"],
            extension=row["extension"],
            text=row["text"],
            ordinal=row["ordinal"],
            checksum=row["checksum"],
            created_at=row["created_at"],
            chunking_method=row["chunking_method"],
            token_count=row["token_count"],
            char_count=row["char_count"],
            page_number=row["page_number"],
            parent_chunk_id=row["parent_chunk_id"],
            level=row["level"],
            metadata=json.loads(row["metadata_json"]),
        )
