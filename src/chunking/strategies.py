from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.utils.hashing import checksum_text
from src.utils.models import ChunkRecord, ChunkingStrategy, LoadedDocument
from src.utils.text import (
    approximate_token_count,
    delimiter_profile,
    heading_density,
    line_density,
    normalize_text,
    split_paragraphs,
    split_sentences,
)
from src.utils.time import utc_now_iso


@dataclass(slots=True)
class StrategySelection:
    strategy: str
    reason: str
    heuristics: dict[str, float | int | str]


class ChunkingService:
    def __init__(self, fixed_size: int = 1200, overlap: int = 150) -> None:
        self.fixed_size = fixed_size
        self.overlap = overlap

    def auto_select(self, document: LoadedDocument) -> StrategySelection:
        text = document.full_text
        profile = delimiter_profile(text)
        heuristics: dict[str, float | int | str] = {
            "heading_density": round(heading_density(text), 4),
            "line_density": round(line_density(text), 2),
            "blank_lines": profile["blank_lines"],
            "bullet_lines": profile["bullet_lines"],
            "commas": profile["commas"],
            "char_count": len(text),
            "extension": document.extension,
        }
        if document.extension in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".c", ".cs", ".go", ".rs", ".sql"}:
            return StrategySelection(
                strategy=ChunkingStrategy.RECURSIVE.value,
                reason="Code-like structure benefits from delimiter-aware recursive chunking.",
                heuristics=heuristics,
            )
        if document.extension in {".csv", ".json", ".xml", ".yaml", ".yml"}:
            return StrategySelection(
                strategy=ChunkingStrategy.FIXED.value,
                reason="Tabular or structured data is more stable with fixed chunk windows.",
                heuristics=heuristics,
            )
        if heuristics["heading_density"] > 0.08 or heuristics["blank_lines"] > 10:
            return StrategySelection(
                strategy=ChunkingStrategy.HIERARCHICAL.value,
                reason="Document shape suggests parent/child sections and coherent hierarchical retrieval.",
                heuristics=heuristics,
            )
        if heuristics["line_density"] < 50:
            return StrategySelection(
                strategy=ChunkingStrategy.SEMANTIC.value,
                reason="Shorter lines and dense sentence boundaries suit semantic grouping.",
                heuristics=heuristics,
            )
        return StrategySelection(
            strategy=ChunkingStrategy.ADAPTIVE.value,
            reason="The document is long-form and mixed-format, so adaptive chunking is appropriate.",
            heuristics=heuristics,
        )

    def chunk_document(
        self,
        document: LoadedDocument,
        requested_strategy: str,
        auto_mode: bool = False,
    ) -> tuple[list[ChunkRecord], StrategySelection]:
        selection = (
            self.auto_select(document)
            if auto_mode or requested_strategy == ChunkingStrategy.AUTO.value
            else StrategySelection(
                strategy=requested_strategy,
                reason="Manual strategy selected in the UI.",
                heuristics={"extension": document.extension, "char_count": len(document.full_text)},
            )
        )
        strategy = selection.strategy
        if strategy == ChunkingStrategy.FIXED.value:
            chunks = self._fixed_chunks(document, target_size=self.fixed_size)
        elif strategy == ChunkingStrategy.SEMANTIC.value:
            chunks = self._semantic_chunks(document)
        elif strategy == ChunkingStrategy.RECURSIVE.value:
            chunks = self._recursive_chunks(document)
        elif strategy == ChunkingStrategy.ADAPTIVE.value:
            chunks = self._adaptive_chunks(document)
        elif strategy == ChunkingStrategy.HIERARCHICAL.value:
            chunks = self._hierarchical_chunks(document)
        else:
            chunks = self._fixed_chunks(document, target_size=self.fixed_size)
        for chunk in chunks:
            chunk.metadata.setdefault("strategy_reason", selection.reason)
            chunk.metadata.setdefault("strategy_heuristics", selection.heuristics)
        return chunks, selection

    def _yield_section_units(
        self, document: LoadedDocument, mode: str
    ) -> Iterable[tuple[str, int | None]]:
        for section in document.sections:
            text = normalize_text(section.text)
            if not text:
                continue
            if mode == "paragraphs":
                pieces = split_paragraphs(text)
            elif mode == "sentences":
                pieces = split_sentences(text)
            else:
                pieces = [text]
            for piece in pieces:
                normalized = normalize_text(piece)
                if normalized:
                    yield normalized, section.page_number

    def _fixed_chunks(self, document: LoadedDocument, target_size: int = 1200) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        buffer = ""
        page_number: int | None = None
        ordinal = 0
        for piece, section_page in self._yield_section_units(document, mode="paragraphs"):
            if not buffer:
                page_number = section_page
            candidate = f"{buffer}\n\n{piece}".strip() if buffer else piece
            if len(candidate) <= target_size:
                buffer = candidate
                continue
            if buffer:
                chunks.append(
                    self._make_chunk(
                        document=document,
                        text=buffer,
                        ordinal=ordinal,
                        method=ChunkingStrategy.FIXED.value,
                        page_number=page_number,
                    )
                )
                ordinal += 1
            tail = buffer[-self.overlap :] if buffer else ""
            buffer = normalize_text(f"{tail}\n{piece}")
            page_number = section_page
        if buffer:
            chunks.append(
                self._make_chunk(
                    document=document,
                    text=buffer,
                    ordinal=ordinal,
                    method=ChunkingStrategy.FIXED.value,
                    page_number=page_number,
                )
            )
        return chunks

    def _recursive_chunks(self, document: LoadedDocument, target_size: int = 1200) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        ordinal = 0
        for section in document.sections:
            for piece in self._split_recursive(normalize_text(section.text), target_size):
                chunks.append(
                    self._make_chunk(
                        document=document,
                        text=piece,
                        ordinal=ordinal,
                        method=ChunkingStrategy.RECURSIVE.value,
                        page_number=section.page_number,
                    )
                )
                ordinal += 1
        return chunks

    def _semantic_chunks(self, document: LoadedDocument, target_size: int = 1000) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        ordinal = 0
        current = ""
        current_page: int | None = None
        for sentence, page_number in self._yield_section_units(document, mode="sentences"):
            if not current:
                current_page = page_number
            if len(current) + len(sentence) + 1 <= target_size:
                current = f"{current} {sentence}".strip()
                continue
            chunks.append(
                self._make_chunk(
                    document=document,
                    text=current,
                    ordinal=ordinal,
                    method=ChunkingStrategy.SEMANTIC.value,
                    page_number=current_page,
                )
            )
            ordinal += 1
            current = sentence
            current_page = page_number
        if current:
            chunks.append(
                self._make_chunk(
                    document=document,
                    text=current,
                    ordinal=ordinal,
                    method=ChunkingStrategy.SEMANTIC.value,
                    page_number=current_page,
                )
            )
        return chunks

    def _adaptive_chunks(self, document: LoadedDocument) -> list[ChunkRecord]:
        text = document.full_text
        if heading_density(text) > 0.1:
            chunks = self._recursive_chunks(document, target_size=1100)
            for chunk in chunks:
                chunk.chunking_method = ChunkingStrategy.ADAPTIVE.value
                chunk.metadata["adaptive_base_strategy"] = ChunkingStrategy.RECURSIVE.value
            return chunks
        if line_density(text) < 55:
            chunks = self._semantic_chunks(document, target_size=900)
            for chunk in chunks:
                chunk.chunking_method = ChunkingStrategy.ADAPTIVE.value
                chunk.metadata["adaptive_base_strategy"] = ChunkingStrategy.SEMANTIC.value
            return chunks
        chunks = self._fixed_chunks(document, target_size=1300)
        for chunk in chunks:
            chunk.chunking_method = ChunkingStrategy.ADAPTIVE.value
            chunk.metadata["adaptive_base_strategy"] = ChunkingStrategy.FIXED.value
        return chunks

    def _hierarchical_chunks(self, document: LoadedDocument) -> list[ChunkRecord]:
        parents = self._recursive_chunks(document, target_size=2200)
        output: list[ChunkRecord] = []
        child_ordinal = 0
        for parent_index, parent in enumerate(parents):
            parent.chunking_method = ChunkingStrategy.HIERARCHICAL.value
            parent.level = 0
            parent.metadata["hierarchy_role"] = "parent"
            parent.ordinal = parent_index
            output.append(parent)
            child_slices = self._split_recursive(parent.text, 850)
            for child_text in child_slices:
                child = self._make_chunk(
                    document=document,
                    text=child_text,
                    ordinal=child_ordinal,
                    method=ChunkingStrategy.HIERARCHICAL.value,
                    page_number=parent.page_number,
                    parent_chunk_id=parent.chunk_id,
                    level=1,
                    extra_metadata={"hierarchy_role": "child"},
                )
                output.append(child)
                child_ordinal += 1
        return output

    def _split_recursive(self, text: str, target_size: int) -> list[str]:
        normalized = normalize_text(text)
        if len(normalized) <= target_size:
            return [normalized] if normalized else []
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
        for separator in separators:
            if separator in normalized:
                parts = normalized.split(separator)
                chunks: list[str] = []
                current = ""
                for part in parts:
                    piece = part.strip()
                    if not piece:
                        continue
                    candidate = f"{current}{separator}{piece}".strip(separator).strip()
                    if len(candidate) <= target_size or not current:
                        current = candidate
                    else:
                        chunks.extend(self._split_recursive(current, target_size))
                        current = piece
                if current:
                    chunks.extend(self._split_recursive(current, target_size))
                if chunks:
                    return chunks
        return [normalized[index : index + target_size] for index in range(0, len(normalized), target_size)]

    def _make_chunk(
        self,
        document: LoadedDocument,
        text: str,
        ordinal: int,
        method: str,
        page_number: int | None = None,
        parent_chunk_id: str | None = None,
        level: int = 0,
        extra_metadata: dict[str, object] | None = None,
    ) -> ChunkRecord:
        normalized = normalize_text(text)
        created_at = utc_now_iso()
        chunk_id = checksum_text(
            f"{document.document_id}:{ordinal}:{level}:{parent_chunk_id or ''}:{normalized[:120]}"
        )[:24]
        metadata = extra_metadata.copy() if extra_metadata else {}
        metadata["document_checksum"] = document.checksum
        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=document.document_id,
            file_path=document.file_path,
            file_name=document.file_name,
            extension=document.extension,
            text=normalized,
            ordinal=ordinal,
            checksum=checksum_text(normalized),
            created_at=created_at,
            chunking_method=method,
            token_count=approximate_token_count(normalized),
            char_count=len(normalized),
            page_number=page_number,
            parent_chunk_id=parent_chunk_id,
            level=level,
            metadata=metadata,
        )
