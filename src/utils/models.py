from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"
    AUTO = "auto"


class RetrievalMode(str, Enum):
    FIXED = "fixed"
    HIERARCHICAL = "hierarchical"


class CheckpointStatus(str, Enum):
    AUTO_APPROVED = "auto_approved"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


@dataclass(slots=True)
class DocumentSection:
    section_id: str
    text: str
    page_number: int | None = None
    heading: str | None = None


@dataclass(slots=True)
class LoadedDocument:
    document_id: str
    file_path: str
    file_name: str
    extension: str
    checksum: str
    ingested_at: str
    sections: list[DocumentSection]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(section.text for section in self.sections if section.text)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    file_path: str
    file_name: str
    extension: str
    text: str
    ordinal: int
    checksum: str
    created_at: str
    chunking_method: str
    token_count: int
    char_count: int
    page_number: int | None = None
    parent_chunk_id: str | None = None
    level: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    file_name: str
    file_path: str
    text: str
    score: float
    source: str
    rank: int
    chunking_method: str
    page_number: int | None = None
    parent_chunk_id: str | None = None
    level: int = 0
    sentence_attention: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CheckpointRecord:
    checkpoint_id: str
    stage: str
    status: str
    created_at: str
    payload: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None


@dataclass(slots=True)
class QueryOptions:
    top_k: int = 6
    use_vector: bool = True
    use_bm25: bool = True
    use_reranking: bool = True
    self_rag: bool = True
    checkpoints_enabled: bool = True
    require_context_review: bool = False
    require_final_approval: bool = False
    evaluation_enabled: bool = False
    sentence_attention: bool = True
    citation_display: bool = True
    retrieval_mode: str = RetrievalMode.FIXED.value
    parallel_enabled: bool = True


@dataclass(slots=True)
class GuardrailResult:
    passed: bool
    confidence: float
    citation_coverage: float
    retrieval_floor_met: bool
    risk_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnswerBundle:
    question: str
    answer: str
    citations: list[RetrievalHit]
    used_methods: list[str]
    retrieval_mode: str
    self_rag_enabled: bool
    confidence: float
    needs_review: bool
    guardrails: GuardrailResult
    reflection: str | None = None
    evaluation_summary: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[CheckpointRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationSample:
    sample_id: str
    question: str
    reference_answer: str
    reference_contexts: list[str]
    expected_files: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationReport:
    report_id: str
    created_at: str
    mode: str
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    artifacts: dict[str, str]
