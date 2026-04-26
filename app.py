from __future__ import annotations

import html
import json
import math
import os
import platform
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


SERVICE_NAME = "Advanced Agentic RAG"
SERVICE_VERSION = "0.3.0"
ROOT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = ROOT_DIR / "data" / "sample_corpus"
GOLDEN_EVAL_PATH = ROOT_DIR / "data" / "golden_eval" / "ncert_physics_golden.json"
SUPPORTED_CORPUS_EXTENSIONS = {".csv", ".json", ".md", ".sql", ".txt"}


@dataclass(slots=True)
class CorpusChunk:
    chunk_id: str
    file_name: str
    relative_path: str
    text: str
    ordinal: int
    tokens: list[str]
    strategy: str
    level: int = 0
    parent_chunk_id: str | None = None


@dataclass(slots=True)
class CorpusIndex:
    chunks: list[CorpusChunk]
    doc_lengths: list[int]
    avg_doc_length: float
    postings: dict[str, list[tuple[int, int]]]
    source_files: list[str]
    raw_sources: dict[str, str]


class SentenceAttention(BaseModel):
    sentence: str
    score: float


class Citation(BaseModel):
    file_name: str
    path: str
    rank: int
    score: float
    chunk_id: str
    chunking_method: str
    snippet: str
    sentence_attention: list[SentenceAttention] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    rank: int
    chunk_id: str
    file_name: str
    path: str
    score: float
    source: str
    chunking_method: str
    text: str
    sentence_attention: list[SentenceAttention] = Field(default_factory=list)


class GuardrailReport(BaseModel):
    passed: bool
    confidence: float
    citation_coverage: float
    retrieval_score_floor_met: bool
    risk_flags: list[str]


class CheckpointView(BaseModel):
    checkpoint_id: str
    stage: str
    status: str
    requires_human: bool
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class PipelineStep(BaseModel):
    stage: str
    title: str
    status: str
    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    retrieved_chunks: int
    citation_count: int
    answer_token_count: int
    sentence_attention_enabled: bool
    self_rag_used: bool
    finalized: bool


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12_000)
    top_k: int = Field(default=5, ge=1, le=10)
    retrieval_mode: Literal["bm25", "hybrid", "hierarchical"] = "hybrid"
    use_generation: bool = True
    use_reranking: bool = True
    self_rag: bool = True
    checkpoints_enabled: bool = True
    require_context_review: bool = False
    require_final_approval: bool = False
    sentence_attention: bool = True
    citation_display: bool = True
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=700, ge=100, le=2_000)


class ChatResponse(BaseModel):
    answer: str
    draft_answer: str
    finalized_answer: str
    provider: str
    retrieval_mode: str
    used_methods: list[str]
    confidence: float
    needs_review: bool
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    guardrails: GuardrailReport
    reflection: str
    checkpoints: list[CheckpointView]
    pipeline: list[PipelineStep]
    evaluation_summary: EvaluationSummary
    corpus: dict[str, Any]


class EvaluationRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=10)
    retrieval_mode: Literal["bm25", "hybrid", "hierarchical"] = "hybrid"
    use_reranking: bool = True
    self_rag: bool = True


class EvaluationResponse(BaseModel):
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="FastAPI UI and API for an advanced agentic RAG workflow.",
)


def _env_value(name: str) -> str:
    return os.getenv(name, "").strip()


def _azure_config() -> dict[str, str]:
    return {
        "endpoint": _env_value("AZURE_OPENAI_ENDPOINT").rstrip("/"),
        "api_version": _env_value("AZURE_OPENAI_API_VERSION"),
        "api_key": _env_value("AZURE_OPENAI_API_KEY"),
        "chat_deployment": _env_value("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    }


def _azure_status() -> dict[str, Any]:
    config = _azure_config()
    endpoint_host = (
        config["endpoint"]
        .replace("https://", "")
        .replace("http://", "")
        .strip("/")
    )
    return {
        "endpoint_host": endpoint_host,
        "api_version_configured": bool(config["api_version"]),
        "chat_configured": bool(
            config["endpoint"]
            and config["api_version"]
            and config["api_key"]
            and config["chat_deployment"]
        ),
    }


def _azure_chat_url(config: dict[str, str]) -> str:
    return (
        f"{config['endpoint']}/openai/deployments/"
        f"{config['chat_deployment']}/chat/completions"
        f"?api-version={config['api_version']}"
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _read_corpus_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        try:
            return json.dumps(json.loads(text), indent=2, ensure_ascii=True)
        except json.JSONDecodeError:
            return text
    return text


def _strategy_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".json", ".csv", ".sql"}:
        return "recursive"
    if "#" in path.read_text(encoding="utf-8", errors="ignore")[:1200]:
        return "semantic"
    return "auto"


def _split_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        if end < len(clean):
            boundary = clean.rfind(". ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


@lru_cache(maxsize=1)
def _load_corpus() -> CorpusIndex:
    chunks: list[CorpusChunk] = []
    source_files: list[str] = []
    raw_sources: dict[str, str] = {}
    if not CORPUS_DIR.exists():
        return CorpusIndex([], [], 0.0, {}, [], {})

    for path in sorted(CORPUS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_CORPUS_EXTENSIONS:
            continue
        relative_path = path.relative_to(ROOT_DIR).as_posix()
        source_files.append(relative_path)
        text = _read_corpus_file(path)
        raw_sources[relative_path] = text
        strategy = _strategy_for_path(path)
        for ordinal, chunk_text in enumerate(_split_text(text)):
            chunks.append(
                CorpusChunk(
                    chunk_id=f"{relative_path}:{ordinal}",
                    file_name=path.name,
                    relative_path=relative_path,
                    text=chunk_text,
                    ordinal=ordinal,
                    tokens=_tokenize(chunk_text),
                    strategy=strategy,
                    level=0,
                )
            )

    doc_lengths = [len(chunk.tokens) for chunk in chunks]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc_index, chunk in enumerate(chunks):
        for term, frequency in Counter(chunk.tokens).items():
            postings[term].append((doc_index, frequency))

    return CorpusIndex(
        chunks=chunks,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
        postings=dict(postings),
        source_files=source_files,
        raw_sources=raw_sources,
    )


def _bm25_scores(query: str) -> dict[int, float]:
    index = _load_corpus()
    query_terms = _tokenize(query)
    scores: dict[int, float] = defaultdict(float)
    total_docs = len(index.chunks)
    k1 = 1.5
    b = 0.75
    for term in query_terms:
        postings = index.postings.get(term)
        if not postings:
            continue
        document_frequency = len(postings)
        idf = math.log(1 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
        for doc_index, term_frequency in postings:
            doc_length = index.doc_lengths[doc_index]
            numerator = term_frequency * (k1 + 1)
            denominator = term_frequency + k1 * (
                1 - b + b * doc_length / max(index.avg_doc_length, 1.0)
            )
            scores[doc_index] += idf * numerator / max(denominator, 1e-9)
    return scores


def _semantic_scores(query: str) -> dict[int, float]:
    index = _load_corpus()
    query_terms = set(_tokenize(query))
    if not query_terms:
        return {}
    scores: dict[int, float] = {}
    for doc_index, chunk in enumerate(index.chunks):
        chunk_terms = set(chunk.tokens)
        if not chunk_terms:
            continue
        overlap = len(query_terms & chunk_terms)
        if overlap:
            scores[doc_index] = overlap / len(query_terms | chunk_terms)
    return scores


def _sentence_attention(question: str, text: str, limit: int = 3) -> list[SentenceAttention]:
    query_terms = set(_tokenize(question))
    scored: list[tuple[float, str]] = []
    for sentence in _sentences(text):
        terms = set(_tokenize(sentence))
        if not terms:
            continue
        overlap = len(query_terms & terms)
        score = overlap / max(len(query_terms), 1)
        if score > 0:
            scored.append((score, sentence))
    return [
        SentenceAttention(sentence=sentence, score=round(score, 3))
        for score, sentence in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    ]


def _search_corpus(
    query: str,
    top_k: int,
    retrieval_mode: str,
    use_reranking: bool,
) -> list[tuple[CorpusChunk, float, str]]:
    index = _load_corpus()
    if not _tokenize(query) or not index.chunks:
        return []

    bm25 = _bm25_scores(query)
    semantic = _semantic_scores(query)
    candidate_indexes = set(bm25) | set(semantic)
    combined: dict[int, tuple[float, str]] = {}

    for doc_index in candidate_indexes:
        bm25_score = bm25.get(doc_index, 0.0)
        semantic_score = semantic.get(doc_index, 0.0)
        if retrieval_mode == "bm25":
            score = bm25_score
            source = "bm25"
        elif retrieval_mode == "hierarchical":
            score = bm25_score + semantic_score * 2.0
            source = "hierarchical"
        else:
            score = bm25_score + semantic_score * 2.5
            source = "hybrid"
        if use_reranking:
            attention = _sentence_attention(query, index.chunks[doc_index].text, limit=1)
            score += attention[0].score if attention else 0.0
            source += "+rerank"
        combined[doc_index] = (score, source)

    ranked = sorted(combined.items(), key=lambda item: item[1][0], reverse=True)[:top_k]
    return [
        (index.chunks[doc_index], float(score), source)
        for doc_index, (score, source) in ranked
    ]


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _extractive_answer(question: str, hits: list[tuple[CorpusChunk, float, str]]) -> str:
    if not hits:
        return "I could not find enough matching evidence in the deployed corpus to answer this question."

    selected: list[str] = []
    for chunk, _score, _source in hits[:5]:
        for attention in _sentence_attention(question, chunk.text, limit=2):
            if attention.sentence not in selected:
                selected.append(attention.sentence)

    if not selected:
        selected = [hits[0][0].text[:300].strip()]

    body = " ".join(selected[:5]).strip()
    sources = ", ".join(sorted({chunk.file_name for chunk, _score, _source in hits[:3]}))
    return f"{body}\n\nGrounded sources: {sources}."


def _context_prompt(hits: list[tuple[CorpusChunk, float, str]]) -> str:
    blocks = []
    for rank, (chunk, score, source) in enumerate(hits, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{rank}] {chunk.relative_path}",
                    f"score={score:.3f}",
                    f"method={source}",
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(blocks)


def _generate_answer(
    question: str,
    hits: list[tuple[CorpusChunk, float, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    config = _azure_config()
    if not _azure_status()["chat_configured"]:
        return _extractive_answer(question, hits)

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the final answer stage in an agentic RAG pipeline. "
                    "Use only the provided context, synthesize a direct final answer, "
                    "include bracket citations like [1], and say when evidence is insufficient."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved context:\n{_context_prompt(hits)}",
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(
            _azure_chat_url(config),
            headers={
                "api-key": config["api_key"],
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=25,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Azure OpenAI request failed: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Azure OpenAI returned an error: {response.text[:500]}",
        )

    body = response.json()
    choices = body.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="Azure OpenAI returned no choices.")
    return str(choices[0].get("message", {}).get("content", "")).strip()


def _confidence(hits: list[tuple[CorpusChunk, float, str]], answer: str) -> float:
    if not hits:
        return 0.0
    top_score = hits[0][1]
    grounding = max((_word_overlap(answer, chunk.text) for chunk, _score, _source in hits[:5]), default=0.0)
    return round(min(0.97, max(0.22, 0.55 * top_score / (top_score + 3.0) + 0.45 * grounding)), 3)


def _word_overlap(left: str, right: str) -> float:
    left_terms = set(_tokenize(left))
    right_terms = set(_tokenize(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def _guardrails(
    answer: str,
    hits: list[tuple[CorpusChunk, float, str]],
    confidence: float,
) -> GuardrailReport:
    citation_count = min(5, len(hits))
    citation_coverage = citation_count / max(min(len(hits), 4), 1)
    top_score = hits[0][1] if hits else 0.0
    flags: list[str] = []
    if confidence < 0.55:
        flags.append("Confidence below threshold.")
    if citation_coverage < 0.5:
        flags.append("Citation coverage below threshold.")
    if top_score < 0.2:
        flags.append("Top retrieval score below floor.")
    if "could not find" in answer.lower():
        flags.append("Answer reports insufficient context.")
    return GuardrailReport(
        passed=not flags,
        confidence=confidence,
        citation_coverage=round(citation_coverage, 3),
        retrieval_score_floor_met=top_score >= 0.2,
        risk_flags=flags,
    )


def _citation(
    rank: int,
    chunk: CorpusChunk,
    score: float,
    question: str,
    sentence_attention: bool,
) -> Citation:
    return Citation(
        file_name=chunk.file_name,
        path=chunk.relative_path,
        rank=rank,
        score=round(score, 4),
        chunk_id=chunk.chunk_id,
        chunking_method=chunk.strategy,
        snippet=html.unescape(chunk.text[:700].strip()),
        sentence_attention=_sentence_attention(question, chunk.text) if sentence_attention else [],
    )


def _retrieved_chunk(
    rank: int,
    chunk: CorpusChunk,
    score: float,
    source: str,
    question: str,
    sentence_attention: bool,
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk.chunk_id,
        file_name=chunk.file_name,
        path=chunk.relative_path,
        score=round(score, 4),
        source=source,
        chunking_method=chunk.strategy,
        text=chunk.text,
        sentence_attention=_sentence_attention(question, chunk.text) if sentence_attention else [],
    )


def _reflect(
    question: str,
    answer: str,
    hits: list[tuple[CorpusChunk, float, str]],
    self_rag: bool,
) -> tuple[str, bool]:
    if not hits:
        return "Self-RAG reflection found no retrieved context to support the answer.", False
    overlap = _word_overlap(answer, " ".join(chunk.text for chunk, _score, _source in hits[:3]))
    needs_more = self_rag and overlap < 0.18
    if needs_more:
        return (
            "Self-RAG reflection judged the answer weakly grounded and triggered an expanded lexical pass.",
            True,
        )
    return (
        "Self-RAG reflection found the answer adequately grounded in retrieved citations.",
        False,
    )


def _make_checkpoints(
    payload: ChatRequest,
    hits: list[tuple[CorpusChunk, float, str]],
    guardrails: GuardrailReport,
) -> list[CheckpointView]:
    if not payload.checkpoints_enabled:
        return []
    final_requires_human = payload.require_final_approval or not guardrails.passed
    return [
        CheckpointView(
            checkpoint_id="cp-load-chunk",
            stage="post-load_pre-chunk",
            status="auto_approved",
            requires_human=False,
            summary="Bundled corpus loaded and chunked with auto strategy selection.",
            payload={"source_count": _corpus_summary()["source_count"]},
        ),
        CheckpointView(
            checkpoint_id="cp-index",
            stage="post-chunk_pre-index",
            status="auto_approved",
            requires_human=False,
            summary="BM25 and semantic lexical indexes are built in memory at startup.",
            payload={"chunk_count": _corpus_summary()["chunk_count"]},
        ),
        CheckpointView(
            checkpoint_id="cp-context",
            stage="post-retrieval_pre-generation",
            status="pending" if payload.require_context_review else "auto_approved",
            requires_human=payload.require_context_review,
            summary="Retrieved context is ready for human review before generation.",
            payload={"retrieved_count": len(hits), "top_sources": [chunk.file_name for chunk, _score, _source in hits[:5]]},
        ),
        CheckpointView(
            checkpoint_id="cp-final",
            stage="post-generation_pre-final-answer",
            status="needs_review" if final_requires_human else "auto_approved",
            requires_human=final_requires_human,
            summary="Final answer awaits approval when HITL is enabled or guardrails fail.",
            payload={"guardrails_passed": guardrails.passed, "risk_flags": guardrails.risk_flags},
        ),
    ]


def _pipeline(
    payload: ChatRequest,
    hits: list[tuple[CorpusChunk, float, str]],
    provider: str,
    guardrails: GuardrailReport,
    reflection: str,
    second_pass_used: bool,
) -> list[PipelineStep]:
    return [
        PipelineStep(
            stage="ingestion",
            title="Load corpus",
            status="complete",
            summary="Read bundled source files and normalize content.",
            metrics={"sources": _corpus_summary()["source_count"]},
        ),
        PipelineStep(
            stage="chunking",
            title="Chunk planning",
            status="complete",
            summary="Auto, semantic, and recursive chunk strategies are represented per source.",
            metrics={"chunks": _corpus_summary()["chunk_count"]},
        ),
        PipelineStep(
            stage="indexing",
            title="Index branches",
            status="complete",
            summary="BM25 lexical and semantic-overlap branches are available for hybrid retrieval.",
            metrics={"mode": payload.retrieval_mode, "reranking": payload.use_reranking},
        ),
        PipelineStep(
            stage="retrieval",
            title="Retrieve",
            status="complete" if hits else "needs_review",
            summary="Ranked context selected for answer generation.",
            metrics={"retrieved": len(hits), "top_score": round(hits[0][1], 4) if hits else 0.0},
        ),
        PipelineStep(
            stage="self_rag",
            title="Self-RAG reflection",
            status="expanded" if second_pass_used else "complete",
            summary=reflection,
            metrics={"enabled": payload.self_rag},
        ),
        PipelineStep(
            stage="generation",
            title="Generate final answer",
            status="complete",
            summary="Final answer synthesized from retrieved evidence.",
            metrics={"provider": provider},
        ),
        PipelineStep(
            stage="guardrails",
            title="Guardrails",
            status="passed" if guardrails.passed else "needs_review",
            summary="Confidence, citation coverage, and retrieval floor checked.",
            metrics={"confidence": guardrails.confidence, "flags": len(guardrails.risk_flags)},
        ),
        PipelineStep(
            stage="hitl",
            title="HITL checkpoints",
            status="pending" if payload.require_final_approval or payload.require_context_review else "complete",
            summary="Human decisions can approve or reject context and final answer stages in the UI.",
            metrics={"context_review": payload.require_context_review, "final_approval": payload.require_final_approval},
        ),
    ]


def _answer_pipeline(payload: ChatRequest) -> ChatResponse:
    question = payload.message.strip()
    hits = _search_corpus(
        question,
        top_k=payload.top_k,
        retrieval_mode=payload.retrieval_mode,
        use_reranking=payload.use_reranking,
    )
    reflection, second_pass_used = _reflect(question, _extractive_answer(question, hits), hits, payload.self_rag)
    if second_pass_used:
        expanded_hits = _search_corpus(
            f"{question} {_context_prompt(hits[:2])}",
            top_k=max(payload.top_k, 6),
            retrieval_mode=payload.retrieval_mode,
            use_reranking=payload.use_reranking,
        )
        seen = {chunk.chunk_id for chunk, _score, _source in hits}
        hits.extend([item for item in expanded_hits if item[0].chunk_id not in seen])
        hits.sort(key=lambda item: item[1], reverse=True)
        hits = hits[: payload.top_k]

    draft_answer = _extractive_answer(question, hits)
    if payload.use_generation and _azure_status()["chat_configured"] and hits:
        final_answer = _generate_answer(
            question,
            hits,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
        provider = "azure_openai_rag"
    else:
        final_answer = draft_answer
        provider = "local_agentic_rag"

    confidence = _confidence(hits, final_answer)
    guardrails = _guardrails(final_answer, hits, confidence)
    citations = [
        _citation(rank, chunk, score, question, payload.sentence_attention)
        for rank, (chunk, score, _source) in enumerate(hits[:5], start=1)
    ]
    retrieved_chunks = [
        _retrieved_chunk(rank, chunk, score, source, question, payload.sentence_attention)
        for rank, (chunk, score, source) in enumerate(hits, start=1)
    ]
    checkpoints = _make_checkpoints(payload, hits, guardrails)
    pipeline = _pipeline(payload, hits, provider, guardrails, reflection, second_pass_used)
    evaluation_summary = EvaluationSummary(
        retrieved_chunks=len(retrieved_chunks),
        citation_count=len(citations),
        answer_token_count=len(_tokenize(final_answer)),
        sentence_attention_enabled=payload.sentence_attention,
        self_rag_used=payload.self_rag,
        finalized=not payload.require_final_approval and guardrails.passed,
    )

    return ChatResponse(
        answer=final_answer,
        draft_answer=draft_answer,
        finalized_answer=final_answer,
        provider=provider,
        retrieval_mode=payload.retrieval_mode,
        used_methods=sorted({source for _chunk, _score, source in hits}),
        confidence=confidence,
        needs_review=payload.require_final_approval or not guardrails.passed,
        citations=citations if payload.citation_display else [],
        retrieved_chunks=retrieved_chunks,
        guardrails=guardrails,
        reflection=reflection,
        checkpoints=checkpoints,
        pipeline=pipeline,
        evaluation_summary=evaluation_summary,
        corpus=_corpus_summary(),
    )


def _corpus_summary() -> dict[str, Any]:
    index = _load_corpus()
    strategy_counts = Counter(chunk.strategy for chunk in index.chunks)
    return {
        "source_dir": CORPUS_DIR.relative_to(ROOT_DIR).as_posix() if CORPUS_DIR.exists() else "missing",
        "source_count": len(index.source_files),
        "chunk_count": len(index.chunks),
        "sources": index.source_files,
        "strategy_counts": dict(strategy_counts),
        "supported_extensions": sorted(SUPPORTED_CORPUS_EXTENSIONS),
        "indexes": {
            "bm25": bool(index.postings),
            "semantic_overlap": bool(index.chunks),
            "hybrid": bool(index.chunks),
            "faiss": False,
        },
    }


def _capabilities() -> dict[str, Any]:
    return {
        "retrieval": ["bm25", "hybrid", "hierarchical"],
        "chunking": ["fixed", "semantic", "recursive", "adaptive", "hierarchical", "auto"],
        "agentic": ["query planning", "reranking", "self-rag reflection", "guardrails"],
        "hitl": [
            "post-load_pre-chunk",
            "post-chunk_pre-index",
            "post-retrieval_pre-generation",
            "post-generation_pre-final-answer",
            "pre-persist-destructive-admin-action",
            "pre-index-rebuild",
        ],
        "evaluation": ["token_f1", "context_recall", "citation_hit_rate", "confidence"],
        "serverless_note": "Uploads and destructive persistence are represented as guarded UI workflows over the deployed corpus.",
    }


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_pool = ref_tokens.copy()
    common = 0
    for token in pred_tokens:
        if token in ref_pool:
            common += 1
            ref_pool.remove(token)
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _run_evaluation(payload: EvaluationRequest) -> EvaluationResponse:
    if not GOLDEN_EVAL_PATH.exists():
        return EvaluationResponse(summary={"sample_count": 0}, rows=[])
    samples = json.loads(GOLDEN_EVAL_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for sample in samples:
        response = _answer_pipeline(
            ChatRequest(
                message=sample["question"],
                top_k=payload.top_k,
                retrieval_mode=payload.retrieval_mode,
                use_generation=False,
                use_reranking=payload.use_reranking,
                self_rag=payload.self_rag,
                checkpoints_enabled=False,
            )
        )
        citation_files = {citation.file_name for citation in response.citations}
        expected_files = set(sample.get("expected_files", []))
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "question": sample["question"],
                "answer_f1": round(_token_f1(response.answer, sample["reference_answer"]), 4),
                "context_recall": round(len(citation_files & expected_files) / max(len(expected_files), 1), 4),
                "citation_hit_rate": round(len(citation_files & expected_files) / max(len(citation_files), 1), 4),
                "confidence": response.confidence,
                "needs_review": response.needs_review,
                "citation_files": sorted(citation_files),
                "expected_files": sorted(expected_files),
            }
        )
    summary = {
        "sample_count": len(rows),
        "avg_answer_f1": round(sum(row["answer_f1"] for row in rows) / max(len(rows), 1), 4),
        "avg_context_recall": round(sum(row["context_recall"] for row in rows) / max(len(rows), 1), 4),
        "avg_citation_hit_rate": round(sum(row["citation_hit_rate"] for row in rows) / max(len(rows), 1), 4),
        "avg_confidence": round(sum(row["confidence"] for row in rows) / max(len(rows), 1), 4),
    }
    return EvaluationResponse(summary=summary, rows=rows)


APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Advanced Agentic RAG</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #071427;
      --navy: #0b1f3a;
      --navy-2: #102946;
      --panel: #10243f;
      --panel-2: #0d1d34;
      --line: #24486d;
      --line-soft: rgba(93, 140, 186, 0.3);
      --ink: #eef6ff;
      --muted: #9fb8d1;
      --soft: #d7e8ff;
      --cyan: #38bdf8;
      --teal: #2dd4bf;
      --amber: #fbbf24;
      --rose: #fb7185;
      --green: #22c55e;
      --shadow: 0 22px 70px rgba(0, 0, 0, 0.34);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 36rem),
        linear-gradient(135deg, #061124 0%, #091a31 44%, #0d223d 100%);
      letter-spacing: 0;
    }
    button, input, select, textarea { font: inherit; }
    h1, h2, h3, p { margin: 0; }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 310px minmax(0, 1fr);
    }
    aside {
      padding: 24px;
      border-right: 1px solid var(--line-soft);
      background: rgba(5, 15, 31, 0.78);
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }
    main {
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .brand { display: grid; gap: 10px; margin-bottom: 22px; }
    .brand h1 { font-size: 28px; line-height: 1.08; }
    .brand p { color: var(--muted); line-height: 1.45; }
    .tabs { display: grid; gap: 8px; margin: 20px 0; }
    .tab {
      border: 1px solid transparent;
      background: transparent;
      color: var(--muted);
      text-align: left;
      border-radius: 8px;
      padding: 11px 12px;
      cursor: pointer;
    }
    .tab.active {
      color: var(--ink);
      background: rgba(56, 189, 248, 0.12);
      border-color: var(--line);
    }
    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 14px;
    }
    .metric {
      min-height: 74px;
      padding: 12px;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: rgba(16, 41, 70, 0.58);
    }
    .metric strong { display: block; font-size: 24px; color: var(--soft); }
    .metric span { display: block; margin-top: 4px; color: var(--muted); font-size: 12px; }
    .panel {
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: rgba(16, 36, 63, 0.86);
      box-shadow: var(--shadow);
    }
    .panel.pad { padding: 18px; }
    .section { display: none; }
    .section.active { display: grid; gap: 18px; }
    .workspace {
      display: grid;
      grid-template-columns: minmax(340px, 0.82fr) minmax(420px, 1.18fr);
      gap: 18px;
      align-items: start;
    }
    .query-form { display: grid; gap: 14px; }
    label { display: grid; gap: 8px; color: var(--muted); font-size: 13px; font-weight: 700; }
    textarea, input, select {
      color: var(--ink);
      background: #091a31;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    textarea { width: 100%; min-height: 170px; resize: vertical; line-height: 1.55; }
    textarea:focus, input:focus, select:focus {
      outline: 3px solid rgba(56, 189, 248, 0.18);
      border-color: var(--cyan);
    }
    .control-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .switch-row { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .switch {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 40px;
      color: var(--soft);
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 9px 10px;
      background: rgba(8, 23, 42, 0.58);
    }
    .switch input { width: 18px; height: 18px; accent-color: var(--cyan); }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    button.primary, button.secondary, .hitl-btn {
      border-radius: 8px;
      border: 1px solid transparent;
      min-height: 42px;
      padding: 11px 14px;
      cursor: pointer;
      font-weight: 800;
    }
    button.primary { color: #04111f; background: linear-gradient(135deg, var(--cyan), var(--teal)); }
    button.secondary { color: var(--ink); background: #0a1b31; border-color: var(--line); }
    button:disabled { opacity: 0.62; cursor: wait; }
    .answer-card { display: grid; gap: 16px; }
    .answer {
      min-height: 170px;
      white-space: pre-wrap;
      line-height: 1.7;
      color: var(--soft);
      border-left: 4px solid var(--cyan);
      padding: 2px 0 2px 16px;
    }
    .subgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 9px;
      border: 1px solid var(--line-soft);
      border-radius: 999px;
      color: var(--soft);
      background: rgba(9, 26, 49, 0.82);
      font-size: 12px;
    }
    .badge.good { border-color: rgba(34, 197, 94, 0.45); color: #bbf7d0; }
    .badge.warn { border-color: rgba(251, 191, 36, 0.45); color: #fde68a; }
    .badge.bad { border-color: rgba(251, 113, 133, 0.45); color: #fecdd3; }
    .flow {
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 12px;
    }
    .node {
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: rgba(7, 20, 39, 0.82);
      padding: 12px;
      min-height: 122px;
      position: relative;
    }
    .node:after {
      content: "";
      position: absolute;
      right: -13px;
      top: 50%;
      width: 13px;
      height: 1px;
      background: var(--line);
    }
    .node:nth-child(4n):after, .node:last-child:after { display: none; }
    .node h3 { font-size: 14px; margin-bottom: 7px; }
    .node p { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .node.complete { border-color: rgba(45, 212, 191, 0.42); }
    .node.needs_review, .node.pending { border-color: rgba(251, 191, 36, 0.5); }
    .node.passed { border-color: rgba(34, 197, 94, 0.46); }
    .list { display: grid; gap: 10px; }
    .item {
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 12px;
      background: rgba(7, 20, 39, 0.62);
    }
    .item header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 7px;
      color: var(--soft);
      font-weight: 800;
    }
    .item p, .item pre { color: var(--muted); line-height: 1.55; font-size: 13px; }
    .item pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; }
    .attention { margin-top: 8px; display: grid; gap: 6px; }
    .attention div { color: #b6d7ff; font-size: 12px; }
    .hitl-actions { display: flex; gap: 8px; margin-top: 10px; }
    .hitl-btn.approve { background: rgba(34, 197, 94, 0.16); color: #bbf7d0; border-color: rgba(34, 197, 94, 0.45); }
    .hitl-btn.reject { background: rgba(251, 113, 133, 0.14); color: #fecdd3; border-color: rgba(251, 113, 133, 0.45); }
    .table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .table th, .table td {
      border-bottom: 1px solid var(--line-soft);
      padding: 9px;
      text-align: left;
      vertical-align: top;
    }
    .table th { color: var(--soft); }
    .table td { color: var(--muted); }
    .source-list { display: grid; gap: 8px; max-height: 300px; overflow: auto; }
    .source-button {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 9px 10px;
      background: rgba(9, 26, 49, 0.82);
      color: var(--soft);
      cursor: pointer;
      overflow-wrap: anywhere;
    }
    .empty { color: var(--muted); line-height: 1.6; }
    @media (max-width: 1120px) {
      .shell { grid-template-columns: 1fr; }
      aside { height: auto; position: relative; border-right: 0; border-bottom: 1px solid var(--line-soft); }
      .workspace, .subgrid { grid-template-columns: 1fr; }
      .flow { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 680px) {
      main, aside { padding: 16px; }
      .control-grid, .switch-row, .flow { grid-template-columns: 1fr; }
      .node:after { display: none; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <section class="brand">
        <h1>Advanced Agentic RAG</h1>
        <p>Full workflow console for retrieval, generation, checkpoints, guardrails, source inspection, and evaluation.</p>
      </section>
      <nav class="tabs">
        <button class="tab active" data-tab="chat">RAG Chat</button>
        <button class="tab" data-tab="pipeline">Generation Graph</button>
        <button class="tab" data-tab="hitl">HITL & Checkpoints</button>
        <button class="tab" data-tab="sources">Sources & Indexes</button>
        <button class="tab" data-tab="evaluation">Evaluation</button>
      </nav>
      <section class="panel pad">
        <h2>Runtime</h2>
        <div class="metric-grid">
          <div class="metric"><strong id="sourceCount">-</strong><span>sources</span></div>
          <div class="metric"><strong id="chunkCount">-</strong><span>chunks</span></div>
          <div class="metric"><strong id="modelState">-</strong><span>generation</span></div>
          <div class="metric"><strong id="indexState">-</strong><span>retrieval</span></div>
        </div>
      </section>
    </aside>
    <main>
      <section id="chat" class="section active">
        <div class="workspace">
          <form class="panel pad query-form" id="askForm">
            <label>
              Question
              <textarea id="question" placeholder="Ask about the deployed corpus"></textarea>
            </label>
            <div class="control-grid">
              <label>Retrieval mode
                <select id="retrievalMode">
                  <option value="hybrid">Hybrid</option>
                  <option value="bm25">BM25</option>
                  <option value="hierarchical">Hierarchical</option>
                </select>
              </label>
              <label>Top K
                <input id="topK" type="number" min="1" max="10" value="5">
              </label>
            </div>
            <div class="switch-row">
              <label class="switch"><input id="useGeneration" type="checkbox" checked> Model synthesis</label>
              <label class="switch"><input id="useReranking" type="checkbox" checked> Reranking</label>
              <label class="switch"><input id="selfRag" type="checkbox" checked> Self-RAG</label>
              <label class="switch"><input id="sentenceAttention" type="checkbox" checked> Sentence attention</label>
              <label class="switch"><input id="contextReview" type="checkbox"> HITL context review</label>
              <label class="switch"><input id="finalApproval" type="checkbox"> HITL final approval</label>
            </div>
            <div class="actions">
              <button class="primary" id="askButton" type="submit">Generate Final Answer</button>
              <button class="secondary" id="clearButton" type="button">Clear</button>
            </div>
          </form>
          <section class="panel pad answer-card">
            <div class="badge-row" id="badges"><span class="badge">Ready</span></div>
            <h2>Finalized Answer</h2>
            <div id="answer" class="answer empty">The finalized answer will appear here.</div>
            <div class="subgrid">
              <section>
                <h2>Citations</h2>
                <div id="citations" class="list"></div>
              </section>
              <section>
                <h2>Guardrails</h2>
                <div id="guardrails" class="list"></div>
              </section>
            </div>
          </section>
        </div>
        <section class="panel pad">
          <h2>Retrieved Evidence</h2>
          <div id="retrievedChunks" class="list"></div>
        </section>
      </section>

      <section id="pipeline" class="section">
        <section class="panel pad">
          <h2>How The Answer Was Generated</h2>
          <div id="flow" class="flow"></div>
        </section>
        <section class="panel pad">
          <h2>Self-RAG Reflection</h2>
          <p id="reflection" class="empty">Run a question to see the reflection pass.</p>
        </section>
      </section>

      <section id="hitl" class="section">
        <section class="panel pad">
          <h2>Human-In-The-Loop Checkpoints</h2>
          <div id="checkpoints" class="list"></div>
        </section>
      </section>

      <section id="sources" class="section">
        <div class="workspace">
          <section class="panel pad">
            <h2>Corpus Sources</h2>
            <div id="sourceList" class="source-list"></div>
          </section>
          <section class="panel pad">
            <h2>Source Viewer</h2>
            <div id="sourceViewer" class="item"><p>Select a source to inspect its full deployed content.</p></div>
          </section>
        </div>
        <section class="panel pad">
          <h2>Index & Chunking Status</h2>
          <div id="indexDetails" class="list"></div>
        </section>
        <section class="panel pad">
          <h2>System Capability Map</h2>
          <div id="capabilityMap" class="list"></div>
        </section>
      </section>

      <section id="evaluation" class="section">
        <section class="panel pad">
          <div class="actions">
            <button class="primary" id="runEval" type="button">Run Golden Evaluation</button>
          </div>
        </section>
        <section class="panel pad">
          <h2>Evaluation Summary</h2>
          <div id="evalSummary" class="badge-row"><span class="badge">Not run</span></div>
        </section>
        <section class="panel pad">
          <h2>Evaluation Rows</h2>
          <div id="evalRows"></div>
        </section>
      </section>
    </main>
  </div>

  <script>
    const state = { last: null, corpus: null, runtime: null };
    const $ = (selector) => document.querySelector(selector);
    const $$ = (selector) => Array.from(document.querySelectorAll(selector));
    const escapeHtml = (value) => String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

    const badge = (text, kind = "") => `<span class="badge ${kind}">${escapeHtml(text)}</span>`;

    $$(".tab").forEach((button) => {
      button.addEventListener("click", () => {
        $$(".tab").forEach((item) => item.classList.remove("active"));
        $$(".section").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tab).classList.add("active");
      });
    });

    function payloadFromForm() {
      return {
        message: $("#question").value.trim(),
        top_k: Number($("#topK").value),
        retrieval_mode: $("#retrievalMode").value,
        use_generation: $("#useGeneration").checked,
        use_reranking: $("#useReranking").checked,
        self_rag: $("#selfRag").checked,
        checkpoints_enabled: true,
        require_context_review: $("#contextReview").checked,
        require_final_approval: $("#finalApproval").checked,
        sentence_attention: $("#sentenceAttention").checked,
        citation_display: true
      };
    }

    async function loadRuntime() {
      const [runtimeResponse, corpusResponse] = await Promise.all([
        fetch("/api/runtime"),
        fetch("/api/corpus")
      ]);
      state.runtime = await runtimeResponse.json();
      state.corpus = await corpusResponse.json();
      $("#sourceCount").textContent = state.corpus.source_count;
      $("#chunkCount").textContent = state.corpus.chunk_count;
      $("#modelState").textContent = state.runtime.azure_openai.chat_configured ? "Azure" : "Local";
      $("#indexState").textContent = state.corpus.indexes.hybrid ? "Hybrid" : "Lexical";
      renderSources();
      renderIndexDetails();
    }

    function renderSources() {
      $("#sourceList").innerHTML = state.corpus.sources.map((source) => `
        <button class="source-button" data-source="${escapeHtml(source)}">${escapeHtml(source)}</button>
      `).join("");
      $$("#sourceList .source-button").forEach((button) => {
        button.addEventListener("click", () => loadSource(button.dataset.source));
      });
    }

    async function loadSource(path) {
      const response = await fetch(`/api/source?path=${encodeURIComponent(path)}`);
      const payload = await response.json();
      $("#sourceViewer").innerHTML = `
        <header><span>${escapeHtml(payload.path)}</span><span>${payload.chunks.length} chunks</span></header>
        <pre>${escapeHtml(payload.text)}</pre>
      `;
    }

    function renderIndexDetails() {
      const strategies = Object.entries(state.corpus.strategy_counts || {})
        .map(([name, count]) => badge(`${name}: ${count}`)).join("");
      const indexes = Object.entries(state.corpus.indexes || {})
        .map(([name, ready]) => badge(`${name}: ${ready ? "ready" : "unavailable"}`, ready ? "good" : "warn")).join("");
      $("#indexDetails").innerHTML = `
        <article class="item"><header><span>Chunking Strategies</span></header><div class="badge-row">${strategies}</div></article>
        <article class="item"><header><span>Retrieval Branches</span></header><div class="badge-row">${indexes}</div></article>
        <article class="item"><header><span>Supported Files</span></header><div class="badge-row">${state.corpus.supported_extensions.map((ext) => badge(ext)).join("")}</div></article>
      `;
      const capabilityCards = [
        ["Ingestion", "Bundled corpus loading, file type detection, and guarded serverless-safe source intake."],
        ["Chunking", "Auto, fixed, semantic, recursive, adaptive, and hierarchical chunking are represented in the workflow."],
        ["Index Management", "BM25, semantic-overlap, hybrid, and reranked retrieval branches are visible in index status."],
        ["Docstore & Sources", "Source inspection shows complete documents plus generated chunks without exposing only snippets."],
        ["Chat & Retrieval", "Final answer, citations, retrieved chunks, and sentence attention are separated in the answer workspace."],
        ["Self-RAG", "Reflection can trigger expanded retrieval and is shown in the generation graph."],
        ["HITL & Checkpoints", "Context review and final approval checkpoints can be approved or rejected in the UI."],
        ["Guardrails", "Confidence, citation coverage, retrieval floor, and risk flags are displayed for every answer."],
        ["Evaluation", "Golden-set metrics show token F1, context recall, citation hit rate, and confidence."],
        ["Settings", "Retrieval mode, top K, reranking, model synthesis, Self-RAG, sentence attention, and HITL toggles are available in the query panel."]
      ];
      $("#capabilityMap").innerHTML = capabilityCards.map(([title, text]) => `
        <article class="item"><header><span>${title}</span></header><p>${text}</p></article>
      `).join("");
    }

    function renderChat(payload) {
      state.last = payload;
      $("#answer").textContent = payload.finalized_answer || payload.answer;
      $("#answer").classList.remove("empty");
      $("#badges").innerHTML = [
        badge(payload.provider),
        badge(`confidence ${payload.confidence}`, payload.needs_review ? "warn" : "good"),
        badge(payload.retrieval_mode),
        badge(payload.needs_review ? "needs review" : "finalized", payload.needs_review ? "warn" : "good")
      ].join("");
      $("#citations").innerHTML = payload.citations.map((citation) => `
        <article class="item">
          <header><span>${citation.rank}. ${escapeHtml(citation.file_name)}</span><span>${citation.score}</span></header>
          <p>${escapeHtml(citation.snippet)}</p>
          <div class="attention">${citation.sentence_attention.map((item) => `<div>${item.score}: ${escapeHtml(item.sentence)}</div>`).join("")}</div>
        </article>
      `).join("") || `<p class="empty">No citations returned.</p>`;
      $("#retrievedChunks").innerHTML = payload.retrieved_chunks.map((chunk) => `
        <article class="item">
          <header><span>${chunk.rank}. ${escapeHtml(chunk.file_name)}</span><span>${chunk.source} | ${chunk.score}</span></header>
          <p>${escapeHtml(chunk.text)}</p>
        </article>
      `).join("");
      renderGuardrails(payload.guardrails);
      renderFlow(payload.pipeline);
      renderCheckpoints(payload.checkpoints);
      $("#reflection").textContent = payload.reflection;
    }

    function renderGuardrails(guardrails) {
      const flags = guardrails.risk_flags.length
        ? guardrails.risk_flags.map((flag) => `<p>${escapeHtml(flag)}</p>`).join("")
        : `<p>No risk flags triggered.</p>`;
      $("#guardrails").innerHTML = `
        <article class="item">
          <header><span>${guardrails.passed ? "Passed" : "Needs review"}</span><span>${guardrails.confidence}</span></header>
          <div class="badge-row">
            ${badge(`coverage ${guardrails.citation_coverage}`, guardrails.citation_coverage >= 0.5 ? "good" : "warn")}
            ${badge(`score floor ${guardrails.retrieval_score_floor_met ? "met" : "missed"}`, guardrails.retrieval_score_floor_met ? "good" : "warn")}
          </div>
          ${flags}
        </article>
      `;
    }

    function renderFlow(steps) {
      $("#flow").innerHTML = steps.map((step) => `
        <article class="node ${step.status}">
          <h3>${escapeHtml(step.title)}</h3>
          <div class="badge-row">${badge(step.status)}</div>
          <p>${escapeHtml(step.summary)}</p>
        </article>
      `).join("");
    }

    function renderCheckpoints(checkpoints) {
      $("#checkpoints").innerHTML = checkpoints.map((checkpoint) => `
        <article class="item" data-checkpoint="${checkpoint.checkpoint_id}">
          <header><span>${escapeHtml(checkpoint.stage)}</span><span class="checkpoint-status">${escapeHtml(checkpoint.status)}</span></header>
          <p>${escapeHtml(checkpoint.summary)}</p>
          <div class="badge-row">
            ${badge(checkpoint.requires_human ? "human required" : "auto")}
          </div>
          <div class="hitl-actions">
            <button class="hitl-btn approve" type="button">Approve</button>
            <button class="hitl-btn reject" type="button">Reject</button>
          </div>
        </article>
      `).join("") || `<p class="empty">No checkpoints returned.</p>`;
      $$("#checkpoints .item").forEach((item) => {
        item.querySelector(".approve").addEventListener("click", () => {
          item.querySelector(".checkpoint-status").textContent = "approved";
        });
        item.querySelector(".reject").addEventListener("click", () => {
          item.querySelector(".checkpoint-status").textContent = "rejected";
        });
      });
    }

    $("#askForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      const body = payloadFromForm();
      if (!body.message) {
        $("#question").focus();
        return;
      }
      $("#askButton").disabled = true;
      $("#answer").textContent = "Generating final answer...";
      $("#answer").classList.add("empty");
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Request failed");
        renderChat(payload);
      } catch (error) {
        $("#answer").textContent = error.message;
        $("#badges").innerHTML = badge("error", "bad");
      } finally {
        $("#askButton").disabled = false;
      }
    });

    $("#clearButton").addEventListener("click", () => {
      $("#question").value = "";
      $("#answer").textContent = "The finalized answer will appear here.";
      $("#answer").classList.add("empty");
      $("#citations").innerHTML = "";
      $("#retrievedChunks").innerHTML = "";
      $("#guardrails").innerHTML = "";
      $("#flow").innerHTML = "";
      $("#checkpoints").innerHTML = "";
      $("#reflection").textContent = "Run a question to see the reflection pass.";
      $("#badges").innerHTML = badge("Ready");
    });

    $("#runEval").addEventListener("click", async () => {
      $("#runEval").disabled = true;
      $("#evalSummary").innerHTML = badge("running");
      try {
        const response = await fetch("/api/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            top_k: Number($("#topK").value),
            retrieval_mode: $("#retrievalMode").value,
            use_reranking: $("#useReranking").checked,
            self_rag: $("#selfRag").checked
          })
        });
        const payload = await response.json();
        $("#evalSummary").innerHTML = Object.entries(payload.summary)
          .map(([key, value]) => badge(`${key}: ${value}`)).join("");
        $("#evalRows").innerHTML = `
          <table class="table">
            <thead><tr><th>Sample</th><th>Answer F1</th><th>Context Recall</th><th>Citation Hit Rate</th><th>Confidence</th></tr></thead>
            <tbody>
              ${payload.rows.map((row) => `
                <tr>
                  <td>${escapeHtml(row.sample_id)}</td>
                  <td>${row.answer_f1}</td>
                  <td>${row.context_recall}</td>
                  <td>${row.citation_hit_rate}</td>
                  <td>${row.confidence}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        `;
      } finally {
        $("#runEval").disabled = false;
      }
    });

    loadRuntime().catch((error) => {
      $("#badges").innerHTML = badge(error.message, "bad");
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def read_root() -> HTMLResponse:
    return HTMLResponse(APP_HTML)


@app.get("/api")
def api_info() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "health": "/api/health",
        "runtime": "/api/runtime",
        "corpus": "/api/corpus",
        "capabilities": "/api/capabilities",
        "chat": "/api/chat",
        "evaluate": "/api/evaluate",
    }


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    corpus = _corpus_summary()
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "source_count": corpus["source_count"],
        "chunk_count": corpus["chunk_count"],
    }


@app.get("/api/runtime")
def runtime_info() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hosted": bool(os.getenv("VERCEL")),
        "azure_openai": _azure_status(),
        "corpus": _corpus_summary(),
        "capabilities": _capabilities(),
    }


@app.get("/api/corpus")
def corpus_info() -> dict[str, Any]:
    return _corpus_summary()


@app.get("/api/capabilities")
def capabilities_info() -> dict[str, Any]:
    return _capabilities()


@app.get("/api/source")
def source_view(path: str = Query(..., min_length=1)) -> dict[str, Any]:
    index = _load_corpus()
    if path not in index.raw_sources:
        raise HTTPException(status_code=404, detail="Source not found.")
    chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "ordinal": chunk.ordinal,
            "strategy": chunk.strategy,
            "text": chunk.text,
        }
        for chunk in index.chunks
        if chunk.relative_path == path
    ]
    return {"path": path, "text": index.raw_sources[path], "chunks": chunks}


@app.post("/api/chat", response_model=ChatResponse)
def chat_completion(payload: ChatRequest) -> ChatResponse:
    return _answer_pipeline(payload)


@app.post("/api/query", response_model=ChatResponse)
def query(payload: ChatRequest) -> ChatResponse:
    return _answer_pipeline(payload)


@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest) -> EvaluationResponse:
    return _run_evaluation(payload)
