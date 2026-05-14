from __future__ import annotations

import html
import base64
import hashlib
import hmac
import json
import math
import os
import platform
import re
import secrets
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

import requests
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


SERVICE_NAME = "Advanced Agentic RAG"
SERVICE_VERSION = "0.3.0"
ROOT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = ROOT_DIR / "data" / "sample_corpus"
GOLDEN_EVAL_PATH = ROOT_DIR / "data" / "golden_eval" / "ncert_physics_golden.json"
CHAT_HISTORY_DIR = ROOT_DIR / "data" / "chat_history"
APP_DB_PATH = ROOT_DIR / "data" / "agentic_rag_app.sqlite3"
SUPPORTED_CORPUS_EXTENSIONS = {".csv", ".json", ".md", ".sql", ".txt"}
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{8,80}$")
AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "agentic_rag_session")
AUTH_SESSION_SECONDS = 60 * 60 * 24 * 7
PASSWORD_HASH_ITERATIONS = 260_000
DEV_AUTH_SECRET = secrets.token_urlsafe(48)
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = "Agentic-RAG/0.3 (https://github.com/rg6830303/Agentic-RAG)"
WIKIPEDIA_TIMEOUT_SECONDS = 4
WIKIPEDIA_CACHE_TTL_SECONDS = 60 * 30
WIKIPEDIA_CACHE: dict[str, tuple[float, Any]] = {}
WIKIPEDIA_CACHE_LOCK = RLock()


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
    source_type: str = "local"
    source_url: str | None = None
    page_title: str | None = None


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
    source_type: str = "local"
    source_url: str | None = None
    page_title: str | None = None


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
    source_type: str = "local"
    source_url: str | None = None
    page_title: str | None = None


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
    session_id: str | None = Field(default=None, max_length=80)
    session_name: str | None = Field(default=None, max_length=120)
    memory_context: str | None = Field(default=None, max_length=4_000)
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
    use_wikipedia: bool = True
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=700, ge=100, le=2_000)


class ChatHistoryTurn(BaseModel):
    turn_id: str
    timestamp: str
    user_prompt: str
    ai_response: str
    user_message_id: str = ""
    assistant_message_id: str = ""
    updated_at: str | None = None
    edited: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    source_snippets: list[dict[str, Any]] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    message_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str
    updated_at: str | None = None
    turn_id: str | None = None
    parent_message_id: str | None = None
    edited: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    source_snippets: list[dict[str, Any]] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class ChatSessionSummary(BaseModel):
    session_id: str
    session_name: str
    user_agenda: str
    memory_summary: str = ""
    created_at: str
    updated_at: str
    exchange_count: int
    last_prompt: str = ""
    last_response: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatSessionView(ChatSessionSummary):
    history: list[ChatHistoryTurn] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatHistoryList(BaseModel):
    sessions: list[ChatSessionSummary]


class ChatSessionUpdate(BaseModel):
    session_name: str | None = Field(default=None, max_length=120)


class ChatSessionCreate(BaseModel):
    session_name: str | None = Field(default=None, max_length=120)


class ChatMessageEditRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12_000)
    session_name: str | None = Field(default=None, max_length=120)
    memory_context: str | None = Field(default=None, max_length=4_000)
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
    use_wikipedia: bool = True
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=700, ge=100, le=2_000)


class ChatRegenerateRequest(BaseModel):
    session_name: str | None = Field(default=None, max_length=120)
    memory_context: str | None = Field(default=None, max_length=4_000)
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
    use_wikipedia: bool = True
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
    session_id: str | None = None
    session_name: str | None = None
    history: list[ChatHistoryTurn] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    agenda_summary: str = ""
    memory_summary: str = ""
    suggestions: list[str] = Field(default_factory=list)
    chat_saved: bool = False


class EvaluationRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=10)
    retrieval_mode: Literal["bm25", "hybrid", "hierarchical"] = "hybrid"
    use_reranking: bool = True
    self_rag: bool = True


class EvaluationResponse(BaseModel):
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


class AuthSignupRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=256)
    display_name: str | None = Field(default=None, max_length=120)


class AuthLoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=256)


class AuthUserView(BaseModel):
    id: str
    email: str
    display_name: str
    created_at: str


class AuthResponse(BaseModel):
    authenticated: bool
    user: AuthUserView | None = None
    message: str = ""


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="FastAPI UI and API for an advanced agentic RAG workflow.",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _copy_chat_request(payload: ChatRequest, **updates: Any) -> ChatRequest:
    if hasattr(payload, "model_copy"):
        return payload.model_copy(update=updates)
    return payload.copy(update=updates)


def _compact_text(text: str, max_chars: int = 180) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "..."


def _session_name_from_prompt(prompt: str) -> str:
    clean = _compact_text(prompt, 72).strip(" .?!")
    return clean or "New RAG conversation"


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _b64_urlsafe(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_urlsafe_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${_b64_urlsafe(salt)}${_b64_urlsafe(digest)}"


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = _b64_urlsafe_decode(salt_text)
        expected = _b64_urlsafe_decode(digest_text)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def _auth_secret() -> bytes:
    secret = os.getenv("SESSION_SECRET") or os.getenv("AUTH_SECRET")
    if secret:
        return secret.encode("utf-8")
    # Local development fallback only. Production deployments should set SESSION_SECRET.
    return DEV_AUTH_SECRET.encode("utf-8")


def _secure_auth_cookie() -> bool:
    return bool(os.getenv("VERCEL") or os.getenv("APP_ENV", "").lower() == "production")


def _public_user(user: dict[str, Any]) -> AuthUserView:
    return AuthUserView(
        id=str(user["id"]),
        email=str(user["email"]),
        display_name=str(user.get("display_name") or user["email"]),
        created_at=str(user.get("created_at") or ""),
    )


def _session_token_for_user(user: dict[str, Any]) -> str:
    payload = {
        "user_id": user["id"],
        "email": user["email"],
        "exp": int(time.time()) + AUTH_SESSION_SECONDS,
    }
    encoded = _b64_urlsafe(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _b64_urlsafe(hmac.new(_auth_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def _decode_session_token(token: str) -> dict[str, Any] | None:
    if not token or "." not in token:
        return None
    encoded, signature = token.split(".", 1)
    expected = _b64_urlsafe(hmac.new(_auth_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_b64_urlsafe_decode(encoded).decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None
    if int(payload.get("exp") or 0) < int(time.time()):
        return None
    return payload


def _set_auth_cookie(response: Response, user: dict[str, Any]) -> None:
    response.set_cookie(
        AUTH_COOKIE_NAME,
        _session_token_for_user(user),
        max_age=AUTH_SESSION_SECONDS,
        httponly=True,
        secure=_secure_auth_cookie(),
        samesite="lax",
        path="/",
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        AUTH_COOKIE_NAME,
        httponly=True,
        secure=_secure_auth_cookie(),
        samesite="lax",
        path="/",
    )


AGENDA_MARKERS = (
    "i want",
    "i need",
    "my goal",
    "goal is",
    "objective",
    "agenda",
    "plan to",
    "trying to",
    "help me",
    "build",
    "create",
    "prepare",
    "study",
    "learn",
    "compare",
    "implement",
    "debug",
    "fix",
    "analyze",
    "understand",
)

AGENDA_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "answer",
    "based",
    "because",
    "being",
    "between",
    "could",
    "does",
    "from",
    "have",
    "help",
    "into",
    "just",
    "like",
    "make",
    "more",
    "need",
    "please",
    "question",
    "show",
    "that",
    "their",
    "there",
    "this",
    "using",
    "want",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def _agenda_keywords(text: str, limit: int = 5) -> list[str]:
    counts = Counter(
        token
        for token in _tokenize(text)
        if len(token) > 3 and token not in AGENDA_STOPWORDS
    )
    return [word for word, _count in counts.most_common(limit)]


def _goal_sentence(prompt: str) -> str:
    sentences = _sentences(prompt) or [prompt]
    for sentence in sentences:
        lowered = sentence.lower()
        if any(marker in lowered for marker in AGENDA_MARKERS):
            return _compact_text(sentence, 140)
    return _compact_text(sentences[0], 120)


def _summarize_agenda(
    existing_agenda: str,
    history: list[dict[str, Any]],
    latest_prompt: str,
    citations: list[Citation],
) -> str:
    prompts = [
        str(turn.get("user_prompt", ""))
        for turn in history[-5:]
        if str(turn.get("user_prompt", "")).strip()
    ]
    prompts.append(latest_prompt)
    prompt_text = " ".join(prompts)
    focus = _goal_sentence(latest_prompt if latest_prompt.strip() else prompt_text)
    if existing_agenda and len(focus.split()) < 5:
        focus = existing_agenda.replace("Focus:", "").strip()
    keywords = _agenda_keywords(
        " ".join(
            [
                prompt_text,
                " ".join(citation.file_name for citation in citations[:3]),
            ]
        ),
        limit=5,
    )
    topic_text = ", ".join(keywords[:4])
    if topic_text:
        summary = f"Focus: {focus}. Key topics: {topic_text}."
    else:
        summary = f"Focus: {focus}."
    return _compact_text(summary, 280)


def _suggest_next_prompts(
    question: str,
    response: ChatResponse,
    agenda_summary: str,
) -> list[str]:
    suggestions: list[str] = []
    citation_files = [citation.file_name for citation in response.citations[:3]]
    top_source = citation_files[0] if citation_files else ""
    agenda_focus = agenda_summary.replace("Focus:", "").split(". Key topics:", 1)[0].strip(" .")

    if agenda_focus:
        suggestions.append(f"How does this answer move my agenda forward: {agenda_focus}?")
    if top_source:
        suggestions.append(f"Which details from {top_source} are most important here?")
    if len(citation_files) > 1:
        suggestions.append(f"Compare the evidence from {citation_files[0]} and {citation_files[1]}.")
    if response.needs_review:
        suggestions.append("What extra context would improve confidence in this answer?")
    elif response.retrieved_chunks:
        keywords = _agenda_keywords(
            f"{question} {response.retrieved_chunks[0].text}",
            limit=4,
        )
        if len(keywords) >= 2:
            suggestions.append(f"Explain the connection between {keywords[0]} and {keywords[1]}.")
    suggestions.append("What should I verify next from the retrieved sources?")

    unique: list[str] = []
    seen: set[str] = set()
    for suggestion in suggestions:
        clean = _compact_text(suggestion, 130)
        key = clean.lower()
        if clean and key not in seen:
            unique.append(clean)
            seen.add(key)
        if len(unique) == 4:
            break
    return unique


def _message_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _turn_user_message_id(turn: dict[str, Any]) -> str:
    return str(turn.get("user_message_id") or f"msg_user_{turn.get('turn_id', uuid4().hex[:12])}")


def _turn_assistant_message_id(turn: dict[str, Any]) -> str:
    return str(turn.get("assistant_message_id") or f"msg_assistant_{turn.get('turn_id', uuid4().hex[:12])}")


def _normalize_turn(turn: dict[str, Any]) -> dict[str, Any]:
    turn_id = str(turn.get("turn_id") or f"turn_{uuid4().hex[:12]}")
    timestamp = str(turn.get("timestamp") or turn.get("created_at") or _utc_now_iso())
    normalized = {
        "turn_id": turn_id,
        "timestamp": timestamp,
        "user_prompt": str(turn.get("user_prompt") or ""),
        "ai_response": str(turn.get("ai_response") or ""),
        "user_message_id": str(turn.get("user_message_id") or f"msg_user_{turn_id}"),
        "assistant_message_id": str(turn.get("assistant_message_id") or f"msg_assistant_{turn_id}"),
        "updated_at": turn.get("updated_at"),
        "edited": bool(turn.get("edited") or False),
        "metadata": dict(turn.get("metadata") or {}),
        "citations": list(turn.get("citations") or []),
        "source_snippets": list(turn.get("source_snippets") or []),
        "suggestions": list(turn.get("suggestions") or []),
    }
    return normalized


def _normalize_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_turn(turn) for turn in history if isinstance(turn, dict)]


def _messages_from_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in _normalize_history(history):
        user_message_id = _turn_user_message_id(turn)
        assistant_message_id = _turn_assistant_message_id(turn)
        messages.append(
            {
                "message_id": user_message_id,
                "role": "user",
                "content": str(turn.get("user_prompt") or ""),
                "created_at": str(turn.get("timestamp") or _utc_now_iso()),
                "updated_at": turn.get("updated_at"),
                "turn_id": turn["turn_id"],
                "parent_message_id": None,
                "edited": bool(turn.get("edited") or False),
                "metadata": {},
                "citations": [],
                "source_snippets": [],
                "suggestions": [],
            }
        )
        messages.append(
            {
                "message_id": assistant_message_id,
                "role": "assistant",
                "content": str(turn.get("ai_response") or ""),
                "created_at": str(turn.get("updated_at") or turn.get("timestamp") or _utc_now_iso()),
                "updated_at": turn.get("updated_at"),
                "turn_id": turn["turn_id"],
                "parent_message_id": user_message_id,
                "edited": False,
                "metadata": dict(turn.get("metadata") or {}),
                "citations": list(turn.get("citations") or []),
                "source_snippets": list(turn.get("source_snippets") or []),
                "suggestions": list(turn.get("suggestions") or []),
            }
        )
    return messages


def _thread_memory_summary(history: list[dict[str, Any]], existing: str = "") -> str:
    normalized = _normalize_history(history)
    if len(normalized) <= 4:
        return _compact_text(existing, 900)
    older = normalized[:-4]
    text = " ".join(
        f"{turn.get('user_prompt', '')} {turn.get('ai_response', '')}"
        for turn in older
    )
    keywords = _agenda_keywords(text, limit=8)
    first_goal = _goal_sentence(str(older[0].get("user_prompt", ""))) if older else ""
    latest_goal = _goal_sentence(str(older[-1].get("user_prompt", ""))) if older else ""
    summary_parts = [
        f"Earlier thread focus began with: {first_goal}" if first_goal else "",
        f"Later older context included: {latest_goal}" if latest_goal and latest_goal != first_goal else "",
        f"Recurring topics: {', '.join(keywords)}" if keywords else "",
    ]
    return _compact_text(" ".join(part for part in summary_parts if part), 900)


def _action_to_chat_request(
    message: str,
    session_id: str,
    options: ChatMessageEditRequest | ChatRegenerateRequest,
    session_name: str | None = None,
) -> ChatRequest:
    data = _model_dump(options)
    data["message"] = message
    data["session_id"] = session_id
    data["session_name"] = session_name
    return ChatRequest(**data)


class ChatHistoryService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = RLock()

    def _ensure_root(self) -> bool:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        return True

    def _path_for(self, session_id: str) -> Path:
        if not SESSION_ID_PATTERN.match(session_id):
            raise ValueError("Invalid session ID.")
        return self.root / f"{session_id}.json"

    def _read_session(self, session_id: str) -> dict[str, Any] | None:
        try:
            path = self._path_for(session_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return self._normalize_session(data)

    def _write_session(self, session: dict[str, Any]) -> bool:
        if not self._ensure_root():
            return False
        try:
            path = self._path_for(str(session["session_id"]))
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(session, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except (OSError, ValueError):
            return False
        return True

    def _normalize_session(self, data: dict[str, Any]) -> dict[str, Any]:
        history = _normalize_history(list(data.get("history") or []))
        last_turn = history[-1] if history else {}
        session_id = str(data.get("session_id") or f"chat_{uuid4().hex[:12]}")
        created_at = str(data.get("created_at") or _utc_now_iso())
        updated_at = str(data.get("updated_at") or created_at)
        session_name = str(
            data.get("session_name")
            or _session_name_from_prompt(str(last_turn.get("user_prompt", "")))
        )
        memory_summary = str(
            data.get("memory_summary")
            or _thread_memory_summary(history)
            or ""
        )
        return {
            "version": 2,
            "session_id": session_id,
            "session_name": _compact_text(session_name, 120),
            "user_agenda": str(data.get("user_agenda") or ""),
            "memory_summary": memory_summary,
            "created_at": created_at,
            "updated_at": updated_at,
            "exchange_count": len(history),
            "last_prompt": str(last_turn.get("user_prompt", "")),
            "last_response": str(last_turn.get("ai_response", "")),
            "metadata": dict(data.get("metadata") or {}),
            "history": history,
            "messages": _messages_from_history(history),
        }

    def create_session(
        self,
        session_name: str | None = None,
        seed: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        now = _utc_now_iso()
        requested_session_id = session_id if session_id and SESSION_ID_PATTERN.match(session_id) else ""
        session_id = requested_session_id or f"chat_{uuid4().hex[:14]}"
        history = _normalize_history(list((seed or {}).get("history") or []))
        session = {
            "version": 2,
            "session_id": session_id,
            "session_name": _compact_text(
                session_name
                or (f"Copy of {(seed or {}).get('session_name', 'conversation')}" if seed else "New RAG conversation"),
                120,
            ),
            "user_agenda": str((seed or {}).get("user_agenda") or ""),
            "memory_summary": str((seed or {}).get("memory_summary") or ""),
            "created_at": now,
            "updated_at": now,
            "exchange_count": len(history),
            "last_prompt": str(history[-1].get("user_prompt", "")) if history else "",
            "last_response": str(history[-1].get("ai_response", "")) if history else "",
            "metadata": dict((seed or {}).get("metadata") or {}),
            "history": history,
        }
        with self._lock:
            return self._normalize_session(session), self._write_session(session)

    def list_sessions(self) -> list[dict[str, Any]]:
        if not self._ensure_root():
            return []
        sessions: list[dict[str, Any]] = []
        for path in self.root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            sessions.append(self._normalize_session(data))
        return sorted(sessions, key=lambda item: item["updated_at"], reverse=True)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read_session(session_id)

    def clone_session(self, session_id: str) -> tuple[dict[str, Any] | None, bool]:
        with self._lock:
            session = self._read_session(session_id)
            if not session:
                return None, False
            cloned, saved = self.create_session(seed=session)
            return cloned, saved

    def update_session(
        self,
        session_id: str,
        session_name: str | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        with self._lock:
            session = self._read_session(session_id)
            if not session:
                return None, False
            if session_name:
                session["session_name"] = _compact_text(session_name, 120)
                session["updated_at"] = _utc_now_iso()
            return self._normalize_session(session), self._write_session(session)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            try:
                path = self._path_for(session_id)
            except ValueError:
                return False
            if not path.exists():
                return False
            try:
                path.unlink()
            except OSError:
                return False
            return True

    def append_turn(
        self,
        session_id: str | None,
        session_name: str | None,
        user_prompt: str,
        ai_response: str,
        metadata: dict[str, Any],
        citations: list[Citation],
        retrieved_chunks: list[RetrievedChunk],
        suggestions: list[str],
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            session = self._read_session(session_id) if session_id else None
            if not session:
                session, _saved = self.create_session(
                    session_name=session_name or _session_name_from_prompt(user_prompt),
                    session_id=session_id,
                )
            elif session_name:
                session["session_name"] = _compact_text(session_name, 120)

            now = _utc_now_iso()
            history = list(session.get("history") or [])
            turn_id = f"turn_{uuid4().hex[:12]}"
            agenda = _summarize_agenda(
                str(session.get("user_agenda") or ""),
                history,
                user_prompt,
                citations,
            )
            turn = {
                "turn_id": turn_id,
                "timestamp": now,
                "updated_at": now,
                "user_message_id": _message_id("msg_user"),
                "assistant_message_id": _message_id("msg_assistant"),
                "user_prompt": user_prompt,
                "ai_response": ai_response,
                "edited": False,
                "metadata": metadata,
                "citations": [_model_dump(citation) for citation in citations[:5]],
                "source_snippets": [
                    {
                        "rank": chunk.rank,
                        "file_name": chunk.file_name,
                        "path": chunk.path,
                        "score": chunk.score,
                        "source": chunk.source,
                        "source_type": chunk.source_type,
                        "source_url": chunk.source_url,
                        "page_title": chunk.page_title,
                        "snippet": _compact_text(chunk.text, 520),
                    }
                    for chunk in retrieved_chunks[:5]
                ],
                "suggestions": suggestions,
            }
            history.append(turn)
            memory_summary = _thread_memory_summary(history, str(session.get("memory_summary") or ""))
            session.update(
                {
                    "user_agenda": agenda,
                    "memory_summary": memory_summary,
                    "updated_at": now,
                    "exchange_count": len(history),
                    "last_prompt": user_prompt,
                    "last_response": ai_response,
                    "history": history,
                    "metadata": {
                        "last_provider": metadata.get("provider"),
                        "last_retrieval_mode": metadata.get("retrieval_mode"),
                        "last_confidence": metadata.get("confidence"),
                    },
                }
            )
            saved = self._write_session(session)
            return self._normalize_session(session), saved

    def _find_user_turn_index(self, history: list[dict[str, Any]], message_id: str) -> int | None:
        for index, turn in enumerate(history):
            identifiers = {
                str(turn.get("turn_id") or ""),
                str(turn.get("user_message_id") or ""),
                f"msg_user_{turn.get('turn_id', '')}",
            }
            if message_id in identifiers:
                return index
        return None

    def session_before_user_message(
        self,
        session_id: str,
        message_id: str,
    ) -> tuple[dict[str, Any] | None, int | None]:
        with self._lock:
            session = self._read_session(session_id)
            if not session:
                return None, None
            history = list(session.get("history") or [])
            index = self._find_user_turn_index(history, message_id)
            if index is None:
                return session, None
            branch = dict(session)
            branch["history"] = history[:index]
            branch["memory_summary"] = _thread_memory_summary(branch["history"], str(session.get("memory_summary") or ""))
            return self._normalize_session(branch), index

    def session_before_latest_turn(self, session_id: str) -> tuple[dict[str, Any] | None, int | None]:
        with self._lock:
            session = self._read_session(session_id)
            if not session:
                return None, None
            history = list(session.get("history") or [])
            if not history:
                return session, None
            index = len(history) - 1
            branch = dict(session)
            branch["history"] = history[:index]
            branch["memory_summary"] = _thread_memory_summary(branch["history"], str(session.get("memory_summary") or ""))
            return self._normalize_session(branch), index

    def replace_turn_branch(
        self,
        session_id: str,
        turn_index: int,
        user_prompt: str,
        ai_response: str,
        metadata: dict[str, Any],
        citations: list[Citation],
        retrieved_chunks: list[RetrievedChunk],
        suggestions: list[str],
        edited: bool,
    ) -> tuple[dict[str, Any] | None, bool]:
        with self._lock:
            session = self._read_session(session_id)
            if not session:
                return None, False
            history = list(session.get("history") or [])
            if turn_index < 0 or turn_index >= len(history):
                return None, False
            original = _normalize_turn(history[turn_index])
            now = _utc_now_iso()
            prefix = history[:turn_index]
            turn = {
                "turn_id": original["turn_id"],
                "timestamp": original["timestamp"],
                "updated_at": now,
                "user_message_id": original["user_message_id"],
                "assistant_message_id": _message_id("msg_assistant"),
                "user_prompt": user_prompt,
                "ai_response": ai_response,
                "edited": bool(edited or original.get("edited")),
                "metadata": metadata,
                "citations": [_model_dump(citation) for citation in citations[:5]],
                "source_snippets": [
                    {
                        "rank": chunk.rank,
                        "file_name": chunk.file_name,
                        "path": chunk.path,
                        "score": chunk.score,
                        "source": chunk.source,
                        "source_type": chunk.source_type,
                        "source_url": chunk.source_url,
                        "page_title": chunk.page_title,
                        "snippet": _compact_text(chunk.text, 520),
                    }
                    for chunk in retrieved_chunks[:5]
                ],
                "suggestions": suggestions,
            }
            new_history = _normalize_history(prefix + [turn])
            agenda = _summarize_agenda(
                "",
                new_history[:-1],
                user_prompt,
                citations,
            )
            session.update(
                {
                    "user_agenda": agenda,
                    "memory_summary": _thread_memory_summary(new_history),
                    "updated_at": now,
                    "exchange_count": len(new_history),
                    "last_prompt": user_prompt,
                    "last_response": ai_response,
                    "history": new_history,
                    "metadata": {
                        "last_provider": metadata.get("provider"),
                        "last_retrieval_mode": metadata.get("retrieval_mode"),
                        "last_confidence": metadata.get("confidence"),
                        "branched_from_turn": original["turn_id"] if edited else "",
                    },
                }
            )
            saved = self._write_session(session)
            return self._normalize_session(session), saved


def _json_text(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=True, separators=(",", ":"))


def _json_list_text(value: Any) -> str:
    return json.dumps(value or [], ensure_ascii=True, separators=(",", ":"))


def _json_loaded(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


class AccountStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL") or f"sqlite:///{APP_DB_PATH}"
        self.is_postgres = self.database_url.startswith(("postgres://", "postgresql://"))
        self._lock = RLock()
        self._psycopg: Any = None
        self._dict_row: Any = None
        if self.is_postgres:
            try:
                import psycopg  # type: ignore
                from psycopg.rows import dict_row  # type: ignore
            except ImportError as exc:
                raise RuntimeError("DATABASE_URL points to Postgres, but psycopg is not installed.") from exc
            self._psycopg = psycopg
            self._dict_row = dict_row
        else:
            self.sqlite_path = self._sqlite_path(self.database_url)
            if self.sqlite_path != ":memory:":
                Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @staticmethod
    def _sqlite_path(database_url: str) -> str:
        if database_url == "sqlite:///:memory:":
            return ":memory:"
        if database_url.startswith("sqlite:///"):
            return database_url[len("sqlite:///") :]
        return database_url

    def _connect(self) -> Any:
        if self.is_postgres:
            return self._psycopg.connect(
                self.database_url,
                autocommit=True,
                row_factory=self._dict_row,
            )
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.is_postgres else sql

    @staticmethod
    def _row(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(row)

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            conn = self._connect()
            cursor = None
            try:
                cursor = conn.execute(self._sql(sql), params)
                if not self.is_postgres:
                    conn.commit()
            finally:
                if cursor is not None:
                    cursor.close()
                conn.close()

    def _executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        if not rows:
            return
        with self._lock:
            conn = self._connect()
            cursor = None
            try:
                cursor = conn.cursor()
                cursor.executemany(self._sql(sql), rows)
                if not self.is_postgres:
                    conn.commit()
            finally:
                if cursor is not None:
                    cursor.close()
                conn.close()

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            cursor = None
            try:
                cursor = conn.execute(self._sql(sql), params)
                row = cursor.fetchone()
            finally:
                if cursor is not None:
                    cursor.close()
                conn.close()
        return self._row(row)

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            cursor = None
            try:
                cursor = conn.execute(self._sql(sql), params)
                rows = cursor.fetchall()
            finally:
                if cursor is not None:
                    cursor.close()
                conn.close()
        return [dict(row) for row in rows]

    def _init_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_threads (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                memory_summary TEXT NOT NULL DEFAULT '',
                user_agenda TEXT NOT NULL DEFAULT '',
                last_preview TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                edited INTEGER NOT NULL DEFAULT 0,
                parent_message_id TEXT,
                turn_id TEXT,
                citations_json TEXT NOT NULL DEFAULT '[]',
                source_snippets_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                suggestions_json TEXT NOT NULL DEFAULT '[]'
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_chat_threads_user_updated ON chat_threads(user_id, updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created ON chat_messages(thread_id, created_at)",
        ]
        for statement in statements:
            self._execute(statement)

    def create_user(self, email: str, password: str, display_name: str | None = None) -> dict[str, Any]:
        normalized_email = _normalize_email(email)
        if not normalized_email or "@" not in normalized_email:
            raise ValueError("Enter a valid email address.")
        if self.get_user_by_email(normalized_email):
            raise ValueError("An account already exists for this email.")
        now = _utc_now_iso()
        user = {
            "id": f"user_{uuid4().hex[:18]}",
            "email": normalized_email,
            "display_name": _compact_text(display_name or normalized_email.split("@")[0] or normalized_email, 120),
            "password_hash": _hash_password(password),
            "created_at": now,
            "updated_at": now,
        }
        self._execute(
            """
            INSERT INTO users (id, email, display_name, password_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                user["email"],
                user["display_name"],
                user["password_hash"],
                user["created_at"],
                user["updated_at"],
            ),
        )
        return user

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        return self._fetchone("SELECT * FROM users WHERE email = ?", (_normalize_email(email),))

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        return self._fetchone("SELECT * FROM users WHERE id = ?", (user_id,))

    def verify_user(self, email: str, password: str) -> dict[str, Any] | None:
        user = self.get_user_by_email(email)
        if not user or not _verify_password(password, str(user.get("password_hash") or "")):
            return None
        return user

    def create_thread(
        self,
        user_id: str,
        title: str | None = None,
        thread_id: str | None = None,
        seed: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now_iso()
        requested_id = thread_id if thread_id and SESSION_ID_PATTERN.match(thread_id) else ""
        session_id = requested_id or f"chat_{uuid4().hex[:14]}"
        history = _normalize_history(list((seed or {}).get("history") or []))
        if seed and history:
            history = [
                {
                    **turn,
                    "turn_id": f"turn_{uuid4().hex[:12]}",
                    "user_message_id": _message_id("msg_user"),
                    "assistant_message_id": _message_id("msg_assistant"),
                }
                for turn in history
            ]
        display_title = _compact_text(
            title
            or (f"Copy of {(seed or {}).get('session_name', 'conversation')}" if seed else "New RAG conversation"),
            120,
        )
        self._execute(
            """
            INSERT INTO chat_threads
                (id, user_id, title, created_at, updated_at, memory_summary, user_agenda, last_preview, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_id,
                display_title,
                now,
                now,
                str((seed or {}).get("memory_summary") or ""),
                str((seed or {}).get("user_agenda") or ""),
                str(history[-1].get("user_prompt", "")) if history else "",
                _json_text((seed or {}).get("metadata") or {"created": True}),
            ),
        )
        if history:
            self._insert_history(user_id, session_id, history)
            self._update_thread_rollup(user_id, session_id, history)
        thread = self.get_thread(user_id, session_id)
        if not thread:
            raise RuntimeError("Could not create chat thread.")
        return thread

    def list_threads(self, user_id: str) -> list[dict[str, Any]]:
        rows = self._fetchall(
            """
            SELECT t.*,
                   (SELECT COUNT(*) FROM chat_messages m WHERE m.thread_id = t.id AND m.user_id = t.user_id AND m.role = 'user') AS exchange_count
            FROM chat_threads t
            WHERE t.user_id = ?
            ORDER BY t.updated_at DESC
            """,
            (user_id,),
        )
        return [self._session_from_thread_row(row, []) for row in rows]

    def get_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        row = self._fetchone(
            "SELECT * FROM chat_threads WHERE id = ? AND user_id = ?",
            (thread_id, user_id),
        )
        if not row:
            return None
        messages = self._fetchall(
            """
            SELECT * FROM chat_messages
            WHERE thread_id = ? AND user_id = ?
            ORDER BY created_at, id
            """,
            (thread_id, user_id),
        )
        return self._session_from_thread_row(row, messages)

    def update_thread(self, user_id: str, thread_id: str, session_name: str | None = None) -> dict[str, Any] | None:
        if session_name:
            self._execute(
                "UPDATE chat_threads SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (_compact_text(session_name, 120), _utc_now_iso(), thread_id, user_id),
            )
        return self.get_thread(user_id, thread_id)

    def delete_thread(self, user_id: str, thread_id: str) -> bool:
        if not self.get_thread(user_id, thread_id):
            return False
        self._execute("DELETE FROM chat_messages WHERE thread_id = ? AND user_id = ?", (thread_id, user_id))
        self._execute("DELETE FROM chat_threads WHERE id = ? AND user_id = ?", (thread_id, user_id))
        return True

    def clone_thread(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        session = self.get_thread(user_id, thread_id)
        if not session:
            return None
        return self.create_thread(
            user_id,
            title=f"Copy of {session.get('session_name', 'conversation')}",
            seed=session,
        )

    def append_turn(
        self,
        user_id: str,
        session_id: str | None,
        session_name: str | None,
        user_prompt: str,
        ai_response: str,
        metadata: dict[str, Any],
        citations: list[Citation],
        retrieved_chunks: list[RetrievedChunk],
        suggestions: list[str],
    ) -> tuple[dict[str, Any], bool]:
        session = self.get_thread(user_id, session_id) if session_id else None
        if not session:
            session = self.create_thread(
                user_id,
                title=session_name or _session_name_from_prompt(user_prompt),
                thread_id=session_id,
            )
        elif session_name:
            session = self.update_thread(user_id, session["session_id"], session_name=session_name) or session
        now = _utc_now_iso()
        history = list(session.get("history") or [])
        turn = self._turn_dict(
            user_prompt=user_prompt,
            ai_response=ai_response,
            metadata=metadata,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            suggestions=suggestions,
            timestamp=now,
        )
        self._insert_history(user_id, session["session_id"], [turn])
        new_history = _normalize_history(history + [turn])
        self._update_thread_rollup(user_id, session["session_id"], new_history)
        return self.get_thread(user_id, session["session_id"]) or session, True

    def session_before_user_message(
        self,
        user_id: str,
        session_id: str,
        message_id: str,
    ) -> tuple[dict[str, Any] | None, int | None]:
        session = self.get_thread(user_id, session_id)
        if not session:
            return None, None
        history = list(session.get("history") or [])
        for index, turn in enumerate(history):
            identifiers = {
                str(turn.get("turn_id") or ""),
                str(turn.get("user_message_id") or ""),
                f"msg_user_{turn.get('turn_id', '')}",
            }
            if message_id in identifiers:
                branch = dict(session)
                branch["history"] = history[:index]
                branch["messages"] = _messages_from_history(branch["history"])
                branch["memory_summary"] = _thread_memory_summary(branch["history"], str(session.get("memory_summary") or ""))
                return branch, index
        return session, None

    def session_before_latest_turn(self, user_id: str, session_id: str) -> tuple[dict[str, Any] | None, int | None]:
        session = self.get_thread(user_id, session_id)
        if not session:
            return None, None
        history = list(session.get("history") or [])
        if not history:
            return session, None
        index = len(history) - 1
        branch = dict(session)
        branch["history"] = history[:index]
        branch["messages"] = _messages_from_history(branch["history"])
        branch["memory_summary"] = _thread_memory_summary(branch["history"], str(session.get("memory_summary") or ""))
        return branch, index

    def replace_turn_branch(
        self,
        user_id: str,
        session_id: str,
        turn_index: int,
        user_prompt: str,
        ai_response: str,
        metadata: dict[str, Any],
        citations: list[Citation],
        retrieved_chunks: list[RetrievedChunk],
        suggestions: list[str],
        edited: bool,
    ) -> tuple[dict[str, Any] | None, bool]:
        session = self.get_thread(user_id, session_id)
        if not session:
            return None, False
        history = list(session.get("history") or [])
        if turn_index < 0 or turn_index >= len(history):
            return None, False
        original = _normalize_turn(history[turn_index])
        now = _utc_now_iso()
        replacement = self._turn_dict(
            user_prompt=user_prompt,
            ai_response=ai_response,
            metadata=metadata,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            suggestions=suggestions,
            turn_id=original["turn_id"],
            timestamp=original["timestamp"],
            user_message_id=original["user_message_id"],
            assistant_message_id=_message_id("msg_assistant"),
            updated_at=now,
            edited=bool(edited or original.get("edited")),
        )
        new_history = _normalize_history(history[:turn_index] + [replacement])
        self._execute("DELETE FROM chat_messages WHERE thread_id = ? AND user_id = ?", (session_id, user_id))
        self._insert_history(user_id, session_id, new_history)
        self._update_thread_rollup(
            user_id,
            session_id,
            new_history,
            metadata_extra={"branched_from_turn": original["turn_id"] if edited else ""},
        )
        return self.get_thread(user_id, session_id), True

    def _turn_dict(
        self,
        user_prompt: str,
        ai_response: str,
        metadata: dict[str, Any],
        citations: list[Citation] | list[dict[str, Any]],
        retrieved_chunks: list[RetrievedChunk] | list[dict[str, Any]],
        suggestions: list[str],
        turn_id: str | None = None,
        timestamp: str | None = None,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        updated_at: str | None = None,
        edited: bool = False,
    ) -> dict[str, Any]:
        return {
            "turn_id": turn_id or f"turn_{uuid4().hex[:12]}",
            "timestamp": timestamp or _utc_now_iso(),
            "updated_at": updated_at or _utc_now_iso(),
            "user_message_id": user_message_id or _message_id("msg_user"),
            "assistant_message_id": assistant_message_id or _message_id("msg_assistant"),
            "user_prompt": user_prompt,
            "ai_response": ai_response,
            "edited": bool(edited),
            "metadata": metadata,
            "citations": [
                citation if isinstance(citation, dict) else _model_dump(citation)
                for citation in list(citations or [])[:5]
            ],
            "source_snippets": [
                self._chunk_snippet(chunk)
                for chunk in list(retrieved_chunks or [])[:5]
            ],
            "suggestions": list(suggestions or []),
        }

    @staticmethod
    def _chunk_snippet(chunk: RetrievedChunk | dict[str, Any]) -> dict[str, Any]:
        if isinstance(chunk, dict):
            text = str(chunk.get("snippet") or chunk.get("text") or "")
            return {
                **chunk,
                "snippet": _compact_text(text, 520),
            }
        return {
            "rank": chunk.rank,
            "file_name": chunk.file_name,
            "path": chunk.path,
            "score": chunk.score,
            "source": chunk.source,
            "source_type": chunk.source_type,
            "source_url": chunk.source_url,
            "page_title": chunk.page_title,
            "snippet": _compact_text(chunk.text, 520),
        }

    def _insert_history(self, user_id: str, thread_id: str, history: list[dict[str, Any]]) -> None:
        rows: list[tuple[Any, ...]] = []
        for raw_turn in history:
            turn = _normalize_turn(raw_turn)
            metadata = dict(turn.get("metadata") or {})
            citations = list(turn.get("citations") or [])
            source_snippets = list(turn.get("source_snippets") or [])
            suggestions = list(turn.get("suggestions") or [])
            rows.append(
                (
                    turn["user_message_id"],
                    thread_id,
                    user_id,
                    "user",
                    turn["user_prompt"],
                    turn["timestamp"],
                    turn.get("updated_at"),
                    1 if turn.get("edited") else 0,
                    None,
                    turn["turn_id"],
                    "[]",
                    "[]",
                    "{}",
                    "[]",
                )
            )
            rows.append(
                (
                    turn["assistant_message_id"],
                    thread_id,
                    user_id,
                    "assistant",
                    turn["ai_response"],
                    turn.get("updated_at") or turn["timestamp"],
                    turn.get("updated_at"),
                    0,
                    turn["user_message_id"],
                    turn["turn_id"],
                    _json_list_text(citations),
                    _json_list_text(source_snippets),
                    _json_text(metadata),
                    _json_list_text(suggestions),
                )
            )
        self._executemany(
            """
            INSERT INTO chat_messages
                (id, thread_id, user_id, role, content, created_at, updated_at, edited,
                 parent_message_id, turn_id, citations_json, source_snippets_json, metadata_json, suggestions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _update_thread_rollup(
        self,
        user_id: str,
        thread_id: str,
        history: list[dict[str, Any]],
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        normalized = _normalize_history(history)
        last_turn = normalized[-1] if normalized else {}
        existing = self.get_thread(user_id, thread_id)
        existing_title = str((existing or {}).get("session_name") or "New RAG conversation")
        title = existing_title
        if existing_title == "New RAG conversation" and last_turn:
            title = _session_name_from_prompt(str(last_turn.get("user_prompt") or ""))
        agenda = _summarize_agenda(
            str((existing or {}).get("user_agenda") or ""),
            normalized[:-1],
            str(last_turn.get("user_prompt") or ""),
            [],
        ) if last_turn else str((existing or {}).get("user_agenda") or "")
        metadata = {
            "last_provider": (last_turn.get("metadata") or {}).get("provider") if last_turn else "",
            "last_retrieval_mode": (last_turn.get("metadata") or {}).get("retrieval_mode") if last_turn else "",
            "last_confidence": (last_turn.get("metadata") or {}).get("confidence") if last_turn else "",
        }
        metadata.update(metadata_extra or {})
        self._execute(
            """
            UPDATE chat_threads
            SET title = ?, updated_at = ?, memory_summary = ?, user_agenda = ?, last_preview = ?, metadata_json = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                title,
                _utc_now_iso(),
                _thread_memory_summary(normalized, str((existing or {}).get("memory_summary") or "")),
                agenda,
                str(last_turn.get("user_prompt") or last_turn.get("ai_response") or "") if last_turn else "",
                _json_text(metadata),
                thread_id,
                user_id,
            ),
        )

    def _session_from_thread_row(self, row: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        history = self._history_from_rows(messages)
        last_turn = history[-1] if history else {}
        exchange_count = int(row.get("exchange_count") or len(history))
        return {
            "version": 3,
            "session_id": row["id"],
            "session_name": _compact_text(str(row.get("title") or "New RAG conversation"), 120),
            "user_agenda": str(row.get("user_agenda") or ""),
            "memory_summary": str(row.get("memory_summary") or ""),
            "created_at": str(row.get("created_at") or _utc_now_iso()),
            "updated_at": str(row.get("updated_at") or _utc_now_iso()),
            "exchange_count": exchange_count,
            "last_prompt": str(row.get("last_preview") or last_turn.get("user_prompt", "")),
            "last_response": str(last_turn.get("ai_response", "")),
            "metadata": _json_loaded(row.get("metadata_json"), {}),
            "history": history,
            "messages": [self._message_view(message) for message in messages],
        }

    def _history_from_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assistant_by_parent = {
            str(row.get("parent_message_id") or ""): row
            for row in rows
            if row.get("role") == "assistant"
        }
        history: list[dict[str, Any]] = []
        for row in rows:
            if row.get("role") != "user":
                continue
            assistant = assistant_by_parent.get(str(row["id"]))
            if not assistant:
                continue
            history.append(
                _normalize_turn(
                    {
                        "turn_id": row.get("turn_id") or assistant.get("turn_id"),
                        "timestamp": row.get("created_at"),
                        "updated_at": assistant.get("updated_at") or assistant.get("created_at"),
                        "user_message_id": row["id"],
                        "assistant_message_id": assistant["id"],
                        "user_prompt": row.get("content") or "",
                        "ai_response": assistant.get("content") or "",
                        "edited": bool(row.get("edited")),
                        "metadata": _json_loaded(assistant.get("metadata_json"), {}),
                        "citations": _json_loaded(assistant.get("citations_json"), []),
                        "source_snippets": _json_loaded(assistant.get("source_snippets_json"), []),
                        "suggestions": _json_loaded(assistant.get("suggestions_json"), []),
                    }
                )
            )
        return history

    @staticmethod
    def _message_view(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "message_id": row["id"],
            "role": row["role"],
            "content": row.get("content") or "",
            "created_at": str(row.get("created_at") or _utc_now_iso()),
            "updated_at": row.get("updated_at"),
            "turn_id": row.get("turn_id"),
            "parent_message_id": row.get("parent_message_id"),
            "edited": bool(row.get("edited")),
            "metadata": _json_loaded(row.get("metadata_json"), {}),
            "citations": _json_loaded(row.get("citations_json"), []),
            "source_snippets": _json_loaded(row.get("source_snippets_json"), []),
            "suggestions": _json_loaded(row.get("suggestions_json"), []),
        }


chat_history_service = ChatHistoryService(CHAT_HISTORY_DIR)
account_store = AccountStore()


def current_user_from_request(request: Request | None) -> dict[str, Any] | None:
    if request is None:
        return None
    token = request.cookies.get(AUTH_COOKIE_NAME)
    payload = _decode_session_token(token or "")
    if not payload:
        return None
    user_id = str(payload.get("user_id") or "")
    if not user_id:
        return None
    return account_store.get_user_by_id(user_id)


def require_current_user(request: Request | None) -> dict[str, Any]:
    user = current_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to access your chats.")
    return user


def _session_summary(session: dict[str, Any]) -> ChatSessionSummary:
    return ChatSessionSummary(
        session_id=session["session_id"],
        session_name=session["session_name"],
        user_agenda=session.get("user_agenda", ""),
        memory_summary=session.get("memory_summary", ""),
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        exchange_count=int(session.get("exchange_count") or 0),
        last_prompt=session.get("last_prompt", ""),
        last_response=session.get("last_response", ""),
        metadata=session.get("metadata", {}),
    )


def _session_view(session: dict[str, Any]) -> ChatSessionView:
    return ChatSessionView(
        **_model_dump(_session_summary(session)),
        history=[ChatHistoryTurn(**turn) for turn in session.get("history", [])],
        messages=[ChatMessage(**message) for message in session.get("messages", _messages_from_history(session.get("history", [])))],
    )


def _memory_context_from_session(payload: ChatRequest, session: dict[str, Any] | None) -> str:
    parts: list[str] = []
    if payload.memory_context:
        parts.append(f"User supplied memory context: {_compact_text(payload.memory_context, 700)}")
    if session:
        agenda = str(session.get("user_agenda") or "").strip()
        if agenda:
            parts.append(f"User agenda: {agenda}")
        memory_summary = str(session.get("memory_summary") or "").strip()
        if memory_summary:
            parts.append(f"Thread memory summary: {memory_summary}")
        recent_turns = list(session.get("history") or [])[-6:]
        if recent_turns:
            transcript = []
            for turn in recent_turns:
                transcript.append(
                    "User: "
                    + _compact_text(str(turn.get("user_prompt", "")), 180)
                    + " Assistant: "
                    + _compact_text(str(turn.get("ai_response", "")), 260)
                )
            parts.append("Recent conversation: " + " | ".join(transcript))
    return _compact_text("\n".join(parts), 3_800)


def _chat_response_metadata(response: ChatResponse) -> dict[str, Any]:
    return {
        "retrieval_mode": response.retrieval_mode,
        "confidence": response.confidence,
        "provider": response.provider,
        "needs_review": response.needs_review,
        "used_methods": response.used_methods,
        "citation_count": len(response.citations),
        "retrieved_chunk_count": len(response.retrieved_chunks),
        "wikipedia_count": sum(1 for chunk in response.retrieved_chunks if chunk.source_type == "wikipedia"),
        "guardrails": _model_dump(response.guardrails),
        "checkpoints": [_model_dump(checkpoint) for checkpoint in response.checkpoints],
        "pipeline": [_model_dump(step) for step in response.pipeline],
    }


def _generate_chat_turn(
    payload: ChatRequest,
    session_context: dict[str, Any] | None,
) -> tuple[ChatResponse, str, list[str], dict[str, Any]]:
    memory_context = _memory_context_from_session(payload, session_context)
    response = _answer_pipeline(_copy_chat_request(payload, memory_context=memory_context))
    projected_agenda = _summarize_agenda(
        str((session_context or {}).get("user_agenda") or ""),
        list((session_context or {}).get("history") or []),
        payload.message.strip(),
        response.citations,
    )
    suggestions = _suggest_next_prompts(payload.message.strip(), response, projected_agenda)
    return response, projected_agenda, suggestions, _chat_response_metadata(response)


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


def _cache_get(key: str) -> Any | None:
    now = time.time()
    with WIKIPEDIA_CACHE_LOCK:
        cached = WIKIPEDIA_CACHE.get(key)
        if not cached:
            return None
        created_at, value = cached
        if now - created_at > WIKIPEDIA_CACHE_TTL_SECONDS:
            WIKIPEDIA_CACHE.pop(key, None)
            return None
        return value


def _cache_set(key: str, value: Any) -> Any:
    with WIKIPEDIA_CACHE_LOCK:
        WIKIPEDIA_CACHE[key] = (time.time(), value)
    return value


def _wikipedia_request(params: dict[str, Any]) -> dict[str, Any] | None:
    cache_key = "wikipedia:" + json.dumps(params, sort_keys=True, ensure_ascii=True)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        response = requests.get(
            WIKIPEDIA_API_URL,
            params={**params, "format": "json", "formatversion": "2", "utf8": 1},
            headers={"User-Agent": WIKIPEDIA_USER_AGENT},
            timeout=WIKIPEDIA_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return None
    if response.status_code >= 400:
        return None
    try:
        return _cache_set(cache_key, response.json())
    except ValueError:
        return None


def search_wikipedia(query: str, limit: int = 3) -> list[dict[str, Any]]:
    terms = _tokenize(query)
    if len(terms) < 2:
        return []
    payload = _wikipedia_request(
        {
            "action": "query",
            "list": "search",
            "srsearch": query[:500],
            "srlimit": max(1, min(limit, 5)),
            "srprop": "snippet|titlesnippet|size",
        }
    )
    if not payload:
        return []
    return list(payload.get("query", {}).get("search", []) or [])


def fetch_wikipedia_page_extract(title: str) -> dict[str, Any] | None:
    payload = _wikipedia_request(
        {
            "action": "query",
            "prop": "extracts|info",
            "titles": title,
            "redirects": 1,
            "explaintext": 1,
            "exsectionformat": "plain",
            "exchars": 5200,
            "inprop": "url",
        }
    )
    pages = (payload or {}).get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    if page.get("missing"):
        return None
    extract = re.sub(r"\s+", " ", str(page.get("extract") or "")).strip()
    if not extract:
        return None
    return {
        "page_id": str(page.get("pageid") or title),
        "title": str(page.get("title") or title),
        "extract": extract,
        "source_url": str(page.get("fullurl") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
    }


def fetch_wikipedia_summary(title: str) -> dict[str, Any] | None:
    return fetch_wikipedia_page_extract(title)


def _wikipedia_score(query: str, text: str, result_rank: int, limit: int) -> float:
    query_terms = set(_tokenize(query))
    text_terms = set(_tokenize(text))
    overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
    rank_boost = (max(limit - result_rank + 1, 1) / max(limit, 1)) * 0.45
    return round(0.7 + rank_boost + overlap * 2.8, 4)


def build_wikipedia_context(query: str, limit: int = 3) -> list[tuple[CorpusChunk, float, str]]:
    if os.getenv("AGENTIC_RAG_DISABLE_WIKIPEDIA", "").strip().lower() in {"1", "true", "yes"}:
        return []

    hits: list[tuple[CorpusChunk, float, str]] = []
    seen_titles: set[str] = set()
    for result_rank, result in enumerate(search_wikipedia(query, limit=limit), start=1):
        title = str(result.get("title") or "").strip()
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        page = fetch_wikipedia_page_extract(title)
        if not page:
            continue
        extract = page["extract"]
        for ordinal, chunk_text in enumerate(_split_text(extract, max_chars=1100, overlap=150)[:2]):
            chunk = CorpusChunk(
                chunk_id=f"wikipedia:{page['page_id']}:{ordinal}",
                file_name=f"Wikipedia: {page['title']}",
                relative_path=page["source_url"],
                text=chunk_text,
                ordinal=ordinal,
                tokens=_tokenize(chunk_text),
                strategy="wikipedia_extract",
                source_type="wikipedia",
                source_url=page["source_url"],
                page_title=page["title"],
            )
            hits.append((chunk, _wikipedia_score(query, chunk_text, result_rank, limit), "wikipedia"))

    hits.sort(key=lambda item: item[1], reverse=True)
    return hits[:limit]


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _chunk_label(chunk: CorpusChunk) -> str:
    if chunk.source_type == "wikipedia":
        return chunk.page_title or chunk.file_name
    return chunk.file_name


def _source_reference(chunk: CorpusChunk) -> str:
    label = _chunk_label(chunk)
    if chunk.source_url:
        return f"{label} ({chunk.source_url})"
    return label


def _extractive_answer(question: str, hits: list[tuple[CorpusChunk, float, str]]) -> str:
    if not hits:
        return "I could not find enough matching evidence in the available knowledge sources to answer this question."

    selected: list[str] = []
    for chunk, _score, _source in hits[:5]:
        for attention in _sentence_attention(question, chunk.text, limit=2):
            if attention.sentence not in selected:
                selected.append(attention.sentence)

    if not selected:
        selected = [hits[0][0].text[:300].strip()]

    body = " ".join(selected[:5]).strip()
    source_lines = [
        f"- [{rank}] {_source_reference(chunk)}"
        for rank, (chunk, _score, _source) in enumerate(hits[:5], start=1)
    ]
    wikipedia_note = (
        "\n\nThis answer includes text retrieved from Wikipedia alongside the bundled corpus."
        if any(chunk.source_type == "wikipedia" for chunk, _score, _source in hits[:5])
        else ""
    )
    return f"{body}{wikipedia_note}\n\nSources:\n" + "\n".join(source_lines)


def _context_prompt(hits: list[tuple[CorpusChunk, float, str]]) -> str:
    blocks = []
    for rank, (chunk, score, source) in enumerate(hits, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{rank}] {chunk.relative_path}",
                    f"score={score:.3f}",
                    f"method={source}",
                    f"source_type={chunk.source_type}",
                    f"source_url={chunk.source_url or ''}",
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
                    "include bracket citations like [1], cite Wikipedia URLs only when provided "
                    "in context, and say when evidence is insufficient."
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
        source_type=chunk.source_type,
        source_url=chunk.source_url,
        page_title=chunk.page_title,
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
        source_type=chunk.source_type,
        source_url=chunk.source_url,
        page_title=chunk.page_title,
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
            payload={"retrieved_count": len(hits), "top_sources": [_chunk_label(chunk) for chunk, _score, _source in hits[:5]]},
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
    wikipedia_count = sum(1 for chunk, _score, _source in hits if chunk.source_type == "wikipedia")
    return [
        PipelineStep(
            stage="ingestion",
            title="Load corpus",
            status="complete",
            summary="Read bundled source files and optional Wikipedia text context.",
            metrics={"sources": _corpus_summary()["source_count"], "wikipedia_chunks": wikipedia_count},
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
            summary="Ranked local and Wikipedia context selected for answer generation.",
            metrics={"retrieved": len(hits), "wikipedia": wikipedia_count, "top_score": round(hits[0][1], 4) if hits else 0.0},
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
    memory_context = (payload.memory_context or "").strip()
    retrieval_query = (
        f"{question}\n\nConversation memory:\n{memory_context}"
        if memory_context
        else question
    )
    generation_question = (
        "Use this conversation memory for continuity, but ground factual claims in the retrieved context.\n\n"
        f"Conversation memory:\n{memory_context}\n\nCurrent user question:\n{question}"
        if memory_context
        else question
    )
    hits = _search_corpus(
        retrieval_query,
        top_k=payload.top_k,
        retrieval_mode=payload.retrieval_mode,
        use_reranking=payload.use_reranking,
    )
    reflection, second_pass_used = _reflect(retrieval_query, _extractive_answer(retrieval_query, hits), hits, payload.self_rag)
    if second_pass_used:
        expanded_hits = _search_corpus(
            f"{retrieval_query} {_context_prompt(hits[:2])}",
            top_k=max(payload.top_k, 6),
            retrieval_mode=payload.retrieval_mode,
            use_reranking=payload.use_reranking,
        )
        seen = {chunk.chunk_id for chunk, _score, _source in hits}
        hits.extend([item for item in expanded_hits if item[0].chunk_id not in seen])
        hits.sort(key=lambda item: item[1], reverse=True)
        hits = hits[: payload.top_k]

    wikipedia_hits = build_wikipedia_context(retrieval_query, limit=min(3, max(1, payload.top_k))) if payload.use_wikipedia else []
    if wikipedia_hits:
        seen_ids = {chunk.chunk_id for chunk, _score, _source in hits}
        hits.extend([item for item in wikipedia_hits if item[0].chunk_id not in seen_ids])
        hits.sort(key=lambda item: item[1], reverse=True)
        hits = hits[: max(payload.top_k, min(5, len(hits)))]
        reflection = f"{reflection} Wikipedia text retrieval added {len(wikipedia_hits)} cited context chunk(s)."

    draft_answer = _extractive_answer(retrieval_query, hits)
    if payload.use_generation and _azure_status()["chat_configured"] and hits:
        final_answer = _generate_answer(
            generation_question,
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
        _citation(rank, chunk, score, retrieval_query, payload.sentence_attention)
        for rank, (chunk, score, _source) in enumerate(hits[:5], start=1)
    ]
    retrieved_chunks = [
        _retrieved_chunk(rank, chunk, score, source, retrieval_query, payload.sentence_attention)
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
            "wikipedia_text": True,
            "faiss": False,
        },
        "external_sources": {
            "wikipedia": "enabled by default with graceful timeout fallback",
        },
    }


def _capabilities() -> dict[str, Any]:
    return {
        "retrieval": ["bm25", "hybrid", "hierarchical", "wikipedia_text"],
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
        "serverless_note": "Chat history is stored in local files plus a browser localStorage fallback; Vercel filesystem persistence may be ephemeral.",
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
                use_wikipedia=False,
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
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --bg: #051424;
      --navy: #010f1f;
      --navy-2: #0d1c2d;
      --panel: #122131;
      --panel-2: #1c2b3c;
      --surface-variant: #273647;
      --line: #3e4850;
      --line-soft: rgba(136, 146, 155, 0.28);
      --ink: #d4e4fa;
      --muted: #bec8d2;
      --soft: #c9e6ff;
      --cyan: #89ceff;
      --teal: #4ae176;
      --amber: #fbbf24;
      --rose: #ffb4ab;
      --green: #4ae176;
      --violet: #c0c1ff;
      --radius: 12px;
      --radius-sm: 8px;
      --space: 16px;
      --shadow: 0 22px 70px rgba(0, 0, 0, 0.34);
      --shadow-soft: 0 10px 30px rgba(0, 0, 0, 0.22);
    }
    * { box-sizing: border-box; }
    html { width: 100%; overflow-x: hidden; }
    body {
      margin: 0;
      min-height: 100vh;
      width: 100%;
      overflow-x: hidden;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(137, 206, 255, 0.12), transparent 34rem),
        linear-gradient(135deg, #010f1f 0%, #051424 45%, #0d1c2d 100%);
      letter-spacing: 0;
    }
    button, input, select, textarea { font: inherit; }
    h1, h2, h3, p { margin: 0; }
    a { color: #7dd3fc; text-decoration: none; }
    a:hover { text-decoration: underline; }
    a, pre, code, .item, .message, .badge, .session-card { overflow-wrap: anywhere; }
    .desktop-hidden { display: none !important; }
    .scroll-safe { max-width: 100%; overflow-x: auto; }
    .truncate {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    body.auth-required .mobile-topbar,
    body.auth-required .drawer-overlay,
    body.auth-required .shell,
    body.auth-required .response-details-backdrop,
    body.auth-required .response-details {
      display: none !important;
    }
    .auth-shell {
      min-height: 100dvh;
      display: none;
      place-items: center;
      padding: 24px;
    }
    body.auth-required .auth-shell { display: grid; }
    .auth-card {
      width: min(100%, 440px);
      border: 1px solid var(--line-soft);
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(16, 36, 63, 0.94), rgba(9, 26, 49, 0.9));
      box-shadow: var(--shadow);
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .auth-card h1 { font-size: 28px; line-height: 1.1; }
    .auth-card p { color: var(--muted); line-height: 1.5; }
    .auth-tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      padding: 4px;
      border: 1px solid var(--line-soft);
      border-radius: 10px;
      background: rgba(3, 18, 34, 0.5);
    }
    .auth-tab {
      min-height: 42px;
      border: 0;
      border-radius: 8px;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
      font-weight: 800;
    }
    .auth-tab.active {
      color: var(--ink);
      background: rgba(56, 189, 248, 0.16);
    }
    .auth-form { display: grid; gap: 12px; }
    .auth-form label,
    .account-card {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }
    .auth-form input {
      min-height: 46px;
      width: 100%;
      color: var(--ink);
      background: rgba(9, 26, 49, 0.82);
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 11px 12px;
    }
    .auth-error { min-height: 20px; color: var(--rose); font-size: 13px; }
    .account-card {
      margin-top: 14px;
      padding: 12px;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: rgba(8, 23, 42, 0.62);
    }
    .account-card strong {
      color: var(--soft);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .account-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    #chat .chat-grid { grid-template-columns: minmax(0, 1fr); }
    #chat .insight-stack,
    #chat > .panel.pad:not(.conversation-panel) { display: none; }
    .response-details-backdrop {
      position: fixed;
      inset: 0;
      z-index: 110;
      background: rgba(1, 8, 18, 0.58);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.16s ease;
    }
    .response-details {
      position: fixed;
      z-index: 120;
      top: 0;
      right: 0;
      width: min(440px, 92vw);
      height: 100dvh;
      padding: 18px;
      padding-top: max(18px, env(safe-area-inset-top));
      padding-bottom: max(18px, env(safe-area-inset-bottom));
      border-left: 1px solid var(--line-soft);
      background: linear-gradient(180deg, rgba(16, 36, 63, 0.98), rgba(9, 26, 49, 0.96));
      box-shadow: -24px 0 70px rgba(0, 0, 0, 0.4);
      transform: translateX(102%);
      transition: transform 0.18s ease;
      overflow: auto;
    }
    body.details-open .response-details { transform: translateX(0); }
    body.details-open .response-details-backdrop {
      opacity: 1;
      pointer-events: auto;
    }
    .details-head {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--line-soft);
      margin-bottom: 14px;
    }
    .details-head h2 { font-size: 18px; }
    .details-body { display: grid; gap: 12px; }
    .material-symbols-outlined {
      font-family: "Material Symbols Outlined";
      font-weight: normal;
      font-style: normal;
      font-size: 20px;
      line-height: 1;
      letter-spacing: normal;
      text-transform: none;
      display: inline-block;
      white-space: nowrap;
      direction: ltr;
      -webkit-font-feature-settings: "liga";
      -webkit-font-smoothing: antialiased;
    }
    .mobile-topbar {
      position: sticky;
      top: 0;
      z-index: 70;
      min-height: 62px;
      padding: 10px 14px;
      padding-top: max(10px, env(safe-area-inset-top));
      align-items: center;
      gap: 12px;
      border-bottom: 1px solid var(--line-soft);
      background: rgba(1, 15, 31, 0.92);
      backdrop-filter: blur(18px);
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24);
    }
    .mobile-title {
      min-width: 0;
      display: grid;
      gap: 2px;
      flex: 1;
    }
    .mobile-title strong {
      display: block;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--ink);
      font-size: 14px;
      line-height: 1.2;
    }
    .drawer-overlay {
      position: fixed;
      inset: 0;
      z-index: 80;
      background: rgba(1, 8, 18, 0.62);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.18s ease;
    }
    body.drawer-open .drawer-overlay {
      opacity: 1;
      pointer-events: auto;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-width: 0;
    }
    aside {
      padding: 22px;
      border-right: 1px solid var(--line-soft);
      background: linear-gradient(180deg, rgba(1, 15, 31, 0.98), rgba(13, 28, 45, 0.96));
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }
    .drawer-head {
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--line-soft);
    }
    main {
      padding: 24px;
      display: grid;
      gap: 18px;
      min-width: 0;
    }
    .brand { display: grid; gap: 10px; margin-bottom: 18px; }
    .brand h1 { font-size: 30px; line-height: 1.08; }
    .brand p { color: var(--muted); line-height: 1.45; }
    .brand-mark {
      width: 42px;
      height: 42px;
      display: grid;
      place-items: center;
      border-radius: 12px;
      color: #04111f;
      background: linear-gradient(135deg, var(--cyan), #0ea5e9);
      font-weight: 900;
      letter-spacing: 0;
      box-shadow: 0 10px 24px rgba(56, 189, 248, 0.24);
    }
    .sidebar-section {
      border-top: 1px solid var(--line-soft);
      padding-top: 16px;
      margin-top: 18px;
      display: grid;
      gap: 12px;
    }
    .sidebar-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .new-chat-button {
      width: 100%;
      justify-content: center;
      margin: 4px 0 2px;
    }
    .stitch-note {
      color: var(--muted);
      font-family: "JetBrains Mono", ui-monospace, monospace;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .tabs { display: grid; gap: 8px; margin: 20px 0; }
    .tab {
      border: 1px solid transparent;
      background: transparent;
      color: var(--muted);
      text-align: left;
      border-radius: 8px;
      padding: 11px 12px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .tab:hover {
      color: var(--soft);
      background: rgba(56, 189, 248, 0.06);
    }
    .tab.active {
      color: var(--ink);
      background: rgba(56, 189, 248, 0.14);
      border-color: var(--cyan);
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
      border-radius: var(--radius);
      background: linear-gradient(180deg, rgba(16, 36, 63, 0.9), rgba(9, 26, 49, 0.84));
      box-shadow: var(--shadow);
    }
    .panel.pad { padding: 20px; }
    .section { display: none; }
    .section.active { display: grid; gap: 18px; }
    .workspace {
      display: grid;
      grid-template-columns: minmax(340px, 0.82fr) minmax(420px, 1.18fr);
      gap: 18px;
      align-items: start;
    }
    .chat-grid {
      display: grid;
      grid-template-columns: minmax(520px, 1.4fr) minmax(300px, 0.72fr);
      gap: 18px;
      align-items: start;
    }
    .insight-stack {
      position: sticky;
      top: 24px;
      max-height: calc(100vh - 48px);
      overflow: auto;
    }
    .panel-heading {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }
    .panel-heading h2 { font-size: 16px; }
    .icon-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      min-width: 38px;
      height: 38px;
      border-radius: 8px;
      padding: 0 10px;
      color: var(--soft);
      background: rgba(9, 26, 49, 0.82);
      border: 1px solid var(--line);
      cursor: pointer;
      font-weight: 900;
    }
    .session-list { display: grid; gap: 9px; }
    .chat-search {
      width: 100%;
      min-height: 40px;
      margin-bottom: 10px;
      color: var(--ink);
      background: rgba(9, 26, 49, 0.82);
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 10px 11px;
    }
    .session-card {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line-soft);
      border-radius: var(--radius-sm);
      padding: 12px;
      background: rgba(8, 23, 42, 0.62);
      color: var(--soft);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .session-card:hover:not(.active) {
      border-color: rgba(56, 189, 248, 0.4);
      background: rgba(8, 23, 42, 0.8);
    }
    .session-card.active {
      border-color: rgba(56, 189, 248, 0.8);
      background: rgba(56, 189, 248, 0.16);
      box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15);
    }
    .session-card strong {
      display: block;
      margin-bottom: 6px;
      font-size: 13.5px;
      line-height: 1.35;
    }
    .session-card span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .session-preview {
      margin-top: 6px;
      color: rgba(190, 200, 210, 0.78);
    }
    .conversation-panel {
      min-height: calc(100vh - 48px);
      display: grid;
      grid-template-rows: auto minmax(360px, 1fr) auto;
      gap: 14px;
    }
    .conversation-head {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: start;
      border-bottom: 1px solid var(--line-soft);
      padding-bottom: 16px;
    }
    .eyebrow {
      color: var(--cyan);
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .conversation-head h2 {
      margin-top: 4px;
      font-size: 22px;
      line-height: 1.2;
    }
    .system-strip {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .message-list {
      display: grid;
      gap: 14px;
      align-content: start;
      overflow: auto;
      padding: 8px 8px 10px 0;
      scroll-behavior: smooth;
    }
    .message-row {
      display: flex;
      width: 100%;
      animation: messageIn 0.18s ease-out;
    }
    .message-row.user { justify-content: flex-end; }
    .message-row.assistant { justify-content: flex-start; }
    .message {
      max-width: min(760px, 88%);
      border: 1px solid var(--line-soft);
      border-radius: 14px;
      padding: 14px 16px;
      line-height: 1.7;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      transition: all 0.15s ease;
    }
    .message-row.user .message {
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(45, 212, 191, 0.12));
      border-color: rgba(56, 189, 248, 0.52);
      color: var(--ink);
      box-shadow: var(--shadow-soft);
    }
    .message-row.assistant .message {
      background: linear-gradient(135deg, rgba(16, 36, 63, 0.8), rgba(10, 27, 49, 0.7));
      border-color: rgba(56, 189, 248, 0.3);
      color: var(--soft);
      box-shadow: var(--shadow-soft);
    }
    .message-row.assistant.loading .message {
      border-color: rgba(45, 212, 191, 0.48);
      background: linear-gradient(135deg, rgba(16, 48, 76, 0.88), rgba(8, 28, 52, 0.78));
    }
    .message h4 {
      margin: 0 0 8px;
      color: var(--soft);
      font-size: 13px;
      font-weight: 800;
    }
    .message-content {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .message-meta {
      margin-top: 9px;
      display: flex;
      gap: 7px;
      flex-wrap: wrap;
    }
    .message-actions {
      margin-top: 10px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      opacity: 0.82;
    }
    .message:hover .message-actions,
    .message:focus-within .message-actions {
      opacity: 1;
    }
    .message-action {
      min-height: 32px;
      border: 1px solid var(--line-soft);
      border-radius: 999px;
      padding: 6px 10px;
      color: var(--soft);
      background: rgba(9, 26, 49, 0.74);
      cursor: pointer;
      font-size: 12px;
      font-weight: 800;
      display: inline-flex;
      gap: 6px;
      align-items: center;
    }
    .message-action:hover:not(:disabled) {
      border-color: rgba(137, 206, 255, 0.55);
      color: var(--cyan);
      background: rgba(16, 36, 63, 0.92);
    }
    .edited-label {
      color: var(--muted);
      font-size: 11px;
      font-family: "JetBrains Mono", ui-monospace, monospace;
      text-transform: uppercase;
    }
    .edit-box {
      display: grid;
      gap: 10px;
    }
    .edit-box textarea {
      width: 100%;
      min-height: 110px;
      resize: vertical;
      color: var(--ink);
      background: rgba(1, 15, 31, 0.9);
      border: 1px solid var(--cyan);
      border-radius: 10px;
      padding: 12px;
      line-height: 1.55;
    }
    .edit-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .composer {
      display: grid;
      gap: 14px;
      border: 1px solid var(--line-soft);
      border-top: 2px solid rgba(56, 189, 248, 0.52);
      border-radius: 12px;
      padding: 14px;
      background: linear-gradient(180deg, rgba(8, 23, 42, 0.78), rgba(5, 15, 31, 0.72));
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    .composer.loading {
      border-color: rgba(45, 212, 191, 0.55);
      box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    .composer textarea {
      min-height: 118px;
      resize: vertical;
      border-radius: 10px;
      font-size: 15px;
    }
    .composer textarea::placeholder { color: #7896b5; }
    .composer-options {
      display: grid;
      gap: 12px;
    }
    .composer-options > summary {
      display: none;
      list-style: none;
      cursor: pointer;
      color: var(--soft);
      font-weight: 900;
      min-height: 44px;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: rgba(9, 26, 49, 0.82);
    }
    .composer-options > summary::-webkit-details-marker { display: none; }
    .composer-options > summary:after {
      content: "expand_more";
      font-family: "Material Symbols Outlined";
      font-size: 20px;
      transition: transform 0.16s ease;
    }
    .composer-options[open] > summary:after { transform: rotate(180deg); }
    .composer-status {
      min-height: 42px;
      display: inline-flex;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .loading-copy {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
    }
    .typing-indicator {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      min-width: 28px;
    }
    .typing-indicator span {
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: var(--cyan);
      animation: typingPulse 1s infinite ease-in-out;
    }
    .typing-indicator span:nth-child(2) { animation-delay: 0.14s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.28s; }
    .insight-stack {
      display: grid;
      gap: 14px;
    }
    details.insight-panel > summary {
      list-style: none;
      cursor: pointer;
      color: var(--ink);
      font-size: 16px;
      font-weight: 800;
      margin-bottom: 12px;
    }
    details.insight-panel > summary::-webkit-details-marker { display: none; }
    details.insight-panel > summary:after {
      content: "";
      display: none;
    }
    .agenda-text {
      color: var(--soft);
      line-height: 1.55;
      font-size: 13px;
    }
    .suggestion-list { display: grid; gap: 8px; }
    .suggestion-button {
      text-align: left;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 11px 12px;
      color: var(--soft);
      background: rgba(9, 26, 49, 0.82);
      cursor: pointer;
      line-height: 1.5;
      transition: all 0.15s ease;
    }
    .suggestion-button:hover {
      border-color: rgba(56, 189, 248, 0.5);
      background: rgba(16, 36, 63, 0.9);
      color: var(--cyan);
    }
    .suggestion-button:active {
      transform: scale(0.98);
    }
    .source-snippet {
      color: var(--muted);
      line-height: 1.5;
      font-size: 12px;
      margin-top: 8px;
    }
    .query-form { display: grid; gap: 14px; }
    label { display: grid; gap: 8px; color: var(--muted); font-size: 13px; font-weight: 700; }
    textarea, input, select {
      color: var(--ink);
      background: #091a31;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      transition: all 0.15s ease;
    }
    textarea { width: 100%; min-height: 170px; resize: vertical; line-height: 1.55; }
    textarea:focus, input:focus, select:focus {
      outline: 3px solid rgba(56, 189, 248, 0.18);
      border-color: var(--cyan);
      background: #0a2043;
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1);
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
      padding: 12px 16px;
      cursor: pointer;
      font-weight: 800;
      transition: all 0.15s ease;
    }
    button.primary { color: #04111f; background: linear-gradient(135deg, var(--cyan), var(--teal)); }
    button.primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(56, 189, 248, 0.3); }
    button.primary:active:not(:disabled) { transform: translateY(0); }
    button.primary.loading:before {
      content: "";
      width: 12px;
      height: 12px;
      margin-right: 8px;
      display: inline-block;
      border: 2px solid rgba(4, 17, 31, 0.36);
      border-top-color: #04111f;
      border-radius: 999px;
      animation: spin 0.72s linear infinite;
      vertical-align: -2px;
    }
    button.secondary { color: var(--ink); background: #0a1b31; border-color: var(--line); }
    button.secondary:hover:not(:disabled) { background: #0d2547; border-color: var(--cyan); }
    button.secondary:active:not(:disabled) { background: #0a1f3a; }
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
      font-family: "JetBrains Mono", ui-monospace, monospace;
      letter-spacing: 0.02em;
    }
    .badge:before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: currentColor;
      opacity: 0.8;
    }
    .badge:not(.good):not(.warn):not(.bad):before { color: var(--cyan); }
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
      min-width: 0;
    }
    .item header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 7px;
      color: var(--soft);
      font-weight: 800;
      min-width: 0;
    }
    .item header span {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .item p, .item pre { color: var(--muted); line-height: 1.55; font-size: 13px; }
    .item pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; }
    .item a {
      display: inline-block;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      vertical-align: bottom;
      white-space: nowrap;
    }
    .citation-card {
      cursor: pointer;
    }
    .citation-card > summary {
      list-style: none;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--soft);
      font-weight: 800;
      min-width: 0;
    }
    .citation-card > summary::-webkit-details-marker { display: none; }
    .citation-card > summary span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .citation-card > p,
    .citation-card > .attention {
      margin-top: 8px;
    }
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
    .empty-state {
      align-self: center;
      justify-self: center;
      width: min(720px, 100%);
      border: 1px solid var(--line-soft);
      border-radius: var(--radius);
      padding: 22px;
      background: rgba(7, 20, 39, 0.55);
      box-shadow: var(--shadow-soft);
    }
    .empty-state h3 {
      font-size: 22px;
      margin-bottom: 8px;
    }
    .empty-state p {
      color: var(--muted);
      line-height: 1.65;
      margin-bottom: 16px;
    }
    .empty-prompts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .empty-prompt {
      text-align: left;
      border: 1px solid var(--line-soft);
      border-radius: var(--radius-sm);
      padding: 12px;
      color: var(--soft);
      background: rgba(9, 26, 49, 0.82);
      cursor: pointer;
      line-height: 1.45;
      transition: all 0.15s ease;
    }
    .empty-prompt:hover {
      border-color: rgba(56, 189, 248, 0.55);
      background: rgba(16, 36, 63, 0.92);
      transform: translateY(-1px);
    }
    @keyframes messageIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes typingPulse {
      0%, 80%, 100% { opacity: 0.38; transform: translateY(0); }
      40% { opacity: 1; transform: translateY(-3px); }
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @media (max-width: 1240px) and (min-width: 1025px) {
      .shell { grid-template-columns: 292px minmax(0, 1fr); }
      main, aside { padding: 18px; }
      .chat-grid { grid-template-columns: minmax(0, 1fr) minmax(270px, 0.55fr); }
      .workspace { grid-template-columns: minmax(280px, 0.8fr) minmax(360px, 1.2fr); }
      .flow { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
      .node:nth-child(2n):after { display: none; }
    }
    @media (max-width: 1024px) {
      .desktop-hidden { display: flex !important; }
      body.drawer-open { overflow: hidden; }
      .shell {
        grid-template-columns: minmax(0, 1fr);
        min-height: calc(100vh - 62px);
      }
      aside {
        position: fixed;
        inset: 0 auto 0 0;
        z-index: 90;
        width: min(86vw, 350px);
        height: 100dvh;
        border-right: 1px solid var(--line-soft);
        border-bottom: 0;
        transform: translateX(-102%);
        transition: transform 0.2s ease;
        box-shadow: 28px 0 72px rgba(0, 0, 0, 0.44);
        padding: 16px;
        padding-top: max(16px, env(safe-area-inset-top));
      }
      body.drawer-open aside { transform: translateX(0); }
      .drawer-head { display: flex; }
      .brand h1 { font-size: 24px; }
      .brand p { font-size: 13px; }
      main {
        padding: 16px;
        padding-bottom: max(16px, env(safe-area-inset-bottom));
      }
      .workspace, .chat-grid, .subgrid { grid-template-columns: minmax(0, 1fr); }
      .insight-stack {
        position: relative;
        top: auto;
        max-height: none;
        overflow: visible;
      }
      .conversation-panel {
        min-height: calc(100dvh - 94px);
        grid-template-rows: minmax(260px, 1fr) auto;
      }
      .conversation-panel > .conversation-head { display: none; }
      .panel.pad { padding: 16px; }
      .flow { grid-template-columns: minmax(0, 1fr); }
      .node:after { display: none; }
      .composer {
        position: sticky;
        bottom: max(10px, env(safe-area-inset-bottom));
        z-index: 30;
        gap: 10px;
        padding: 12px;
        border-radius: 16px;
      }
      .composer textarea {
        min-height: 82px;
        max-height: 32vh;
        font-size: 16px;
      }
      .composer-options > summary { display: flex; }
      .composer-options:not([open]) .control-grid,
      .composer-options:not([open]) .switch-row { display: none; }
      details.insight-panel > summary {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        min-height: 44px;
        margin: -2px 0 0;
      }
      details.insight-panel > summary:after {
        content: "expand_more";
        display: inline-block;
        font-family: "Material Symbols Outlined";
        font-size: 20px;
        color: var(--cyan);
        transition: transform 0.16s ease;
      }
      details.insight-panel[open] > summary:after { transform: rotate(180deg); }
      details.insight-panel:not([open]) { padding-bottom: 14px; }
      .table-wrap { overflow-x: visible; }
    }
    @media (max-width: 768px) {
      main { padding: 12px; }
      .section.active { gap: 12px; }
      .panel { border-radius: 10px; box-shadow: var(--shadow-soft); }
      .message-list {
        min-height: clamp(240px, calc(100dvh - 400px), 580px);
        max-height: none;
        padding: 4px 0 8px;
      }
      .message-row.user .message { border-top-right-radius: 6px; }
      .message-row.assistant .message { border-top-left-radius: 6px; }
      .message {
        max-width: 94%;
        padding: 12px 13px;
        font-size: 14px;
        line-height: 1.62;
      }
      .message-action { min-height: 40px; }
      .edit-actions { gap: 8px; }
      .message-row.assistant .message { max-width: 100%; }
      .empty-state {
        align-self: start;
        padding: 16px;
      }
      .empty-state h3 { font-size: 19px; }
      .empty-prompts { grid-template-columns: minmax(0, 1fr); }
      .control-grid, .switch-row { grid-template-columns: minmax(0, 1fr); }
      .actions {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: stretch;
      }
      .actions .primary,
      .actions .secondary { min-height: 46px; }
      .actions .secondary { padding-inline: 12px; }
      .composer-status {
        grid-column: 1 / -1;
        min-height: 24px;
        font-size: 12px;
      }
      .system-strip, #badges { gap: 6px; }
      .badge {
        max-width: 100%;
        font-size: 11px;
        padding: 5px 8px;
      }
      .metric-grid { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
      .item header,
      .citation-card > summary {
        align-items: flex-start;
        flex-wrap: wrap;
      }
      .item a { max-width: 100%; }
      .source-list { max-height: 38vh; }
      .source-button {
        min-height: 44px;
        padding: 11px 12px;
      }
      .hitl-actions,
      .actions:not(.mobile-topbar) { gap: 8px; }
      .hitl-btn { min-height: 44px; }
      .responsive-table,
      .responsive-table thead,
      .responsive-table tbody,
      .responsive-table tr,
      .responsive-table th,
      .responsive-table td { display: block; width: 100%; }
      .responsive-table thead { display: none; }
      .responsive-table tr {
        margin-bottom: 10px;
        border: 1px solid var(--line-soft);
        border-radius: 8px;
        background: rgba(7, 20, 39, 0.62);
        padding: 10px;
      }
      .responsive-table td {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        border: 0;
        padding: 7px 0;
      }
      .responsive-table td:before {
        content: attr(data-label);
        color: var(--soft);
        font-weight: 800;
        flex: 0 0 45%;
      }
    }
    @media (max-width: 640px) {
      .response-details {
        top: auto;
        left: 0;
        right: 0;
        bottom: 0;
        width: 100%;
        height: min(78dvh, 720px);
        border-left: 0;
        border-top: 1px solid var(--line-soft);
        border-radius: 18px 18px 0 0;
        transform: translateY(104%);
      }
      body.details-open .response-details { transform: translateY(0); }
      .mobile-topbar { min-height: 58px; padding-inline: 10px; }
      .mobile-topbar .badge {
        max-width: 94px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .icon-button {
        min-width: 44px;
        min-height: 44px;
      }
      .panel.pad.conversation-panel {
        min-height: calc(100dvh - 82px);
        border: 0;
        background: transparent;
        box-shadow: none;
        padding: 0;
      }
      .chat-grid { gap: 12px; }
      .message { max-width: 96%; }
      .message-row.assistant .message { max-width: 100%; }
      .message .attention .item p,
      #citations .item p,
      #sourceSnippets .item p {
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        overflow: hidden;
      }
      .citation-card[open] p,
      #citations .citation-card[open] p,
      #sourceSnippets .citation-card[open] p {
        display: block;
      }
      .composer textarea::placeholder { font-size: 13px; }
      .switch { min-height: 44px; }
      .brand { margin-bottom: 14px; }
      .tabs { margin: 16px 0; }
      .tab { min-height: 44px; }
      .source-button {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        overflow-wrap: normal;
      }
    }
    @media (max-width: 480px) {
      aside { width: min(92vw, 340px); }
      main { padding: 10px; }
      .panel.pad { padding: 14px; }
      .message-list { min-height: clamp(220px, calc(100dvh - 390px), 440px); }
      .message {
        max-width: 100%;
        font-size: 13.5px;
      }
      .edit-actions {
        display: grid;
        grid-template-columns: 1fr;
      }
      .edit-actions .message-action {
        justify-content: center;
        min-height: 44px;
      }
      .composer {
        padding: 10px;
        gap: 9px;
      }
      .composer textarea { min-height: 76px; }
      .composer-status { display: none; }
      .actions { grid-template-columns: 1fr; }
      .actions .secondary { display: none; }
      .metric-grid { grid-template-columns: minmax(0, 1fr); }
      .badge-row { gap: 6px; }
      .mobile-title .eyebrow { display: none; }
      .empty-state p { font-size: 13px; }
      .responsive-table td { display: grid; gap: 4px; }
      .responsive-table td:before { flex-basis: auto; }
    }
  </style>
</head>
<body class="auth-required">
  <section class="auth-shell" id="authShell" aria-label="Sign in">
    <div class="auth-card">
      <div class="brand-mark"><span class="material-symbols-outlined">memory</span></div>
      <div>
        <span class="stitch-note">Secure Workspace</span>
        <h1>Agentic RAG</h1>
        <p>Sign in to keep your RAG conversations, memory, citations, and source traces isolated to your account.</p>
      </div>
      <div class="auth-tabs" role="tablist" aria-label="Authentication">
        <button class="auth-tab active" id="loginTab" type="button" data-auth-mode="login">Login</button>
        <button class="auth-tab" id="signupTab" type="button" data-auth-mode="signup">Sign up</button>
      </div>
      <form class="auth-form" id="authForm">
        <label id="displayNameLabel" hidden>Name
          <input id="authName" autocomplete="name" placeholder="Your name">
        </label>
        <label>Email
          <input id="authEmail" type="email" autocomplete="email" placeholder="you@example.com" required>
        </label>
        <label>Password
          <input id="authPassword" type="password" autocomplete="current-password" placeholder="At least 8 characters" required>
        </label>
        <button class="primary" id="authSubmit" type="submit">Login</button>
        <p class="auth-error" id="authError" aria-live="polite"></p>
      </form>
    </div>
  </section>
  <header class="mobile-topbar desktop-hidden" aria-label="Mobile navigation">
    <button class="icon-button" id="drawerToggle" type="button" aria-controls="sidebarDrawer" aria-expanded="false" title="Open navigation">
      <span class="material-symbols-outlined">menu</span>
    </button>
    <div class="mobile-title">
      <span class="eyebrow">Agentic RAG</span>
      <strong id="mobileSessionName">New RAG conversation</strong>
    </div>
    <span class="badge good" id="mobileStatusBadge">Ready</span>
  </header>
  <div class="drawer-overlay desktop-hidden" id="drawerOverlay" aria-hidden="true"></div>
  <div class="shell">
    <aside id="sidebarDrawer" aria-label="Agentic RAG navigation">
      <div class="drawer-head">
        <span class="stitch-note">Workspace</span>
        <button class="icon-button" id="drawerClose" type="button" title="Close navigation">
          <span class="material-symbols-outlined">close</span>
        </button>
      </div>
      <section class="brand">
        <div class="brand-mark"><span class="material-symbols-outlined">memory</span></div>
        <h1>Agentic RAG</h1>
        <span class="stitch-note">Precision Intelligence</span>
        <p>Production-style AI chat over local corpus files and Wikipedia text retrieval.</p>
      </section>
      <button class="primary new-chat-button" id="newSessionButton" type="button"><span class="material-symbols-outlined">add</span> New Chat</button>
      <section class="account-card" id="accountCard">
        <span class="stitch-note">Signed in</span>
        <div class="account-row">
          <strong id="accountEmail">-</strong>
          <button class="icon-button" id="logoutButton" type="button" title="Logout"><span class="material-symbols-outlined">logout</span></button>
        </div>
      </section>
      <section class="sidebar-section">
        <div class="sidebar-title">
          <span>Saved Chats</span>
          <button class="icon-button" id="cloneSessionButton" type="button" title="Clone active chat"><span class="material-symbols-outlined">content_copy</span> Clone</button>
        </div>
        <input class="chat-search" id="chatSearch" type="search" placeholder="Search chats" aria-label="Search saved chats">
        <div id="sessionList" class="session-list">
          <p class="empty">No saved sessions yet.</p>
        </div>
      </section>
      <nav class="tabs">
        <button class="tab active" data-tab="chat"><span class="material-symbols-outlined">search</span> Search Corpus</button>
        <button class="tab" data-tab="pipeline"><span class="material-symbols-outlined">account_tree</span> Generation Graph</button>
        <button class="tab" data-tab="hitl"><span class="material-symbols-outlined">verified_user</span> HITL Checkpoints</button>
        <button class="tab" data-tab="sources"><span class="material-symbols-outlined">database</span> Knowledge Base</button>
        <button class="tab" data-tab="evaluation"><span class="material-symbols-outlined">analytics</span> Diagnostics</button>
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
        <div class="chat-grid">
          <section class="panel pad conversation-panel">
            <div class="conversation-head">
              <div>
                <span class="eyebrow">Output Generation</span>
                <h2 id="currentSessionName">New RAG conversation</h2>
                <div class="system-strip" id="systemStrip">
                  <span class="badge">Hybrid Engine</span>
                  <span class="badge">Wikipedia Text</span>
                  <span class="badge">Guardrails</span>
                </div>
              </div>
              <div class="badge-row" id="badges"><span class="badge">Ready</span></div>
            </div>
            <div id="messages" class="message-list">
              <section class="empty-state">
                <h3>Ask your knowledge base anything</h3>
                <p>Use the bundled corpus, optional Wikipedia text retrieval, citations, guardrails, and session memory in one workspace.</p>
                <div class="empty-prompts">
                  <button class="empty-prompt" type="button">Ask about the uploaded knowledge base</button>
                  <button class="empty-prompt" type="button">Summarize the corpus</button>
                  <button class="empty-prompt" type="button">Compare retrieved sources</button>
                  <button class="empty-prompt" type="button">Search Wikipedia-backed knowledge</button>
                </div>
              </section>
            </div>
            <form class="composer" id="askForm">
              <label>
                Message
                <textarea id="question" placeholder="Ask about the deployed corpus, compare sources, or search Wikipedia-backed knowledge" aria-label="Message prompt"></textarea>
              </label>
              <details class="composer-options" id="composerOptions" open>
                <summary>Retrieval settings</summary>
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
                  <label class="switch"><input id="useWikipedia" type="checkbox" checked> Wikipedia text</label>
                  <label class="switch"><input id="sentenceAttention" type="checkbox" checked> Sentence attention</label>
                  <label class="switch"><input id="contextReview" type="checkbox"> HITL context review</label>
                  <label class="switch"><input id="finalApproval" type="checkbox"> HITL final approval</label>
                </div>
              </details>
              <div class="actions">
                <button class="primary" id="askButton" type="submit"><span id="askButtonText">Send Message</span></button>
                <button class="secondary" id="clearButton" type="button">Clear Composer</button>
                <span class="composer-status" id="composerStatus" aria-live="polite"></span>
              </div>
            </form>
          </section>

          <section class="insight-stack">
            <details class="panel pad insight-panel" open>
              <summary>Agenda Summary</summary>
              <p id="agendaSummary" class="agenda-text empty">The session agenda will build from your prompts.</p>
            </details>
            <details class="panel pad insight-panel" open>
              <summary>Suggested Next Questions</summary>
              <div id="suggestions" class="suggestion-list">
                <p class="empty">Suggestions appear after an answer.</p>
              </div>
            </details>
            <details class="panel pad insight-panel" open>
              <summary>Run Metadata</summary>
              <div id="runMetadata" class="badge-row"><span class="badge">No run yet</span></div>
            </details>
            <details class="panel pad insight-panel" open>
              <summary>Citations</summary>
              <div id="citations" class="list"></div>
            </details>
            <details class="panel pad insight-panel" open>
              <summary>Guardrails</summary>
              <div id="guardrails" class="list"></div>
            </details>
            <details class="panel pad insight-panel" open>
              <summary>Source Snippets</summary>
              <div id="sourceSnippets" class="list"></div>
            </details>
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
        <section class="panel pad">
          <div class="conversation-head">
            <div>
              <span class="eyebrow">Knowledge Base</span>
              <h2>Corpus, Retrieval, And Source Viewer</h2>
            </div>
            <div class="badge-row" id="knowledgeBadges"><span class="badge">Loading</span></div>
          </div>
        </section>
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
          <div class="conversation-head">
            <div>
              <span class="eyebrow">Evaluation &amp; Diagnostics</span>
              <h2>Runtime Health And Retrieval Quality</h2>
            </div>
            <div class="badge-row" id="diagnosticBadges"><span class="badge">Ready</span></div>
          </div>
        </section>
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
  <div class="response-details-backdrop" id="detailsBackdrop" aria-hidden="true"></div>
  <section class="response-details" id="responseDetails" aria-hidden="true" aria-label="Response details">
    <div class="details-head">
      <div>
        <span class="stitch-note">Response Details</span>
        <h2 id="detailsTitle">Sources and guardrails</h2>
      </div>
      <button class="icon-button" id="detailsClose" type="button" title="Close details">
        <span class="material-symbols-outlined">close</span>
      </button>
    </div>
    <div class="details-body" id="detailsBody">
      <p class="empty">Open details from an assistant message to inspect citations, retrieved sources, and guardrails.</p>
    </div>
  </section>

  <script>
    const state = {
      last: null,
      corpus: null,
      runtime: null,
      sessions: [],
      activeSessionId: null,
      activeSession: null,
      isGenerating: false,
      responsiveReady: false,
      editingMessageId: null,
      chatSearch: "",
      currentUser: null,
      authMode: "login",
      detailsTurn: null
    };
    const LOCAL_CHAT_KEY = "agentic_rag_chats_v1";
    const $ = (selector) => document.querySelector(selector);
    const $$ = (selector) => Array.from(document.querySelectorAll(selector));
    const escapeHtml = (value) => String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

    const badge = (text, kind = "") => `<span class="badge ${kind}">${escapeHtml(text)}</span>`;
    const empty = (text) => `<p class="empty">${escapeHtml(text)}</p>`;

    async function apiFetch(url, options = {}) {
      const response = await fetch(url, {
        credentials: "same-origin",
        ...options,
        headers: {
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...(options.headers || {})
        }
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || payload.message || "Request failed");
      return payload;
    }

    function setAuthMode(mode) {
      state.authMode = mode;
      $$(".auth-tab").forEach((button) => {
        button.classList.toggle("active", button.dataset.authMode === mode);
      });
      $("#displayNameLabel").hidden = mode !== "signup";
      $("#authSubmit").textContent = mode === "signup" ? "Create Account" : "Login";
      $("#authPassword").setAttribute("autocomplete", mode === "signup" ? "new-password" : "current-password");
      $("#authError").textContent = "";
    }

    function showAuthenticatedApp(user) {
      state.currentUser = user;
      document.body.classList.remove("auth-required");
      $("#accountEmail").textContent = user.email;
      $("#authPassword").value = "";
      $("#authError").textContent = "";
    }

    function showAuthGate() {
      state.currentUser = null;
      state.sessions = [];
      state.activeSession = null;
      state.activeSessionId = null;
      document.body.classList.add("auth-required");
      renderSession(null);
    }

    function readLocalChats() {
      if (state.currentUser) return [];
      try {
        const payload = JSON.parse(localStorage.getItem(LOCAL_CHAT_KEY) || "[]");
        return Array.isArray(payload) ? payload : [];
      } catch (_error) {
        return [];
      }
    }

    function writeLocalChats(sessions) {
      if (state.currentUser) return;
      try {
        localStorage.setItem(LOCAL_CHAT_KEY, JSON.stringify(sessions.slice(0, 40)));
      } catch (_error) {
        // Browser localStorage is a best-effort fallback for Vercel/serverless deployments.
      }
    }

    function sessionSummary(session) {
      const history = normalizedHistory(session);
      const lastTurn = history.length ? history[history.length - 1] : {};
      return {
        session_id: session.session_id,
        session_name: session.session_name || session.title || "RAG conversation",
        user_agenda: session.user_agenda || session.agenda_summary || "",
        memory_summary: session.memory_summary || "",
        created_at: session.created_at || new Date().toISOString(),
        updated_at: session.updated_at || new Date().toISOString(),
        exchange_count: session.exchange_count || history.length || 0,
        last_prompt: session.last_prompt || lastTurn.user_prompt || "",
        last_response: session.last_response || lastTurn.ai_response || "",
        metadata: session.metadata || {}
      };
    }

    function mergeSessions(primary, fallback) {
      const map = new Map();
      [...fallback, ...primary].forEach((session) => {
        if (!session || !session.session_id) return;
        const previous = map.get(session.session_id);
        const sessionTurns = Number(session.exchange_count || (session.history || []).length || 0);
        const previousTurns = Number(previous && (previous.exchange_count || (previous.history || []).length || 0));
        const sessionHasHistory = Array.isArray(session.history) && session.history.length > 0;
        const previousHasHistory = previous && Array.isArray(previous.history) && previous.history.length > 0;
        const shouldReplace = !previous
          || sessionTurns > previousTurns
          || (sessionTurns === previousTurns && sessionHasHistory && !previousHasHistory)
          || (sessionTurns === previousTurns && sessionHasHistory === previousHasHistory && new Date(session.updated_at || 0) >= new Date(previous.updated_at || 0));
        if (shouldReplace) {
          map.set(session.session_id, session);
        }
      });
      return Array.from(map.values())
        .sort((a, b) => new Date(b.updated_at || 0) - new Date(a.updated_at || 0));
    }

    function saveLocalSession(session) {
      if (state.currentUser) return;
      if (!session || !session.session_id) return;
      const local = readLocalChats().filter((item) => item.session_id !== session.session_id);
      const stored = {
        ...sessionSummary(session),
        history: normalizedHistory(session),
        messages: session.messages || []
      };
      writeLocalChats([stored, ...local]);
    }

    function findLocalSession(sessionId) {
      return readLocalChats().find((session) => session.session_id === sessionId) || null;
    }

    function formatDate(value) {
      if (!value) return "";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit"
      }).format(date);
    }

    function userMessageId(turn) {
      return turn.user_message_id || `msg_user_${turn.turn_id || "draft"}`;
    }

    function assistantMessageId(turn) {
      return turn.assistant_message_id || `msg_assistant_${turn.turn_id || "draft"}`;
    }

    function clientId(prefix) {
      const cryptoApi = window.crypto || null;
      const random = cryptoApi && cryptoApi.randomUUID
        ? cryptoApi.randomUUID().replaceAll("-", "").slice(0, 14)
        : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
      return `${prefix}_${random}`;
    }

    function sessionTitleFromPrompt(prompt) {
      const clean = prompt.replace(/\s+/g, " ").trim().replace(/[.?!]+$/, "");
      return clean ? clean.slice(0, 72) : "New RAG conversation";
    }

    function normalizedHistory(session) {
      if (!session) return [];
      const history = Array.isArray(session.history) ? session.history : [];
      return history.map((turn) => ({
        ...turn,
        user_message_id: turn.user_message_id || userMessageId(turn),
        assistant_message_id: turn.assistant_message_id || assistantMessageId(turn),
        edited: Boolean(turn.edited)
      }));
    }

    function sourceSnippetsFromPayload(payload) {
      return (payload.retrieved_chunks || []).slice(0, 5).map((chunk) => ({
        rank: chunk.rank,
        file_name: chunk.file_name,
        path: chunk.path,
        score: chunk.score,
        source: chunk.source,
        source_type: chunk.source_type,
        source_url: chunk.source_url,
        page_title: chunk.page_title,
        snippet: (chunk.text || "").replace(/\s+/g, " ").trim().slice(0, 520)
      }));
    }

    function metadataFromPayload(payload) {
      return {
        retrieval_mode: payload.retrieval_mode,
        confidence: payload.confidence,
        provider: payload.provider,
        needs_review: payload.needs_review,
        used_methods: payload.used_methods || [],
        citation_count: (payload.citations || []).length,
        retrieved_chunk_count: (payload.retrieved_chunks || []).length,
        wikipedia_count: (payload.retrieved_chunks || []).filter((chunk) => chunk.source_type === "wikipedia").length,
        guardrails: payload.guardrails,
        checkpoints: payload.checkpoints || [],
        pipeline: payload.pipeline || []
      };
    }

    function createLocalThread(prompt) {
      const now = new Date().toISOString();
      return {
        session_id: clientId("chat"),
        session_name: sessionTitleFromPrompt(prompt),
        user_agenda: "",
        memory_summary: "",
        created_at: now,
        updated_at: now,
        exchange_count: 0,
        last_prompt: "",
        last_response: "",
        metadata: {},
        history: [],
        messages: []
      };
    }

    async function ensureActiveThread(prompt) {
      if (state.activeSession && state.activeSessionId) return state.activeSession;
      if (state.currentUser) {
        const payload = await apiFetch("/api/chats", {
          method: "POST",
          body: JSON.stringify({ session_name: sessionTitleFromPrompt(prompt) })
        });
        renderSession(payload);
        return state.activeSession;
      }
      const thread = createLocalThread(prompt);
      state.activeSession = thread;
      state.activeSessionId = thread.session_id;
      $("#currentSessionName").textContent = thread.session_name;
      updateMobileHeader("Draft");
      state.sessions = mergeSessions([sessionSummary(thread)], state.sessions);
      saveLocalSession(thread);
      renderSessionList();
      return thread;
    }

    function compactClientText(text, maxChars = 700) {
      const clean = String(text || "").replace(/\s+/g, " ").trim();
      return clean.length > maxChars ? `${clean.slice(0, maxChars - 1).trim()}...` : clean;
    }

    function buildThreadMemory(thread, currentPrompt = "", uptoTurnIndex = null) {
      const history = normalizedHistory(thread);
      const scopedHistory = uptoTurnIndex === null ? history : history.slice(0, uptoTurnIndex);
      const recent = scopedHistory.slice(-6);
      const older = scopedHistory.slice(0, Math.max(0, scopedHistory.length - recent.length));
      const parts = [];
      if (thread && thread.user_agenda) parts.push(`Thread agenda: ${compactClientText(thread.user_agenda, 420)}`);
      if (thread && thread.memory_summary) parts.push(`Thread memory summary: ${compactClientText(thread.memory_summary, 620)}`);
      if (older.length) {
        const olderText = older.map((turn) => turn.user_prompt).filter(Boolean).join(" | ");
        parts.push(`Earlier thread questions: ${compactClientText(olderText, 620)}`);
      }
      if (recent.length) {
        parts.push("Recent turns:");
        recent.forEach((turn) => {
          parts.push(`User: ${compactClientText(turn.user_prompt, 220)}`);
          parts.push(`Assistant: ${compactClientText(turn.ai_response, 300)}`);
        });
      }
      if (currentPrompt) parts.push(`Current prompt: ${compactClientText(currentPrompt, 500)}`);
      return compactClientText(parts.join("\n"), 3800);
    }

    function buildClientMemorySummary(history) {
      const turns = normalizedHistory({ history });
      if (turns.length <= 4) return "";
      const older = turns.slice(0, -4);
      const questions = older.map((turn) => turn.user_prompt).filter(Boolean).join(" | ");
      return compactClientText(`Earlier in this thread, the user asked: ${questions}`, 900);
    }

    function createTurnFromPayload(prompt, payload, previousTurn = {}) {
      const now = new Date().toISOString();
      const turnId = previousTurn.turn_id || clientId("turn");
      const userMessageIdValue = previousTurn.user_message_id || clientId("msg_user");
      return {
        turn_id: turnId,
        timestamp: previousTurn.timestamp || now,
        updated_at: now,
        user_message_id: userMessageIdValue,
        assistant_message_id: clientId("msg_assistant"),
        user_prompt: prompt,
        ai_response: payload.finalized_answer || payload.answer || "",
        edited: Boolean(previousTurn.edited),
        metadata: metadataFromPayload(payload),
        citations: payload.citations || [],
        source_snippets: sourceSnippetsFromPayload(payload),
        suggestions: payload.suggestions || []
      };
    }

    function threadWithHistory(thread, history, payload = null) {
      const now = new Date().toISOString();
      const lastTurn = history.length ? history[history.length - 1] : {};
      return {
        ...(thread || createLocalThread(lastTurn.user_prompt || "")),
        session_id: (thread && thread.session_id) || (payload && payload.session_id) || clientId("chat"),
        session_name: (thread && thread.session_name && thread.session_name !== "New RAG conversation")
          ? thread.session_name
          : (payload && payload.session_name) || sessionTitleFromPrompt(lastTurn.user_prompt || ""),
        user_agenda: (payload && payload.agenda_summary) || (thread && thread.user_agenda) || "",
        memory_summary: (payload && payload.memory_summary) || buildClientMemorySummary(history) || (thread && thread.memory_summary) || "",
        updated_at: now,
        exchange_count: history.length,
        last_prompt: lastTurn.user_prompt || "",
        last_response: lastTurn.ai_response || "",
        metadata: lastTurn.metadata || (thread && thread.metadata) || {},
        history,
        messages: []
      };
    }

    function persistActiveThread(thread) {
      state.activeSession = { ...thread, history: normalizedHistory(thread) };
      state.activeSessionId = state.activeSession.session_id;
      saveLocalSession(state.activeSession);
      state.sessions = mergeSessions([sessionSummary(state.activeSession)], state.sessions);
    }

    function renderLatestPayloadDetails(payload) {
      if (!payload) return;
      renderRunMetadata(metadataFromPayload(payload));
      $("#badges").innerHTML = [
        payload.provider ? badge(payload.provider) : "",
        typeof payload.confidence === "number" ? badge(`confidence ${payload.confidence}`, payload.needs_review ? "warn" : "good") : "",
        payload.retrieval_mode ? badge(payload.retrieval_mode) : "",
        badge(payload.needs_review ? "needs review" : "finalized", payload.needs_review ? "warn" : "good"),
        payload.chat_saved === false ? badge("browser saved", "warn") : badge("saved", "good")
      ].join("");
      renderCitations(payload.citations || []);
      renderSourceSnippets(payload.retrieved_chunks || []);
      $("#retrievedChunks").innerHTML = (payload.retrieved_chunks || []).map((chunk) => `
        <details class="item citation-card" open>
          <summary><span>${chunk.rank}. ${escapeHtml(chunk.page_title || chunk.file_name)}</span><span>${escapeHtml(chunk.source_type || "local")} | ${escapeHtml(chunk.source)} | ${chunk.score}</span></summary>
          <p>${escapeHtml(chunk.text)}</p>
          ${chunk.source_url ? `<p><a href="${escapeHtml(chunk.source_url)}" target="_blank" rel="noreferrer">${escapeHtml(chunk.source_url)}</a></p>` : ""}
        </details>
      `).join("") || empty("No retrieved chunks returned.");
      if (payload.guardrails) renderGuardrails(payload.guardrails);
      if (payload.pipeline) renderFlow(payload.pipeline);
      if (payload.checkpoints) renderCheckpoints(payload.checkpoints);
      if (payload.reflection) $("#reflection").textContent = payload.reflection;
      renderSuggestions(payload.suggestions || []);
    }

    function requestOptions() {
      return {
        top_k: Number($("#topK").value),
        retrieval_mode: $("#retrievalMode").value,
        use_generation: $("#useGeneration").checked,
        use_reranking: $("#useReranking").checked,
        self_rag: $("#selfRag").checked,
        use_wikipedia: $("#useWikipedia").checked,
        checkpoints_enabled: true,
        require_context_review: $("#contextReview").checked,
        require_final_approval: $("#finalApproval").checked,
        sentence_attention: $("#sentenceAttention").checked,
        citation_display: true
      };
    }

    function isCompactLayout() {
      return window.matchMedia && window.matchMedia("(max-width: 1024px)").matches;
    }

    function updateMobileHeader(statusText = "") {
      const title = $("#currentSessionName") ? $("#currentSessionName").textContent.trim() : "New RAG conversation";
      $("#mobileSessionName").textContent = title || "New RAG conversation";
      if (statusText) {
        $("#mobileStatusBadge").textContent = statusText;
      } else if (state.isGenerating) {
        $("#mobileStatusBadge").textContent = "Generating";
      } else if (state.activeSessionId) {
        $("#mobileStatusBadge").textContent = "Saved";
      } else {
        $("#mobileStatusBadge").textContent = "Ready";
      }
      $("#mobileStatusBadge").classList.toggle("warn", state.isGenerating);
      $("#mobileStatusBadge").classList.toggle("good", !state.isGenerating);
    }

    function openDrawer() {
      document.body.classList.add("drawer-open");
      $("#drawerToggle").setAttribute("aria-expanded", "true");
    }

    function closeDrawer() {
      document.body.classList.remove("drawer-open");
      $("#drawerToggle").setAttribute("aria-expanded", "false");
    }

    function syncResponsivePanels() {
      const compact = isCompactLayout();
      const options = $("#composerOptions");
      if (options && !options.dataset.touched) options.open = !compact;
      $$(".insight-panel").forEach((panel, index) => {
        if (!panel.dataset.touched) panel.open = !compact || index < 2;
      });
      $$(".citation-card").forEach((panel) => {
        panel.open = !compact;
      });
      if (!compact) closeDrawer();
      updateMobileHeader();
    }

    function bindPromptButtons() {
      $$(".empty-prompt, .suggestion-button").forEach((button) => {
        button.addEventListener("click", () => {
          $("#question").value = button.textContent.trim();
          $("#question").focus();
        });
      });
    }

    function renderEmptyChat() {
      $("#messages").innerHTML = `
        <section class="empty-state">
          <h3>Ask your knowledge base anything</h3>
          <p>Start with the local corpus, ask for Wikipedia-backed context, or compare retrieved evidence with citations.</p>
          <div class="empty-prompts">
            <button class="empty-prompt" type="button">Ask about the uploaded knowledge base</button>
            <button class="empty-prompt" type="button">Summarize the corpus</button>
            <button class="empty-prompt" type="button">Compare retrieved sources</button>
            <button class="empty-prompt" type="button">Search Wikipedia-backed knowledge</button>
          </div>
        </section>
      `;
      bindPromptButtons();
    }

    function activateTab(button) {
      $$(".tab").forEach((item) => item.classList.remove("active"));
      $$(".section").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(button.dataset.tab).classList.add("active");
      closeDrawer();
    }

    $$(".tab").forEach((button) => {
      button.addEventListener("click", () => activateTab(button));
    });

    function payloadFromForm() {
      return {
        ...requestOptions(),
        message: $("#question").value.trim(),
        session_id: state.activeSessionId
      };
    }

    function isMobileTextInput() {
      return window.matchMedia && window.matchMedia("(hover: none) and (pointer: coarse)").matches;
    }

    function setGenerating(isGenerating) {
      state.isGenerating = isGenerating;
      $("#askButton").disabled = isGenerating;
      $("#clearButton").disabled = isGenerating;
      $("#askButton").classList.toggle("loading", isGenerating);
      $("#askForm").classList.toggle("loading", isGenerating);
      $("#askForm").setAttribute("aria-busy", isGenerating ? "true" : "false");
      $("#askButtonText").textContent = isGenerating ? "Generating" : "Send Message";
      $("#composerStatus").textContent = isGenerating
        ? "Retrieving sources and composing the answer..."
        : "";
      updateMobileHeader(isGenerating ? "Generating" : "");
    }

    function appendPendingTurn(message) {
      if ($("#messages .empty-state")) {
        $("#messages").innerHTML = "";
      }
      $("#messages").insertAdjacentHTML("beforeend", `
        <article class="message-row user"><div class="message">${escapeHtml(message)}</div></article>
        <article class="message-row assistant loading" data-pending-response="true">
          <div class="message">
            <div class="loading-copy">
              <span class="typing-indicator" aria-hidden="true"><span></span><span></span><span></span></span>
              <span>Retrieving evidence and generating the RAG answer...</span>
            </div>
          </div>
        </article>
      `);
      $("#messages").scrollTop = $("#messages").scrollHeight;
    }

    function replacePendingWithError(message) {
      const pending = document.querySelector('[data-pending-response="true"]');
      if (!pending) {
        $("#messages").insertAdjacentHTML("beforeend", `
          <article class="message-row assistant"><div class="message">${escapeHtml(message)}</div></article>
        `);
        return;
      }
      pending.classList.remove("loading");
      pending.removeAttribute("data-pending-response");
      pending.querySelector(".message").innerHTML = escapeHtml(message);
    }

    async function submitPrompt() {
      if (state.isGenerating) return;
      const body = payloadFromForm();
      if (!body.message) {
        $("#question").focus();
        return;
      }

      const thread = await ensureActiveThread(body.message);
      body.session_id = thread.session_id;
      body.session_name = thread.session_name;
      body.memory_context = buildThreadMemory(thread, body.message);
      setGenerating(true);
      appendPendingTurn(body.message);
      $("#question").value = "";
      try {
        const endpoint = `/api/chats/${encodeURIComponent(thread.session_id)}/messages`;
        const payload = await apiFetch(endpoint, {
          method: "POST",
          body: JSON.stringify(body)
        });
        renderChat(payload, body.message, thread);
        await loadHistory();
        renderSessionList();
      } catch (error) {
        replacePendingWithError(error.message);
        $("#badges").innerHTML = badge("error", "bad");
      } finally {
        setGenerating(false);
      }
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
      $("#systemStrip").innerHTML = [
        badge(`${state.corpus.source_count} files`, "good"),
        badge(`${state.corpus.chunk_count} chunks`),
        badge(state.corpus.indexes.wikipedia_text ? "Wikipedia text" : "Local only", state.corpus.indexes.wikipedia_text ? "good" : "warn"),
        badge(state.runtime.azure_openai.chat_configured ? "Azure synthesis" : "Local synthesis")
      ].join("");
      $("#knowledgeBadges").innerHTML = [
        badge(`${state.corpus.source_count} files`, "good"),
        badge(`${state.corpus.chunk_count} chunks`),
        badge(state.corpus.indexes.wikipedia_text ? "Wikipedia enabled" : "Wikipedia off", state.corpus.indexes.wikipedia_text ? "good" : "warn"),
        badge(state.runtime.azure_openai.chat_configured ? "Azure ready" : "Local fallback")
      ].join("");
      $("#diagnosticBadges").innerHTML = [
        badge("Runtime healthy", "good"),
        badge(state.runtime.azure_openai.chat_configured ? "Azure configured" : "No secrets exposed"),
        badge(`${state.corpus.chunk_count} chunks`)
      ].join("");
      if (!state.last) {
        $("#badges").innerHTML = [
          badge(`${state.corpus.source_count} local sources`, "good"),
          badge("Wikipedia text", "good"),
          badge(state.runtime.azure_openai.chat_configured ? "Azure synthesis" : "Local extractive")
        ].join("");
      }
      renderSources();
      renderIndexDetails();
      await loadHistory();
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
      const chunks = (payload.chunks || []).slice(0, 10).map((chunk) => `
        <article class="item">
          <header><span>Chunk ${chunk.ordinal + 1}</span><span>${escapeHtml(chunk.strategy)}</span></header>
          <p>${escapeHtml(chunk.text || "")}</p>
        </article>
      `).join("");
      $("#sourceViewer").innerHTML = `
        <header><span>${escapeHtml(payload.path)}</span><span>${(payload.chunks || []).length} chunks</span></header>
        <pre>${escapeHtml(payload.text)}</pre>
        <div class="attention">${chunks}</div>
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
        ["Wikipedia Text", "MediaWiki API search and extract retrieval adds text-only external context with source URLs."],
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

    async function loadHistory() {
      const localSessions = state.currentUser ? [] : readLocalChats().map(sessionSummary);
      try {
        const payload = await apiFetch("/api/chats");
        state.sessions = state.currentUser
          ? (payload.sessions || [])
          : mergeSessions(payload.sessions || [], localSessions);
        renderSessionList();
      } catch (error) {
        state.sessions = localSessions;
        renderSessionList();
        if (!state.sessions.length) {
          $("#sessionList").innerHTML = empty(`${error.message}. Browser-saved chats will appear here.`);
        }
      }
    }

    function renderSessionList() {
      const query = state.chatSearch.trim().toLowerCase();
      const sessions = query
        ? state.sessions.filter((session) => [
            session.session_name,
            session.last_prompt,
            session.last_response,
            session.user_agenda
          ].join(" ").toLowerCase().includes(query))
        : state.sessions;
      if (!sessions.length) {
        $("#sessionList").innerHTML = empty("No saved sessions yet.");
        return;
      }
      $("#sessionList").innerHTML = sessions.map((session) => `
        <button class="session-card ${session.session_id === state.activeSessionId ? "active" : ""}" data-session="${escapeHtml(session.session_id)}">
          <strong>${escapeHtml(session.session_name)}</strong>
          <span>${session.exchange_count} exchanges - ${escapeHtml(formatDate(session.updated_at))}</span>
          <span class="session-preview">${escapeHtml(session.last_prompt || session.user_agenda || "No preview yet.")}</span>
        </button>
      `).join("");
      $$("#sessionList .session-card").forEach((button) => {
        button.addEventListener("click", () => {
          closeDrawer();
          loadSession(button.dataset.session);
        });
      });
    }

    async function loadSession(sessionId) {
      const local = state.currentUser ? null : findLocalSession(sessionId);
      try {
        const payload = await apiFetch(`/api/chats/${encodeURIComponent(sessionId)}`);
        const payloadTurns = (payload.history || []).length;
        const localTurns = local ? normalizedHistory(local).length : 0;
        const session = local && localTurns > payloadTurns ? local : payload;
        renderSession(session);
        saveLocalSession(session);
      } catch (_error) {
        if (!local) throw _error;
        renderSession(local);
      }
      renderSessionList();
    }

    function renderSession(session) {
      const normalizedSession = session ? { ...session, history: normalizedHistory(session), messages: session.messages || [] } : null;
      state.activeSession = normalizedSession;
      state.activeSessionId = session ? session.session_id : null;
      state.editingMessageId = null;
      if (normalizedSession) saveLocalSession(normalizedSession);
      $("#currentSessionName").textContent = normalizedSession ? normalizedSession.session_name : "New RAG conversation";
      updateMobileHeader();
      $("#agendaSummary").textContent = normalizedSession && normalizedSession.user_agenda
        ? normalizedSession.user_agenda
        : "The session agenda will build from your prompts.";
      $("#agendaSummary").classList.toggle("empty", !(normalizedSession && normalizedSession.user_agenda));
      const history = normalizedSession ? normalizedSession.history : [];
      renderSessionMessages(history);
      const lastTurn = history.length ? history[history.length - 1] : null;
      renderSuggestions(lastTurn ? (lastTurn.suggestions || []) : []);
      renderCitations(lastTurn ? (lastTurn.citations || []) : []);
      renderSourceSnippets(lastTurn ? (lastTurn.source_snippets || []) : []);
      renderRunMetadata(lastTurn ? lastTurn.metadata : null);
      if (lastTurn && lastTurn.metadata && lastTurn.metadata.guardrails) {
        renderGuardrails(lastTurn.metadata.guardrails);
      } else {
        $("#guardrails").innerHTML = empty("Run metadata will appear after an answer.");
      }
      if (lastTurn && lastTurn.metadata && lastTurn.metadata.pipeline) {
        renderFlow(lastTurn.metadata.pipeline);
      } else {
        $("#flow").innerHTML = "";
      }
      if (lastTurn && lastTurn.metadata && lastTurn.metadata.checkpoints) {
        renderCheckpoints(lastTurn.metadata.checkpoints);
      } else {
        $("#checkpoints").innerHTML = "";
      }
      $("#retrievedChunks").innerHTML = "";
      $("#reflection").textContent = history.length
        ? "Loaded a saved session. Ask a follow-up to continue with its agenda and recent context."
        : "Run a question to see the reflection pass.";
      $("#badges").innerHTML = session
        ? badge(`${history.length} saved exchanges`, "good")
        : badge("Ready");
    }

    function renderSessionMessages(history) {
      const turns = (history || []).map((turn) => ({
        ...turn,
        user_message_id: userMessageId(turn),
        assistant_message_id: assistantMessageId(turn),
        edited: Boolean(turn.edited)
      }));
      if (!turns.length) {
        renderEmptyChat();
        return;
      }
      $("#messages").innerHTML = turns.map((turn, index) => {
        const meta = turn.metadata || {};
        const isLatestTurn = index === turns.length - 1;
        const metaBadges = [
          meta.provider ? badge(meta.provider) : "",
          meta.retrieval_mode ? badge(meta.retrieval_mode) : "",
          typeof meta.confidence === "number" ? badge(`confidence ${meta.confidence}`, meta.needs_review ? "warn" : "good") : "",
          meta.wikipedia_count ? badge(`${meta.wikipedia_count} wiki`, "good") : ""
        ].join("");
        const sourceCount = (turn.citations || []).length || (turn.source_snippets || []).length || 0;
        const editing = state.editingMessageId === turn.user_message_id;
        const userBody = editing
          ? `
            <div class="edit-box">
              <textarea id="editDraft" aria-label="Edit prompt">${escapeHtml(turn.user_prompt)}</textarea>
              <div class="edit-actions">
                <button class="message-action" data-action="save-edit" data-message-id="${escapeHtml(turn.user_message_id)}" type="button">Save & regenerate</button>
                <button class="message-action" data-action="cancel-edit" type="button">Cancel</button>
              </div>
            </div>
          `
          : `
            <div class="message-content">${escapeHtml(turn.user_prompt)}</div>
            <div class="message-actions">
              ${turn.edited ? `<span class="edited-label">Edited</span>` : ""}
              <button class="message-action" data-action="edit-user" data-message-id="${escapeHtml(turn.user_message_id)}" type="button">Edit</button>
              <button class="message-action" data-action="copy" data-copy="${escapeHtml(turn.user_prompt)}" type="button">Copy</button>
            </div>
          `;
        return `
          <article class="message-row user" data-message-id="${escapeHtml(turn.user_message_id)}">
            <div class="message">${userBody}</div>
          </article>
          <article class="message-row assistant" data-message-id="${escapeHtml(turn.assistant_message_id)}">
            <div class="message">
              <div class="message-content">${escapeHtml(turn.ai_response)}</div>
              <div class="message-meta">${metaBadges}</div>
              <div class="message-actions">
                <button class="message-action" data-action="copy" data-copy="${escapeHtml(turn.ai_response)}" type="button">Copy</button>
                <button class="message-action" data-action="details" data-message-id="${escapeHtml(turn.assistant_message_id)}" type="button">${sourceCount ? `${sourceCount} Sources` : "Details"}</button>
                ${isLatestTurn ? `<button class="message-action" data-action="regenerate" type="button">Regenerate</button>` : ""}
              </div>
            </div>
          </article>
        `;
      }).join("");
      const messageList = $("#messages");
      messageList.scrollTop = messageList.scrollHeight;
      bindMessageActions();
      syncResponsivePanels();
    }

    async function copyText(text) {
      try {
        await navigator.clipboard.writeText(text);
      } catch (_error) {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
    }

    function bindMessageActions() {
      $$("#messages [data-action='copy']").forEach((button) => {
        button.addEventListener("click", () => copyText(button.dataset.copy || ""));
      });
      $$("#messages [data-action='edit-user']").forEach((button) => {
        button.addEventListener("click", () => {
          state.editingMessageId = button.dataset.messageId;
          renderSessionMessages(normalizedHistory(state.activeSession));
          const draft = $("#editDraft");
          if (draft) {
            draft.focus();
            draft.selectionStart = draft.value.length;
            draft.selectionEnd = draft.value.length;
          }
        });
      });
      $$("#messages [data-action='cancel-edit']").forEach((button) => {
        button.addEventListener("click", () => {
          state.editingMessageId = null;
          renderSessionMessages(normalizedHistory(state.activeSession));
        });
      });
      $$("#messages [data-action='save-edit']").forEach((button) => {
        button.addEventListener("click", () => editUserMessage(button.dataset.messageId));
      });
      $$("#messages [data-action='regenerate']").forEach((button) => {
        button.addEventListener("click", regenerateLatestResponse);
      });
      $$("#messages [data-action='details']").forEach((button) => {
        button.addEventListener("click", () => openResponseDetails(button.dataset.messageId));
      });
      const draft = $("#editDraft");
      if (draft) {
        draft.addEventListener("keydown", (event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            state.editingMessageId = null;
            renderSessionMessages(normalizedHistory(state.activeSession));
            return;
          }
          const shouldSave = event.key === "Enter"
            && !event.shiftKey
            && !event.altKey
            && !event.ctrlKey
            && !event.metaKey
            && !event.isComposing
            && !isMobileTextInput();
          if (shouldSave) {
            event.preventDefault();
            if (!draft.value.trim() || state.isGenerating) return;
            editUserMessage(state.editingMessageId);
          }
        });
      }
    }

    function openResponseDetails(assistantMessageId) {
      const turn = normalizedHistory(state.activeSession).find((item) => item.assistant_message_id === assistantMessageId);
      if (!turn) return;
      state.detailsTurn = turn;
      const meta = turn.metadata || {};
      const citations = turn.citations || [];
      const snippets = turn.source_snippets || [];
      $("#detailsTitle").textContent = `Response ${turn.turn_id ? turn.turn_id.replace("turn_", "#") : "details"}`;
      $("#detailsBody").innerHTML = `
        <article class="item">
          <header><span>Run Metadata</span><span>${escapeHtml(meta.provider || "RAG")}</span></header>
          <div class="badge-row">
            ${meta.retrieval_mode ? badge(meta.retrieval_mode) : ""}
            ${typeof meta.confidence === "number" ? badge(`confidence ${meta.confidence}`, meta.needs_review ? "warn" : "good") : ""}
            ${meta.needs_review ? badge("needs review", "warn") : badge("finalized", "good")}
          </div>
        </article>
        <article class="item">
          <header><span>Citations</span><span>${citations.length}</span></header>
          <div class="list">
            ${citations.map((citation) => `
              <details class="item citation-card" open>
                <summary>
                  <span>${escapeHtml(citation.page_title || citation.file_name || "Source")}</span>
                  <span>${escapeHtml(citation.source_type || "local")}</span>
                </summary>
                <p>${escapeHtml(citation.snippet || "")}</p>
                ${citation.source_url ? `<p><a href="${escapeHtml(citation.source_url)}" target="_blank" rel="noreferrer">${escapeHtml(citation.source_url)}</a></p>` : ""}
              </details>
            `).join("") || empty("No citations returned for this response.")}
          </div>
        </article>
        <article class="item">
          <header><span>Retrieved Sources</span><span>${snippets.length}</span></header>
          <div class="list">
            ${snippets.map((item) => `
              <details class="item citation-card">
                <summary>
                  <span>${escapeHtml(item.page_title || item.file_name || item.path || "Source")}</span>
                  <span>${escapeHtml(item.source_type || item.source || "local")}</span>
                </summary>
                <p>${escapeHtml(item.snippet || item.text || "")}</p>
                ${item.source_url ? `<p><a href="${escapeHtml(item.source_url)}" target="_blank" rel="noreferrer">${escapeHtml(item.source_url)}</a></p>` : ""}
              </details>
            `).join("") || empty("No retrieved snippets stored for this response.")}
          </div>
        </article>
        ${meta.guardrails ? `
          <article class="item">
            <header><span>Guardrails</span><span>${meta.guardrails.passed ? "Passed" : "Review"}</span></header>
            <div class="badge-row">
              ${badge(`coverage ${meta.guardrails.citation_coverage}`, meta.guardrails.citation_coverage >= 0.5 ? "good" : "warn")}
              ${badge(`floor ${meta.guardrails.retrieval_score_floor_met ? "met" : "missed"}`, meta.guardrails.retrieval_score_floor_met ? "good" : "warn")}
            </div>
            ${(meta.guardrails.risk_flags || []).map((flag) => `<p>${escapeHtml(flag)}</p>`).join("") || "<p>No risk flags triggered.</p>"}
          </article>
        ` : ""}
        ${meta.pipeline && meta.pipeline.length ? `
          <article class="item">
            <header><span>Pipeline</span><span>${meta.pipeline.length} steps</span></header>
            <div class="list">
              ${meta.pipeline.map((step) => `<p><strong>${escapeHtml(step.title || step.stage)}</strong>: ${escapeHtml(step.status)} - ${escapeHtml(step.summary || "")}</p>`).join("")}
            </div>
          </article>
        ` : ""}
      `;
      document.body.classList.add("details-open");
      $("#responseDetails").setAttribute("aria-hidden", "false");
    }

    function closeResponseDetails() {
      document.body.classList.remove("details-open");
      $("#responseDetails").setAttribute("aria-hidden", "true");
    }

    async function editUserMessage(messageId) {
      if (!state.activeSessionId || state.isGenerating) return;
      const draft = $("#editDraft");
      const message = draft ? draft.value.trim() : "";
      if (!message) {
        if (draft) draft.focus();
        return;
      }
      const history = normalizedHistory(state.activeSession);
      const turnIndex = history.findIndex((turn) => turn.user_message_id === messageId || turn.turn_id === messageId);
      if (turnIndex < 0) return;
      const branchThread = threadWithHistory(state.activeSession, history.slice(0, turnIndex));
      const requestBody = {
        ...requestOptions(),
        message,
        session_id: state.activeSessionId,
        session_name: state.activeSession.session_name,
        memory_context: buildThreadMemory(state.activeSession, message, turnIndex)
      };
      setGenerating(true);
      try {
        const payload = await apiFetch(`/api/chats/${encodeURIComponent(state.activeSessionId)}/messages/${encodeURIComponent(messageId)}`, {
          method: "PATCH",
          body: JSON.stringify(requestBody)
        });
        state.editingMessageId = null;
        if (payload.answer !== undefined || payload.finalized_answer !== undefined) {
          const editedTurn = createTurnFromPayload(message, payload, {
            ...history[turnIndex],
            user_prompt: message,
            edited: true
          });
          editedTurn.edited = true;
          const updatedThread = threadWithHistory(state.activeSession, [...history.slice(0, turnIndex), editedTurn], payload);
          persistActiveThread(updatedThread);
          renderSession(state.activeSession);
          renderLatestPayloadDetails(payload);
        } else {
          renderSession(payload);
        }
        await loadHistory();
        renderSessionList();
      } catch (error) {
        $("#badges").innerHTML = badge(error.message, "bad");
        renderSession(state.activeSession || branchThread);
      } finally {
        setGenerating(false);
      }
    }

    async function regenerateLatestResponse() {
      if (!state.activeSessionId || state.isGenerating) return;
      const history = normalizedHistory(state.activeSession);
      if (!history.length) return;
      const turnIndex = history.length - 1;
      const latestTurn = history[turnIndex];
      const requestBody = {
        ...requestOptions(),
        message: latestTurn.user_prompt,
        session_id: state.activeSessionId,
        session_name: state.activeSession.session_name,
        memory_context: buildThreadMemory(state.activeSession, latestTurn.user_prompt, turnIndex)
      };
      setGenerating(true);
      $("#messages").insertAdjacentHTML("beforeend", `
        <article class="message-row assistant loading" data-pending-response="true">
          <div class="message">
            <div class="loading-copy">
              <span class="typing-indicator" aria-hidden="true"><span></span><span></span><span></span></span>
              <span>Regenerating the latest answer with thread memory...</span>
            </div>
          </div>
        </article>
      `);
      $("#messages").scrollTop = $("#messages").scrollHeight;
      try {
        const payload = await apiFetch(`/api/chats/${encodeURIComponent(state.activeSessionId)}/regenerate`, {
          method: "POST",
          body: JSON.stringify(requestBody)
        });
        if (payload.answer !== undefined || payload.finalized_answer !== undefined) {
          const regeneratedTurn = createTurnFromPayload(latestTurn.user_prompt, payload, latestTurn);
          regeneratedTurn.edited = Boolean(latestTurn.edited);
          const updatedThread = threadWithHistory(state.activeSession, [...history.slice(0, turnIndex), regeneratedTurn], payload);
          persistActiveThread(updatedThread);
          renderSession(state.activeSession);
          renderLatestPayloadDetails(payload);
        } else {
          renderSession(payload);
        }
        await loadHistory();
        renderSessionList();
      } catch (error) {
        replacePendingWithError(error.message);
        $("#badges").innerHTML = badge(error.message, "bad");
      } finally {
        setGenerating(false);
      }
    }

    function renderRunMetadata(metadata) {
      if (!metadata) {
        $("#runMetadata").innerHTML = badge("No run yet");
        return;
      }
      $("#runMetadata").innerHTML = [
        metadata.provider ? badge(metadata.provider) : "",
        metadata.retrieval_mode ? badge(metadata.retrieval_mode) : "",
        typeof metadata.confidence === "number" ? badge(`confidence ${metadata.confidence}`, metadata.needs_review ? "warn" : "good") : "",
        metadata.citation_count !== undefined ? badge(`${metadata.citation_count} citations`) : "",
        metadata.retrieved_chunk_count !== undefined ? badge(`${metadata.retrieved_chunk_count} chunks`) : "",
        metadata.wikipedia_count ? badge(`${metadata.wikipedia_count} wiki`, "good") : ""
      ].join("");
    }

    function renderSuggestions(suggestions) {
      if (!suggestions || !suggestions.length) {
        $("#suggestions").innerHTML = empty("Suggestions appear after an answer.");
        return;
      }
      $("#suggestions").innerHTML = suggestions.map((suggestion) => `
        <button class="suggestion-button" type="button">${escapeHtml(suggestion)}</button>
      `).join("");
      bindPromptButtons();
    }

    function renderChat(payload, prompt = "", baseThread = null) {
      state.last = payload;
      state.editingMessageId = null;
      const thread = baseThread || state.activeSession || createLocalThread(prompt || "RAG conversation");
      const history = normalizedHistory(thread);
      const responsePrompt = prompt
        || (payload.history && payload.history.length ? payload.history[payload.history.length - 1].user_prompt : "")
        || "";
      const nextTurn = createTurnFromPayload(responsePrompt, payload);
      const payloadHistory = normalizedHistory(payload);
      const nextHistory = state.currentUser && payloadHistory.length >= history.length + 1
        ? payloadHistory
        : [...history, nextTurn];
      const updatedThread = threadWithHistory(thread, nextHistory, payload);
      persistActiveThread(updatedThread);
      $("#currentSessionName").textContent = state.activeSession.session_name;
      updateMobileHeader("Saved");
      $("#agendaSummary").textContent = state.activeSession.user_agenda || "The session agenda will build from your prompts.";
      $("#agendaSummary").classList.toggle("empty", !state.activeSession.user_agenda);
      renderSessionMessages(state.activeSession.history || []);
      renderLatestPayloadDetails(payload);
      renderSessionList();
      syncResponsivePanels();
    }

    function renderCitations(citations) {
      $("#citations").innerHTML = citations.map((citation) => `
        <details class="item citation-card" open>
          <summary>
            <span>${citation.rank}. ${escapeHtml(citation.page_title || citation.file_name)}</span>
            <span>${escapeHtml(citation.source_type || "local")} | ${citation.score}</span>
          </summary>
          <p>${escapeHtml(citation.snippet || "")}</p>
          ${citation.source_url ? `<p><a href="${escapeHtml(citation.source_url)}" target="_blank" rel="noreferrer">${escapeHtml(citation.source_url)}</a></p>` : ""}
          <div class="attention">${(citation.sentence_attention || []).map((item) => `<div>${item.score}: ${escapeHtml(item.sentence)}</div>`).join("")}</div>
        </details>
      `).join("") || empty("No citations returned.");
      syncResponsivePanels();
    }

    function renderSourceSnippets(snippets) {
      $("#sourceSnippets").innerHTML = snippets.map((item) => {
        const text = item.snippet || item.text || "";
        return `
          <details class="item citation-card" open>
            <summary><span>${item.rank}. ${escapeHtml(item.page_title || item.file_name)}</span><span>${escapeHtml(item.source_type || item.source || item.path || "")}</span></summary>
            <p class="source-snippet">${escapeHtml(text)}</p>
            ${item.source_url ? `<p><a href="${escapeHtml(item.source_url)}" target="_blank" rel="noreferrer">${escapeHtml(item.source_url)}</a></p>` : ""}
          </details>
        `;
      }).join("") || empty("Source snippets appear after retrieval.");
      syncResponsivePanels();
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

    async function bootstrapAuth() {
      const payload = await apiFetch("/api/auth/me");
      if (!payload.authenticated || !payload.user) {
        showAuthGate();
        return;
      }
      showAuthenticatedApp(payload.user);
      await loadRuntime();
      renderSessionList();
    }

    $$(".auth-tab").forEach((button) => {
      button.addEventListener("click", () => setAuthMode(button.dataset.authMode));
    });

    $("#authForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      $("#authSubmit").disabled = true;
      $("#authError").textContent = "";
      try {
        const payload = await apiFetch(state.authMode === "signup" ? "/api/auth/signup" : "/api/auth/login", {
          method: "POST",
          body: JSON.stringify({
            email: $("#authEmail").value.trim(),
            password: $("#authPassword").value,
            display_name: $("#authName").value.trim()
          })
        });
        showAuthenticatedApp(payload.user);
        await loadRuntime();
        renderSession(null);
        renderSessionList();
        $("#question").focus();
      } catch (error) {
        $("#authError").textContent = error.message;
      } finally {
        $("#authSubmit").disabled = false;
      }
    });

    $("#logoutButton").addEventListener("click", async () => {
      await apiFetch("/api/auth/logout", { method: "POST" }).catch(() => null);
      showAuthGate();
    });

    $("#askForm").addEventListener("submit", (event) => {
      event.preventDefault();
      submitPrompt();
    });

    $("#question").addEventListener("keydown", (event) => {
      const shouldSubmit = event.key === "Enter"
        && !event.shiftKey
        && !event.altKey
        && !event.ctrlKey
        && !event.metaKey
        && !event.isComposing
        && !isMobileTextInput();
      if (shouldSubmit) {
        event.preventDefault();
        if (!$("#question").value.trim() || state.isGenerating) return;
        if ($("#askForm").requestSubmit) {
          $("#askForm").requestSubmit();
        } else {
          submitPrompt();
        }
      }
    });

    $("#clearButton").addEventListener("click", () => {
      $("#question").value = "";
      $("#question").focus();
    });

    $("#chatSearch").addEventListener("input", (event) => {
      state.chatSearch = event.target.value;
      renderSessionList();
    });

    $("#newSessionButton").addEventListener("click", async () => {
      closeDrawer();
      $("#newSessionButton").disabled = true;
      try {
        const payload = state.currentUser
          ? await apiFetch("/api/chats", { method: "POST", body: JSON.stringify({ session_name: "New RAG conversation" }) })
          : null;
        renderSession(payload);
        await loadHistory();
        renderSessionList();
        $("#question").value = "";
        $("#question").focus();
      } finally {
        $("#newSessionButton").disabled = false;
      }
    });

    $("#cloneSessionButton").addEventListener("click", async () => {
      if (!state.activeSessionId) return;
      $("#cloneSessionButton").disabled = true;
      try {
        const payload = await apiFetch(`/api/chat/history/${encodeURIComponent(state.activeSessionId)}/clone`, {
          method: "POST"
        });
        renderSession(payload);
        await loadHistory();
        renderSessionList();
      } finally {
        $("#cloneSessionButton").disabled = false;
      }
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
          <table class="table responsive-table">
            <thead><tr><th>Sample</th><th>Answer F1</th><th>Context Recall</th><th>Citation Hit Rate</th><th>Confidence</th></tr></thead>
            <tbody>
              ${payload.rows.map((row) => `
                <tr>
                  <td data-label="Sample">${escapeHtml(row.sample_id)}</td>
                  <td data-label="Answer F1">${row.answer_f1}</td>
                  <td data-label="Context Recall">${row.context_recall}</td>
                  <td data-label="Citation Hit Rate">${row.citation_hit_rate}</td>
                  <td data-label="Confidence">${row.confidence}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        `;
      } finally {
        $("#runEval").disabled = false;
      }
    });

    $("#drawerToggle").addEventListener("click", openDrawer);
    $("#drawerClose").addEventListener("click", closeDrawer);
    $("#drawerOverlay").addEventListener("click", closeDrawer);
    $("#detailsClose").addEventListener("click", closeResponseDetails);
    $("#detailsBackdrop").addEventListener("click", closeResponseDetails);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeDrawer();
        closeResponseDetails();
      }
    });
    document.addEventListener("toggle", (event) => {
      if (!state.responsiveReady) return;
      if (event.target.matches(".insight-panel, .composer-options")) {
        event.target.dataset.touched = "true";
      }
    }, true);
    window.addEventListener("resize", syncResponsivePanels);
    syncResponsivePanels();
    state.responsiveReady = true;

    setAuthMode("login");
    bootstrapAuth().catch((error) => {
      $("#badges").innerHTML = badge(error.message, "bad");
      showAuthGate();
    });
    bindPromptButtons();
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
        "auth": {
            "signup": "/api/auth/signup",
            "login": "/api/auth/login",
            "logout": "/api/auth/logout",
            "me": "/api/auth/me",
        },
        "chat": "/api/chat",
        "chat_history": "/api/chat/history",
        "chats": "/api/chats",
        "chat_messages": "/api/chats/{session_id}/messages",
        "chat_message_edit": "/api/chats/{session_id}/messages/{message_id}",
        "chat_regenerate": "/api/chats/{session_id}/regenerate",
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
        "auth": {
            "cookie_name": AUTH_COOKIE_NAME,
            "session_secret_configured": bool(os.getenv("SESSION_SECRET") or os.getenv("AUTH_SECRET")),
        },
        "persistence": {
            "database": "postgres" if account_store.is_postgres else "sqlite",
            "database_url_configured": bool(os.getenv("DATABASE_URL")),
        },
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


@app.post("/api/auth/signup", response_model=AuthResponse)
def auth_signup(payload: AuthSignupRequest, response: Response) -> AuthResponse:
    try:
        user = account_store.create_user(
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _set_auth_cookie(response, user)
    return AuthResponse(
        authenticated=True,
        user=_public_user(user),
        message="Account created. Welcome to Agentic RAG.",
    )


@app.post("/api/auth/login", response_model=AuthResponse)
def auth_login(payload: AuthLoginRequest, response: Response) -> AuthResponse:
    user = account_store.verify_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    _set_auth_cookie(response, user)
    return AuthResponse(
        authenticated=True,
        user=_public_user(user),
        message="Signed in.",
    )


@app.post("/api/auth/logout", response_model=AuthResponse)
def auth_logout(response: Response) -> AuthResponse:
    _clear_auth_cookie(response)
    return AuthResponse(authenticated=False, user=None, message="Signed out.")


@app.get("/api/auth/me", response_model=AuthResponse)
def auth_me(request: Request) -> AuthResponse:
    user = current_user_from_request(request)
    if not user:
        return AuthResponse(authenticated=False, user=None, message="Signed out.")
    return AuthResponse(authenticated=True, user=_public_user(user), message="Signed in.")


@app.get("/api/chat/history", response_model=ChatHistoryList)
def chat_history(request: Request = None) -> ChatHistoryList:
    user = current_user_from_request(request)
    if request is not None and not user:
        raise HTTPException(status_code=401, detail="Please log in to view saved chats.")
    sessions = account_store.list_threads(user["id"]) if user else chat_history_service.list_sessions()
    return ChatHistoryList(sessions=[_session_summary(session) for session in sessions])


@app.get("/api/chat/history/{session_id}", response_model=ChatSessionView)
def chat_session(session_id: str, request: Request = None) -> ChatSessionView:
    user = current_user_from_request(request)
    if request is not None and not user:
        raise HTTPException(status_code=401, detail="Please log in to open this chat.")
    session = account_store.get_thread(user["id"], session_id) if user else chat_history_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return _session_view(session)


@app.post("/api/chat/history/{session_id}/clone", response_model=ChatSessionView)
def clone_chat_session(session_id: str, request: Request = None) -> ChatSessionView:
    user = current_user_from_request(request)
    if request is not None and not user:
        raise HTTPException(status_code=401, detail="Please log in to clone chats.")
    if user:
        session = account_store.clone_thread(user["id"], session_id)
        saved = bool(session)
    else:
        session, saved = chat_history_service.clone_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    session.setdefault("metadata", {})["cloned"] = saved
    return _session_view(session)


@app.get("/api/chats", response_model=ChatHistoryList)
def chats(request: Request = None) -> ChatHistoryList:
    return chat_history(request)


@app.post("/api/chats", response_model=ChatSessionView)
def create_chat(
    payload: ChatSessionCreate | None = None,
    request: Request = None,
    session_name: str | None = Query(default=None, max_length=120),
) -> ChatSessionView:
    user = current_user_from_request(request)
    name = (payload.session_name if payload else None) or session_name
    if request is not None and not user:
        raise HTTPException(status_code=401, detail="Please log in to create chats.")
    if user:
        session = account_store.create_thread(user["id"], title=name)
        saved = True
    else:
        session, saved = chat_history_service.create_session(session_name=name)
    session.setdefault("metadata", {})["created"] = saved
    return _session_view(session)


@app.get("/api/chats/{session_id}", response_model=ChatSessionView)
def get_chat(session_id: str, request: Request = None) -> ChatSessionView:
    return chat_session(session_id, request)


@app.patch("/api/chats/{session_id}", response_model=ChatSessionView)
def update_chat(
    session_id: str,
    payload: ChatSessionUpdate,
    request: Request = None,
) -> ChatSessionView:
    user = current_user_from_request(request)
    if request is not None and not user:
        raise HTTPException(status_code=401, detail="Please log in to update chats.")
    if user:
        session = account_store.update_thread(user["id"], session_id, session_name=payload.session_name)
    else:
        session, _saved = chat_history_service.update_session(
            session_id,
            session_name=payload.session_name,
        )
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return _session_view(session)


@app.post("/api/chats/{session_id}/messages", response_model=ChatResponse)
def append_chat_message(
    session_id: str,
    payload: ChatRequest,
    request: Request = None,
) -> ChatResponse:
    return chat_completion(_copy_chat_request(payload, session_id=session_id), request)


@app.patch("/api/chats/{session_id}/messages/{message_id}", response_model=ChatSessionView)
def edit_chat_message(
    session_id: str,
    message_id: str,
    payload: ChatMessageEditRequest,
    request: Request = None,
) -> ChatSessionView:
    user = current_user_from_request(request)
    if request is not None and not user:
        raise HTTPException(status_code=401, detail="Please log in to edit chats.")
    if user:
        branch_session, turn_index = account_store.session_before_user_message(user["id"], session_id, message_id)
    else:
        branch_session, turn_index = chat_history_service.session_before_user_message(session_id, message_id)
    if not branch_session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    if turn_index is None:
        raise HTTPException(status_code=404, detail="User message not found.")
    prompt = payload.message.strip()
    chat_payload = _action_to_chat_request(
        prompt,
        session_id,
        payload,
        session_name=branch_session.get("session_name"),
    )
    response, _projected_agenda, suggestions, metadata = _generate_chat_turn(chat_payload, branch_session)
    if user:
        session, saved = account_store.replace_turn_branch(
            user_id=user["id"],
            session_id=session_id,
            turn_index=turn_index,
            user_prompt=prompt,
            ai_response=response.finalized_answer or response.answer,
            metadata=metadata,
            citations=response.citations,
            retrieved_chunks=response.retrieved_chunks,
            suggestions=suggestions,
            edited=True,
        )
    else:
        session, saved = chat_history_service.replace_turn_branch(
            session_id=session_id,
            turn_index=turn_index,
            user_prompt=prompt,
            ai_response=response.finalized_answer or response.answer,
            metadata=metadata,
            citations=response.citations,
            retrieved_chunks=response.retrieved_chunks,
            suggestions=suggestions,
            edited=True,
        )
    if not session:
        raise HTTPException(status_code=404, detail="User message not found.")
    session.setdefault("metadata", {})["last_edit_saved"] = saved
    return _session_view(session)


@app.post("/api/chats/{session_id}/regenerate", response_model=ChatSessionView)
def regenerate_chat_response(
    session_id: str,
    payload: ChatRegenerateRequest,
    request: Request = None,
) -> ChatSessionView:
    user = current_user_from_request(request)
    if request is not None and not user:
        raise HTTPException(status_code=401, detail="Please log in to regenerate chats.")
    if user:
        branch_session, turn_index = account_store.session_before_latest_turn(user["id"], session_id)
    else:
        branch_session, turn_index = chat_history_service.session_before_latest_turn(session_id)
    if not branch_session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    if turn_index is None:
        raise HTTPException(status_code=400, detail="No assistant response to regenerate.")
    original_session = account_store.get_thread(user["id"], session_id) if user else chat_history_service.get_session(session_id)
    if not original_session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    history = list(original_session.get("history") or [])
    latest_turn = _normalize_turn(history[turn_index])
    prompt = str(latest_turn.get("user_prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Latest user prompt is empty.")
    chat_payload = _action_to_chat_request(
        prompt,
        session_id,
        payload,
        session_name=branch_session.get("session_name"),
    )
    response, _projected_agenda, suggestions, metadata = _generate_chat_turn(chat_payload, branch_session)
    if user:
        session, saved = account_store.replace_turn_branch(
            user_id=user["id"],
            session_id=session_id,
            turn_index=turn_index,
            user_prompt=prompt,
            ai_response=response.finalized_answer or response.answer,
            metadata=metadata,
            citations=response.citations,
            retrieved_chunks=response.retrieved_chunks,
            suggestions=suggestions,
            edited=bool(latest_turn.get("edited")),
        )
    else:
        session, saved = chat_history_service.replace_turn_branch(
            session_id=session_id,
            turn_index=turn_index,
            user_prompt=prompt,
            ai_response=response.finalized_answer or response.answer,
            metadata=metadata,
            citations=response.citations,
            retrieved_chunks=response.retrieved_chunks,
            suggestions=suggestions,
            edited=bool(latest_turn.get("edited")),
        )
    if not session:
        raise HTTPException(status_code=404, detail="Latest turn not found.")
    session.setdefault("metadata", {})["last_regenerate_saved"] = saved
    return _session_view(session)


@app.delete("/api/chats/{session_id}")
def delete_chat(session_id: str, request: Request = None) -> dict[str, Any]:
    user = current_user_from_request(request)
    if request is not None and not user:
        raise HTTPException(status_code=401, detail="Please log in to delete chats.")
    deleted = account_store.delete_thread(user["id"], session_id) if user else chat_history_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return {"deleted": True, "session_id": session_id}


@app.post("/api/chat", response_model=ChatResponse)
def chat_completion(payload: ChatRequest, request: Request = None) -> ChatResponse:
    user = current_user_from_request(request)
    active_session = None
    if user:
        active_session = account_store.get_thread(user["id"], payload.session_id) if payload.session_id else None
    elif payload.session_id:
        active_session = chat_history_service.get_session(payload.session_id)
    response, projected_agenda, suggestions, metadata = _generate_chat_turn(payload, active_session)
    if user:
        session, saved = account_store.append_turn(
            user_id=user["id"],
            session_id=payload.session_id,
            session_name=payload.session_name,
            user_prompt=payload.message.strip(),
            ai_response=response.finalized_answer or response.answer,
            metadata=metadata,
            citations=response.citations,
            retrieved_chunks=response.retrieved_chunks,
            suggestions=suggestions,
        )
    else:
        session, saved = chat_history_service.append_turn(
            session_id=payload.session_id,
            session_name=payload.session_name,
            user_prompt=payload.message.strip(),
            ai_response=response.finalized_answer or response.answer,
            metadata=metadata,
            citations=response.citations,
            retrieved_chunks=response.retrieved_chunks,
            suggestions=suggestions,
        )
    response.session_id = session["session_id"]
    response.session_name = session["session_name"]
    session_view = _session_view(session)
    response.history = session_view.history
    response.messages = session_view.messages
    response.agenda_summary = session.get("user_agenda", projected_agenda)
    response.memory_summary = session.get("memory_summary", "")
    response.suggestions = suggestions
    response.chat_saved = saved
    return response


@app.post("/api/query", response_model=ChatResponse)
def query(payload: ChatRequest, request: Request = None) -> ChatResponse:
    return chat_completion(payload, request)


@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest) -> EvaluationResponse:
    return _run_evaluation(payload)
