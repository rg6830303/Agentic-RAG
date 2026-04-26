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
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


SERVICE_NAME = "Advanced Agentic RAG"
SERVICE_VERSION = "0.2.0"
ROOT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = ROOT_DIR / "data" / "sample_corpus"
VERCEL_DEPLOYMENT = bool(os.getenv("VERCEL"))
SUPPORTED_CORPUS_EXTENSIONS = {
    ".csv",
    ".json",
    ".md",
    ".sql",
    ".txt",
}


@dataclass(slots=True)
class CorpusChunk:
    chunk_id: str
    file_name: str
    relative_path: str
    text: str
    ordinal: int
    tokens: list[str]


@dataclass(slots=True)
class CorpusIndex:
    chunks: list[CorpusChunk]
    doc_lengths: list[int]
    avg_doc_length: float
    postings: dict[str, list[tuple[int, int]]]
    source_files: list[str]


class Citation(BaseModel):
    file_name: str
    path: str
    rank: int
    score: float
    snippet: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12_000)
    top_k: int = Field(default=5, ge=1, le=10)
    use_generation: bool = True
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=700, ge=100, le=2_000)


class ChatResponse(BaseModel):
    answer: str
    provider: str
    confidence: float
    citations: list[Citation]
    corpus: dict[str, Any]


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="Vercel-native UI and API for a bundled-corpus RAG system.",
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
            payload = json.loads(text)
            return json.dumps(payload, indent=2, ensure_ascii=True)
        except json.JSONDecodeError:
            return text
    return text


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
    if not CORPUS_DIR.exists():
        return CorpusIndex([], [], 0.0, {}, [])

    for path in sorted(CORPUS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_CORPUS_EXTENSIONS:
            continue
        relative_path = path.relative_to(ROOT_DIR).as_posix()
        source_files.append(relative_path)
        text = _read_corpus_file(path)
        for ordinal, chunk_text in enumerate(_split_text(text)):
            chunks.append(
                CorpusChunk(
                    chunk_id=f"{relative_path}:{ordinal}",
                    file_name=path.name,
                    relative_path=relative_path,
                    text=chunk_text,
                    ordinal=ordinal,
                    tokens=_tokenize(chunk_text),
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
    )


def _search_corpus(query: str, top_k: int) -> list[tuple[CorpusChunk, float]]:
    index = _load_corpus()
    query_terms = _tokenize(query)
    if not query_terms or not index.chunks:
        return []

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

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [(index.chunks[doc_index], float(score)) for doc_index, score in ranked]


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _extractive_answer(question: str, hits: list[tuple[CorpusChunk, float]]) -> str:
    query_terms = set(_tokenize(question))
    if not hits:
        return "I could not find matching context in the deployed corpus."

    scored_sentences: list[tuple[int, str]] = []
    for chunk, _score in hits:
        for sentence in _sentences(chunk.text):
            overlap = len(query_terms.intersection(_tokenize(sentence)))
            if overlap:
                scored_sentences.append((overlap, sentence))

    if not scored_sentences:
        excerpts = [chunk.text[:260].strip() for chunk, _score in hits[:2]]
        return " ".join(excerpts)

    selected = [
        sentence
        for _score, sentence in sorted(scored_sentences, key=lambda item: item[0], reverse=True)[:3]
    ]
    return " ".join(selected)


def _context_prompt(hits: list[tuple[CorpusChunk, float]]) -> str:
    blocks = []
    for rank, (chunk, score) in enumerate(hits, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{rank}] {chunk.relative_path}",
                    f"score={score:.3f}",
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(blocks)


def _generate_answer(
    question: str,
    hits: list[tuple[CorpusChunk, float]],
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
                    "Answer using only the provided RAG context. Be concise, "
                    "cite source numbers in brackets, and say when the context "
                    "does not contain enough evidence."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRAG context:\n{_context_prompt(hits)}",
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


def _confidence(hits: list[tuple[CorpusChunk, float]]) -> float:
    if not hits:
        return 0.0
    top_score = hits[0][1]
    return round(min(0.95, max(0.25, top_score / (top_score + 3.0))), 2)


def _citation(rank: int, chunk: CorpusChunk, score: float) -> Citation:
    snippet = html.unescape(chunk.text[:520].strip())
    return Citation(
        file_name=chunk.file_name,
        path=chunk.relative_path,
        rank=rank,
        score=round(score, 4),
        snippet=snippet,
    )


def _corpus_summary() -> dict[str, Any]:
    index = _load_corpus()
    return {
        "source_dir": CORPUS_DIR.relative_to(ROOT_DIR).as_posix()
        if CORPUS_DIR.exists()
        else "missing",
        "source_count": len(index.source_files),
        "chunk_count": len(index.chunks),
        "sources": index.source_files,
    }


APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Advanced Agentic RAG</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f6f2;
      --panel: #ffffff;
      --ink: #161616;
      --muted: #666a73;
      --line: #d9d6cc;
      --teal: #0f766e;
      --teal-soft: #e3f3f1;
      --amber: #b7791f;
      --rose: #b42342;
      --slate: #2f3a4a;
      --shadow: 0 14px 38px rgba(25, 28, 33, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(280px, 360px) 1fr;
    }
    aside {
      border-right: 1px solid var(--line);
      padding: 28px 24px;
      background: #eeece5;
    }
    main {
      padding: 28px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 18px;
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 26px; line-height: 1.12; font-weight: 750; }
    h2 { font-size: 14px; text-transform: uppercase; color: var(--muted); }
    .brand {
      display: grid;
      gap: 10px;
      margin-bottom: 24px;
    }
    .status-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin: 18px 0 24px;
    }
    .metric {
      background: rgba(255,255,255,0.6);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 72px;
    }
    .metric strong {
      display: block;
      font-size: 24px;
      line-height: 1.1;
      color: var(--slate);
    }
    .metric span {
      display: block;
      margin-top: 5px;
      font-size: 12px;
      color: var(--muted);
    }
    .source-list {
      display: grid;
      gap: 8px;
      max-height: 260px;
      overflow: auto;
      padding-right: 4px;
    }
    .source-pill {
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.58);
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 13px;
      color: var(--slate);
      overflow-wrap: anywhere;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(320px, 0.85fr) minmax(360px, 1.15fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .query {
      padding: 18px;
      display: grid;
      gap: 14px;
    }
    label {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    textarea {
      width: 100%;
      min-height: 190px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      font: inherit;
      color: var(--ink);
      background: #fbfaf7;
    }
    textarea:focus, input:focus {
      outline: 3px solid var(--teal-soft);
      border-color: var(--teal);
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      align-items: end;
    }
    input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      font: inherit;
      background: #fbfaf7;
    }
    .switch {
      display: flex;
      gap: 10px;
      align-items: center;
      min-height: 42px;
      color: var(--slate);
    }
    .switch input { width: 18px; height: 18px; accent-color: var(--teal); }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    button {
      border: 0;
      border-radius: 8px;
      padding: 12px 16px;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
      background: var(--slate);
      color: #fff;
      min-height: 44px;
    }
    button.secondary {
      background: #ece8dd;
      color: var(--slate);
      border: 1px solid var(--line);
    }
    button:disabled { opacity: 0.62; cursor: wait; }
    .answer {
      min-height: 520px;
      padding: 18px;
      display: grid;
      gap: 16px;
    }
    .answer-text {
      border-left: 4px solid var(--teal);
      padding: 2px 0 2px 14px;
      line-height: 1.65;
      white-space: pre-wrap;
    }
    .meta-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      color: var(--slate);
      background: #fbfaf7;
    }
    .badge.warn { color: var(--amber); border-color: #e7c98d; }
    .badge.error { color: var(--rose); border-color: #e8a4b3; }
    .citations {
      display: grid;
      gap: 10px;
    }
    .citation {
      border-top: 1px solid var(--line);
      padding-top: 12px;
      display: grid;
      gap: 6px;
    }
    .citation header {
      display: flex;
      gap: 8px;
      justify-content: space-between;
      color: var(--slate);
      font-weight: 750;
      font-size: 14px;
    }
    .citation p {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .empty {
      color: var(--muted);
      line-height: 1.6;
    }
    @media (max-width: 940px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .workspace { grid-template-columns: 1fr; }
      main { padding: 18px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <section class="brand">
        <h1>Advanced Agentic RAG</h1>
        <p class="empty">Vercel deployment</p>
      </section>
      <section>
        <h2>Runtime</h2>
        <div class="status-grid">
          <div class="metric"><strong id="sourceCount">-</strong><span>sources</span></div>
          <div class="metric"><strong id="chunkCount">-</strong><span>chunks</span></div>
          <div class="metric"><strong id="modelState">-</strong><span>generation</span></div>
          <div class="metric"><strong id="deployState">-</strong><span>host</span></div>
        </div>
      </section>
      <section>
        <h2>Corpus</h2>
        <div id="sources" class="source-list"></div>
      </section>
    </aside>
    <main>
      <section class="workspace">
        <form class="panel query" id="askForm">
          <label>
            Question
            <textarea id="question" name="question" placeholder="Ask about the deployed corpus"></textarea>
          </label>
          <div class="controls">
            <label>
              Top K
              <input id="topK" type="number" min="1" max="10" value="5">
            </label>
            <label class="switch">
              <input id="useGeneration" type="checkbox" checked>
              Azure synthesis
            </label>
          </div>
          <div class="actions">
            <button id="askButton" type="submit">Ask</button>
            <button class="secondary" id="clearButton" type="button">Clear</button>
          </div>
        </form>
        <section class="panel answer">
          <div class="meta-row" id="badges">
            <span class="badge">Ready</span>
          </div>
          <div id="answer" class="answer-text empty">The answer will appear here.</div>
          <section>
            <h2>Citations</h2>
            <div id="citations" class="citations"></div>
          </section>
        </section>
      </section>
    </main>
  </div>
  <script>
    const form = document.querySelector("#askForm");
    const question = document.querySelector("#question");
    const topK = document.querySelector("#topK");
    const useGeneration = document.querySelector("#useGeneration");
    const askButton = document.querySelector("#askButton");
    const clearButton = document.querySelector("#clearButton");
    const answer = document.querySelector("#answer");
    const citations = document.querySelector("#citations");
    const badges = document.querySelector("#badges");

    const setBadges = (items) => {
      badges.innerHTML = items.map((item) => `<span class="badge ${item.kind || ""}">${item.text}</span>`).join("");
    };

    const escapeHtml = (value) => String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

    async function loadRuntime() {
      const [runtimeResponse, corpusResponse] = await Promise.all([
        fetch("/api/runtime"),
        fetch("/api/corpus")
      ]);
      const runtime = await runtimeResponse.json();
      const corpus = await corpusResponse.json();
      document.querySelector("#sourceCount").textContent = corpus.source_count;
      document.querySelector("#chunkCount").textContent = corpus.chunk_count;
      document.querySelector("#modelState").textContent = runtime.azure_openai.chat_configured ? "Azure" : "Local";
      document.querySelector("#deployState").textContent = runtime.vercel ? "Vercel" : "Local";
      document.querySelector("#sources").innerHTML = corpus.sources
        .map((source) => `<div class="source-pill">${escapeHtml(source)}</div>`)
        .join("");
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = question.value.trim();
      if (!message) {
        question.focus();
        return;
      }
      askButton.disabled = true;
      answer.textContent = "Retrieving...";
      answer.classList.add("empty");
      citations.innerHTML = "";
      setBadges([{ text: "Working" }]);
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message,
            top_k: Number(topK.value),
            use_generation: useGeneration.checked
          })
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Request failed");
        }
        answer.textContent = payload.answer || "No answer returned.";
        answer.classList.remove("empty");
        setBadges([
          { text: payload.provider },
          { text: `confidence ${payload.confidence}` },
          { text: `${payload.citations.length} citations` }
        ]);
        citations.innerHTML = payload.citations.map((citation) => `
          <article class="citation">
            <header>
              <span>${citation.rank}. ${escapeHtml(citation.file_name)}</span>
              <span>${citation.score}</span>
            </header>
            <p>${escapeHtml(citation.snippet)}</p>
          </article>
        `).join("");
      } catch (error) {
        answer.textContent = error.message;
        answer.classList.add("empty");
        setBadges([{ text: "Error", kind: "error" }]);
      } finally {
        askButton.disabled = false;
      }
    });

    clearButton.addEventListener("click", () => {
      question.value = "";
      answer.textContent = "The answer will appear here.";
      answer.classList.add("empty");
      citations.innerHTML = "";
      setBadges([{ text: "Ready" }]);
      question.focus();
    });

    loadRuntime().catch(() => setBadges([{ text: "Runtime unavailable", kind: "error" }]));
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
        "chat": "/api/chat",
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
        "vercel": VERCEL_DEPLOYMENT,
        "azure_openai": _azure_status(),
        "corpus": _corpus_summary(),
    }


@app.get("/api/corpus")
def corpus_info() -> dict[str, Any]:
    return _corpus_summary()


@app.post("/api/chat", response_model=ChatResponse)
def chat_completion(payload: ChatRequest) -> ChatResponse:
    question = payload.message.strip()
    hits = _search_corpus(question, payload.top_k)
    if payload.use_generation and _azure_status()["chat_configured"] and hits:
        answer = _generate_answer(
            question,
            hits,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
        provider = "azure_openai_rag"
    else:
        answer = _extractive_answer(question, hits)
        provider = "local_rag"

    return ChatResponse(
        answer=answer,
        provider=provider,
        confidence=_confidence(hits),
        citations=[
            _citation(rank, chunk, score)
            for rank, (chunk, score) in enumerate(hits, start=1)
        ],
        corpus=_corpus_summary(),
    )


@app.post("/api/query", response_model=ChatResponse)
def query(payload: ChatRequest) -> ChatResponse:
    return chat_completion(payload)
