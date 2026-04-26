# Advanced Agentic RAG

This repository contains a complete local-first Agentic RAG system built with Python and Streamlit for Windows CMD usage, plus a slim FastAPI surface for Vercel. The full app keeps ingestion, chunking, SQLite docstore persistence, FAISS indexing, BM25 retrieval, orchestration, evaluation artifacts, and UI logic on local disk. Azure OpenAI is used only for chat and embeddings when the configured `.env`, Streamlit secrets, or Vercel environment variables support those capabilities.

## Highlights

- Thin `streamlit_app.py` bootstrap with multipage Streamlit UI under `pages/`
- Vercel-compatible FastAPI entrypoint at `app.py`, also re-exported from `api/index.py`
- Local ingestion for PDF, DOCX, text, markdown, code, CSV, JSON, and SQL
- Fixed, semantic, recursive, adaptive, hierarchical, and auto-selected chunking
- SQLite docstore, persisted FAISS, and persisted standalone BM25
- Self-RAG reflection, checkpoints, HITL approval flows, and sentence-level relevance
- Local heuristic evaluation plus an optional RAGAS/LiteLLM adapter surface
- Original NCERT Class 12 Physics-inspired sample corpus and golden evaluation dataset

## Quick Start

1. Create and activate a virtual environment.
2. Install local app dependencies with `pip install -r requirements-local.txt`.
3. Configure Azure OpenAI with either:
   - local `.env` values in the repository root, or
   - Streamlit secrets for deployment and `.streamlit/secrets.toml` for local Streamlit-based secret loading
4. Run `python -m streamlit run streamlit_app.py` from Windows CMD.

Detailed Windows steps are in `WINDOWS_RUN.md`.

## Vercel Deployment

Vercel's Python runtime expects a serverless-compatible ASGI application. The Streamlit UI and local RAG workspace depend on a writable local filesystem, so the Vercel deployment intentionally exposes a lightweight FastAPI API from `app.py`. The same app is re-exported from `api/index.py` for compatibility.

`requirements.txt` is intentionally slim for Vercel. Use `requirements-local.txt` for the full Streamlit app.

`vercel.json` intentionally does not define a `functions` block. Vercel's FastAPI framework detector owns the root `app.py` entrypoint, and adding a manual `functions` pattern can cause unmatched function pattern errors.

`pyproject.toml` also points Vercel to the root API app with:

```toml
[project.scripts]
app = "app:app"
```

Deploy the repository with Vercel after committing these files. The deployment exposes:

- `/` for service metadata
- `/docs` for FastAPI's OpenAPI UI
- `/api/health` for health checks
- `/api/runtime` for deployment diagnostics without secret values
- `/api/chat` for optional Azure OpenAI chat calls

Set these Vercel Project Settings environment variables for `/api/chat`:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`

The full Streamlit UI still runs locally or on Streamlit-compatible hosting with `python -m streamlit run streamlit_app.py`.

## Streamlit Cloud Secrets

- Use the placeholder file at `secrets.toml` as a reference only.
- In Streamlit Cloud, open your app settings and paste the same keys into the `Secrets` editor.
- Locally, if you prefer Streamlit-managed secrets over `.env`, copy those keys into `.streamlit/secrets.toml`.
