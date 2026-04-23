# Advanced Agentic RAG

This repository contains a complete local-first Agentic RAG system built with Python and Streamlit for Windows CMD usage. The app keeps ingestion, chunking, SQLite docstore persistence, FAISS indexing, BM25 retrieval, orchestration, evaluation artifacts, and UI logic on local disk. Azure OpenAI is used only for chat and embeddings when the configured `.env` supports those capabilities.

## Highlights

- Thin `app.py` bootstrap with multipage Streamlit UI under `pages/`
- Local ingestion for PDF, DOCX, text, markdown, code, CSV, JSON, and SQL
- Fixed, semantic, recursive, adaptive, hierarchical, and auto-selected chunking
- SQLite docstore, persisted FAISS, and persisted standalone BM25
- Self-RAG reflection, checkpoints, HITL approval flows, and sentence-level relevance
- Local heuristic evaluation plus an optional RAGAS/LiteLLM adapter surface
- Original NCERT Class 12 Physics-inspired sample corpus and golden evaluation dataset

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Keep the existing `.env` in the repository root.
4. Run `python -m streamlit run app.py` from Windows CMD.

Detailed Windows steps are in `WINDOWS_RUN.md`.
