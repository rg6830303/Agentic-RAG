from __future__ import annotations

import streamlit as st

from src.ui.helpers.bootstrap import get_runtime
from src.ui.state.session import initialize_session


initialize_session()
runtime = get_runtime()
stats = runtime.docstore.stats()
diagnostics = runtime.settings.diagnostics()

st.title("Home")
st.write(
    "This local-first Agentic RAG system combines Azure OpenAI-backed embeddings and chat with on-disk FAISS, a standalone BM25 path, SQLite docstore persistence, multiple chunking strategies, Self-RAG reflection, checkpoints, HITL controls, and offline-friendly evaluation artifacts."
)

overview_cols = st.columns(4)
overview_cols[0].metric("Documents", stats["document_count"])
overview_cols[1].metric("Chunks", stats["chunk_count"])
overview_cols[2].metric("Parent Chunks", stats["parent_chunk_count"])
overview_cols[3].metric("Child Chunks", stats["child_chunk_count"])

st.subheader("Capabilities")
st.markdown(
    """
- Multi-file ingestion for PDF, DOCX, text, markdown, CSV, JSON, SQL, and common code files
- Fixed, semantic, recursive, adaptive, hierarchical, and auto-selected chunking
- Local SQLite docstore plus persisted FAISS and BM25 indexes
- Fixed and hierarchical retrieval modes with optional reranking and Self-RAG refinement
- Checkpoints for ingestion, retrieval, generation, finalization, and destructive admin actions
- Sentence-level relevance display, citation-aware answers, local evaluation reports, and optional RAGAS adapter hooks
"""
)

st.subheader("Diagnostics")
st.json(diagnostics)

st.subheader("Next Steps")
st.page_link("pages/02_Ingestion.py", label="Go to Ingestion")
st.page_link("pages/03_Index_Management.py", label="Go to Index Management")
st.page_link("pages/05_Chat.py", label="Go to Chat")
st.page_link("pages/07_Evaluation.py", label="Go to Evaluation")
