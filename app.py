from __future__ import annotations

import streamlit as st

from src.ui.helpers.bootstrap import get_runtime
from src.ui.state.session import initialize_session


st.set_page_config(
    page_title="Advanced Agentic RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session()
runtime = get_runtime()

st.title("Advanced Agentic RAG")
st.caption("Thin bootstrap entrypoint for the local-first Windows-friendly Streamlit app.")

issues = runtime.settings.validate()
if issues:
    for issue in issues:
        if "missing" in issue.lower():
            st.warning(issue)
        else:
            st.info(issue)

stats = runtime.docstore.stats()
cols = st.columns(4)
cols[0].metric("Documents", stats["document_count"])
cols[1].metric("Chunks", stats["chunk_count"])
cols[2].metric("FAISS", "Ready" if runtime.index_manager.status().faiss_ready else "Not Ready")
cols[3].metric("BM25", "Ready" if runtime.index_manager.status().bm25_ready else "Not Ready")

st.write(
    "Use the pages in the sidebar to ingest files, inspect indexes, chat over the local corpus, review sources, and run evaluations."
)
st.page_link("pages/01_Home.py", label="Open Home")
st.page_link("pages/02_Ingestion.py", label="Open Ingestion")
st.page_link("pages/05_Chat.py", label="Open Chat")
