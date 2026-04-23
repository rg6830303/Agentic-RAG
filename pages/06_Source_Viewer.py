from __future__ import annotations

import streamlit as st

from src.ui.components.renderers import render_source_chunks
from src.ui.helpers.bootstrap import get_runtime
from src.ui.state.session import initialize_session
from src.utils.models import RetrievalHit


initialize_session()
runtime = get_runtime()
request = st.session_state.get("source_viewer_request", {}) or {}
documents = runtime.docstore.list_documents()

st.title("Source Viewer")
st.caption("Inspect the local source chunks that supported an answer, with chunk highlighting and sentence-level relevance.")

available_names = [document["file_name"] for document in documents]
default_name = request.get("file_name") if request.get("file_name") in available_names else (available_names[0] if available_names else None)
selected_file = st.selectbox("File", options=available_names, index=available_names.index(default_name) if default_name else 0) if available_names else None
query_terms = request.get("query_terms", [])
focus_chunk_ids = set(request.get("chunk_ids", []))

if selected_file:
    chunks = runtime.docstore.list_chunks(file_name=selected_file)
    hit_views = [
        RetrievalHit(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            file_name=chunk.file_name,
            file_path=chunk.file_path,
            text=chunk.text,
            score=1.0 if chunk.chunk_id in focus_chunk_ids else 0.5,
            source="docstore",
            rank=index + 1,
            chunking_method=chunk.chunking_method,
            page_number=chunk.page_number,
            parent_chunk_id=chunk.parent_chunk_id,
            level=chunk.level,
            sentence_attention=[],
            metadata=chunk.metadata,
        )
        for index, chunk in enumerate(chunks)
    ]
    st.write(f"Loaded {len(hit_views)} chunks from `{selected_file}`.")
    render_source_chunks(hit_views, focus_chunk_ids, query_terms)
else:
    st.info("No files are available in the docstore yet.")
