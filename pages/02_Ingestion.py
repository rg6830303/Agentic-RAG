from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ingestion.loaders import SUPPORTED_EXTENSIONS
from src.ui.helpers.bootstrap import get_runtime
from src.ui.state.session import initialize_session
from src.utils.models import ChunkingStrategy


initialize_session()
runtime = get_runtime()
upload_dir = runtime.settings.data_dir / "uploads"
upload_dir.mkdir(parents=True, exist_ok=True)

st.title("Ingestion")
st.caption("Upload or select local files, preview chunk plans, approve them, and commit them to the local docstore and indexes.")

col_left, col_right = st.columns([2, 1])
with col_left:
    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
    )
with col_right:
    st.info(f"Parallel workers: {runtime.ingestion_service.max_workers}")
    st.write(f"Sample corpus: `{runtime.settings.sample_corpus_dir}`")

sample_files = sorted(
    [
        path
        for path in runtime.settings.sample_corpus_dir.rglob("*")
        if path.is_file()
    ]
)
selected_samples = st.multiselect(
    "Select sample corpus files",
    options=[str(path) for path in sample_files],
)

strategy = st.selectbox(
    "Chunking method",
    options=[strategy.value for strategy in ChunkingStrategy],
    index=5,
)
auto_mode = st.checkbox("Auto-select chunking per file", value=True)
hitl_enabled = st.checkbox("Require chunk approval before commit", value=True)
checkpoints_enabled = st.checkbox("Enable checkpoints", value=True)
parallel_enabled = st.checkbox("Enable parallel ingestion", value=True)
rebuild_indexes = st.checkbox("Rebuild indexes after commit", value=True)

saved_paths: list[Path] = []
for upload in uploaded_files or []:
    destination = upload_dir / upload.name
    destination.write_bytes(upload.getbuffer())
    saved_paths.append(destination)
selected_paths = saved_paths + [Path(path) for path in selected_samples]

if st.button("Preview Ingestion", type="primary", disabled=not selected_paths):
    messages: list[str] = []
    prepared = runtime.ingestion_service.prepare(
        paths=selected_paths,
        strategy=strategy,
        auto_mode=auto_mode,
        checkpoints_enabled=checkpoints_enabled,
        parallel_enabled=parallel_enabled,
        progress_callback=messages.append,
    )
    st.session_state["prepared_ingestion"] = prepared
    for message in messages:
        st.info(message)

prepared = st.session_state.get("prepared_ingestion")
if prepared:
    st.subheader("Prepared Files")
    st.write(f"Documents: {len(prepared.documents)} | Chunks: {len(prepared.chunks)}")
    if prepared.skipped:
        for item in prepared.skipped:
            st.warning(item)

    strategy_rows = []
    for document in prepared.documents:
        selection = prepared.selections[document.document_id]
        strategy_rows.append(
            {
                "file_name": document.file_name,
                "strategy": selection.strategy,
                "reason": selection.reason,
                "heuristics": selection.heuristics,
            }
        )
    if strategy_rows:
        st.dataframe(pd.DataFrame(strategy_rows), use_container_width=True, hide_index=True)

    preview_rows = [
        {
            "file_name": chunk.file_name,
            "chunk_id": chunk.chunk_id,
            "method": chunk.chunking_method,
            "level": chunk.level,
            "page": chunk.page_number,
            "chars": chunk.char_count,
            "preview": chunk.text[:180],
        }
        for chunk in prepared.chunks[:50]
    ]
    if preview_rows:
        st.subheader("Chunk Preview")
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    if hitl_enabled:
        approve_cols = st.columns(2)
        if approve_cols[0].button("Approve Chunk Set and Commit", type="primary"):
            messages: list[str] = []
            result = runtime.ingestion_service.commit(
                prepared,
                rebuild_indexes=rebuild_indexes,
                progress_callback=messages.append,
            )
            st.session_state["last_ingestion_result"] = result
            st.session_state["prepared_ingestion"] = None
            for message in messages:
                st.info(message)
            st.success("Ingestion committed.")
        if approve_cols[1].button("Reject Prepared Chunk Set"):
            st.session_state["prepared_ingestion"] = None
            st.warning("Prepared ingestion discarded.")
    elif st.button("Commit Ingestion", type="primary"):
        messages = []
        result = runtime.ingestion_service.commit(
            prepared,
            rebuild_indexes=rebuild_indexes,
            progress_callback=messages.append,
        )
        st.session_state["last_ingestion_result"] = result
        st.session_state["prepared_ingestion"] = None
        for message in messages:
            st.info(message)
        st.success("Ingestion committed.")

last_result = st.session_state.get("last_ingestion_result")
if last_result:
    st.subheader("Last Commit Result")
    st.json(last_result)
