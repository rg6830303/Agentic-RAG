from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.helpers.bootstrap import get_runtime
from src.ui.state.session import initialize_session


initialize_session()
runtime = get_runtime()
documents = runtime.docstore.list_documents()
stats = runtime.docstore.stats()

st.title("Docstore Management")
st.caption("Inspect the SQLite-backed document and chunk store, browse metadata, and confirm destructive actions before they execute.")

st.json(stats)
if documents:
    st.dataframe(pd.DataFrame(documents), use_container_width=True, hide_index=True)

selected_files = st.multiselect(
    "Select files to remove",
    options=[document["file_path"] for document in documents],
    format_func=lambda path: path.split("\\")[-1].split("/")[-1],
)

remove_cols = st.columns(2)
if remove_cols[0].button("Request Removal of Selected Files", disabled=not selected_files):
    checkpoint = runtime.checkpoint_manager.create(
        stage="pre-persist-destructive-admin-action",
        payload={"action": "remove_files", "file_paths": selected_files},
        enabled=True,
        requires_human=True,
    )
    st.session_state["pending_admin_action"] = {
        "kind": "remove_files",
        "file_paths": selected_files,
        "checkpoint_id": checkpoint.checkpoint_id,
    }

if remove_cols[1].button("Request Remove All Files", disabled=not documents):
    checkpoint = runtime.checkpoint_manager.create(
        stage="pre-persist-destructive-admin-action",
        payload={"action": "clear_all"},
        enabled=True,
        requires_human=True,
    )
    st.session_state["pending_admin_action"] = {
        "kind": "clear_all",
        "checkpoint_id": checkpoint.checkpoint_id,
    }

pending = st.session_state.get("pending_admin_action")
if pending and pending["kind"] in {"remove_files", "clear_all"}:
    st.warning(f"Pending destructive action: {pending['kind']}")
    approve_cols = st.columns(2)
    if approve_cols[0].button("Approve Destructive Action", type="primary"):
        runtime.checkpoint_manager.approve(
            pending["checkpoint_id"], notes="Approved in Docstore Management page."
        )
        if pending["kind"] == "remove_files":
            runtime.docstore.remove_files(pending["file_paths"])
        else:
            runtime.docstore.clear_all()
            runtime.index_manager.faiss_store.delete()
            runtime.index_manager.bm25_store.delete()
        st.session_state["pending_admin_action"] = None
        st.success("Docstore action completed.")
        st.rerun()
    if approve_cols[1].button("Reject Destructive Action"):
        runtime.checkpoint_manager.reject(
            pending["checkpoint_id"], notes="Rejected in Docstore Management page."
        )
        st.session_state["pending_admin_action"] = None
        st.info("Destructive action rejected.")

st.subheader("Chunk Preview")
chunk_rows = runtime.docstore.list_chunks(limit=50)
if chunk_rows:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "file_name": chunk.file_name,
                    "chunk_id": chunk.chunk_id,
                    "method": chunk.chunking_method,
                    "level": chunk.level,
                    "page": chunk.page_number,
                    "chars": chunk.char_count,
                }
                for chunk in chunk_rows
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
