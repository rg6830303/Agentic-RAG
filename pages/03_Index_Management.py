from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.helpers.bootstrap import get_runtime
from src.ui.state.session import initialize_session


initialize_session()
runtime = get_runtime()
status = runtime.index_manager.status()
stats = runtime.docstore.stats()

st.title("Index Management")
st.caption("Inspect local FAISS and BM25 state, request rebuilds, and confirm index-changing actions through checkpoints.")

metrics = st.columns(4)
metrics[0].metric("FAISS Ready", "Yes" if status.faiss_ready else "No")
metrics[1].metric("BM25 Ready", "Yes" if status.bm25_ready else "No")
metrics[2].metric("FAISS Package", "Yes" if status.faiss_available else "No")
metrics[3].metric("Chunk Count", status.chunk_count)

st.write("Configured index paths:")
st.code(
    f"FAISS: {runtime.settings.faiss_index_path}\nBM25: {runtime.settings.bm25_path}",
    language="text",
)

if stats["files"]:
    st.dataframe(pd.DataFrame(stats["files"]), use_container_width=True, hide_index=True)

if st.button("Request Rebuild: All Indexes", type="primary"):
    checkpoint = runtime.checkpoint_manager.create(
        stage="pre-index-rebuild",
        payload={"action": "rebuild_all"},
        enabled=True,
        requires_human=True,
        notes="User requested full index rebuild.",
    )
    st.session_state["pending_admin_action"] = {"kind": "rebuild_all", "checkpoint_id": checkpoint.checkpoint_id}

if st.button("Request Rebuild: FAISS Only"):
    checkpoint = runtime.checkpoint_manager.create(
        stage="pre-index-rebuild",
        payload={"action": "rebuild_faiss"},
        enabled=True,
        requires_human=True,
        notes="User requested FAISS-only rebuild.",
    )
    st.session_state["pending_admin_action"] = {"kind": "rebuild_faiss", "checkpoint_id": checkpoint.checkpoint_id}

if st.button("Request Rebuild: BM25 Only"):
    checkpoint = runtime.checkpoint_manager.create(
        stage="pre-index-rebuild",
        payload={"action": "rebuild_bm25"},
        enabled=True,
        requires_human=True,
        notes="User requested BM25-only rebuild.",
    )
    st.session_state["pending_admin_action"] = {"kind": "rebuild_bm25", "checkpoint_id": checkpoint.checkpoint_id}

pending = st.session_state.get("pending_admin_action")
if pending:
    st.warning(f"Pending admin action: {pending['kind']}")
    approve_cols = st.columns(2)
    if approve_cols[0].button("Approve Pending Admin Action", type="primary"):
        runtime.checkpoint_manager.approve(pending["checkpoint_id"], notes="Approved in Index Management page.")
        if pending["kind"] == "rebuild_all":
            result = runtime.index_manager.rebuild_all()
        elif pending["kind"] == "rebuild_faiss":
            result = runtime.index_manager.rebuild_faiss()
        else:
            result = runtime.index_manager.rebuild_bm25()
        st.success("Admin action completed.")
        st.json(result)
        st.session_state["pending_admin_action"] = None
    if approve_cols[1].button("Reject Pending Admin Action"):
        runtime.checkpoint_manager.reject(pending["checkpoint_id"], notes="Rejected in Index Management page.")
        st.session_state["pending_admin_action"] = None
        st.info("Pending admin action rejected.")

st.subheader("Recent Checkpoints")
checkpoints = runtime.docstore.list_checkpoints(limit=20)
if checkpoints:
    st.dataframe(pd.DataFrame(checkpoints), use_container_width=True, hide_index=True)
