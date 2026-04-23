from __future__ import annotations

import streamlit as st

from src.ui.components.renderers import render_checkpoints, render_guardrails, render_hits
from src.ui.helpers.bootstrap import get_runtime
from src.ui.state.session import initialize_session
from src.utils.models import QueryOptions, RetrievalHit, RetrievalMode


initialize_session()
runtime = get_runtime()
stats = runtime.docstore.stats()

st.title("Chat")
st.caption("Ask grounded questions over the local corpus, combine FAISS and BM25 retrieval, review evidence, and apply Self-RAG before finalizing answers.")

if stats["chunk_count"] == 0:
    st.warning("No chunks are currently indexed. Ingest files first.")

question = st.text_area("Question", height=120, placeholder="Ask a question about the ingested corpus...")

control_cols = st.columns(4)
use_reranking = control_cols[0].checkbox("Reranking", value=True)
use_bm25 = control_cols[1].checkbox("BM25", value=True)
self_rag = control_cols[2].checkbox("Self-RAG", value=True)
sentence_attention = control_cols[3].checkbox("Sentence Attention", value=True)

control_cols_2 = st.columns(4)
checkpoints_enabled = control_cols_2[0].checkbox("Checkpoints", value=True)
parallel_enabled = control_cols_2[1].checkbox("Parallel Branches", value=True)
evaluation_enabled = control_cols_2[2].checkbox("Evaluation Summary", value=True)
citation_display = control_cols_2[3].checkbox("Show Citations", value=True)

control_cols_3 = st.columns(4)
require_context_review = control_cols_3[0].checkbox("Manual Context Review", value=False)
require_final_approval = control_cols_3[1].checkbox("Manual Final Approval", value=False)
use_vector = control_cols_3[2].checkbox("FAISS", value=True)
top_k = control_cols_3[3].slider("Top K", min_value=2, max_value=10, value=6)

control_cols_4 = st.columns(3)
retrieval_mode = control_cols_4[0].selectbox(
    "Retrieval Mode",
    options=[RetrievalMode.FIXED.value, RetrievalMode.HIERARCHICAL.value],
)
chunking_visibility = control_cols_4[1].selectbox(
    "Chunking Strategy Lens",
    options=["auto/current", "fixed", "semantic", "recursive", "adaptive", "hierarchical"],
)
st.caption(f"Parallel execution state: {'enabled' if parallel_enabled else 'sequential'}")

options = QueryOptions(
    top_k=top_k,
    use_vector=use_vector,
    use_bm25=use_bm25,
    use_reranking=use_reranking,
    self_rag=self_rag,
    checkpoints_enabled=checkpoints_enabled,
    require_context_review=require_context_review,
    require_final_approval=require_final_approval,
    evaluation_enabled=evaluation_enabled,
    sentence_attention=sentence_attention,
    citation_display=citation_display,
    retrieval_mode=retrieval_mode,
    parallel_enabled=parallel_enabled,
)

if st.button("Retrieve Contexts", disabled=not question.strip()):
    hits, checkpoints = runtime.rag_service.retrieve_contexts(
        question,
        options,
        require_human_review=require_context_review,
    )
    if chunking_visibility != "auto/current":
        hits = [hit for hit in hits if hit.chunking_method == chunking_visibility]
    st.session_state["pending_context_review"] = {
        "question": question,
        "options": options,
        "hits": hits,
        "checkpoints": checkpoints,
    }

pending_context = st.session_state.get("pending_context_review")
if pending_context and pending_context["question"] == question:
    st.subheader("Retrieved Contexts")
    render_checkpoints(pending_context["checkpoints"])
    render_hits(pending_context["hits"], question, prefix="context-review")
    if require_context_review:
        review_cols = st.columns(2)
        if review_cols[0].button("Approve Contexts and Generate Answer", type="primary"):
            bundle = runtime.rag_service.generate_from_hits(
                question,
                pending_context["hits"],
                options,
                checkpoints=pending_context["checkpoints"],
            )
            st.session_state["pending_answer_bundle"] = bundle
        if review_cols[1].button("Reject Retrieved Contexts"):
            checkpoint_id = pending_context["checkpoints"][0].checkpoint_id
            runtime.checkpoint_manager.reject(checkpoint_id, notes="Rejected from Chat page.")
            st.session_state["pending_context_review"] = None
            st.info("Retrieved contexts rejected.")

if st.button("Ask", type="primary", disabled=not question.strip()):
    bundle = runtime.rag_service.answer(question, options)
    if chunking_visibility != "auto/current":
        bundle.citations = [citation for citation in bundle.citations if citation.chunking_method == chunking_visibility]
    st.session_state["pending_answer_bundle"] = bundle

bundle = st.session_state.get("pending_answer_bundle")
if bundle and bundle.question == question:
    st.subheader("Answer")
    if bundle.needs_review:
        st.warning("This answer is marked as needs review.")
    st.write(bundle.answer)

    if bundle.reflection:
        st.caption(f"Reflection: {bundle.reflection}")
    if citation_display and bundle.citations:
        st.subheader("Citations")
        for citation in bundle.citations:
            cols = st.columns([3, 2])
            if cols[0].button(citation.file_name, key=f"citation-{citation.chunk_id}"):
                st.session_state["source_viewer_request"] = {
                    "file_name": citation.file_name,
                    "chunk_ids": [citation.chunk_id],
                    "query_terms": question.split(),
                }
                st.switch_page("pages/06_Source_Viewer.py")
            cols[1].caption(
                f"score={citation.score:.3f} | method={citation.chunking_method} | source={citation.source}"
            )
    render_guardrails(bundle)
    render_checkpoints(bundle.checkpoints)

    st.subheader("Retrieved Chunks")
    hits_for_view = [
        RetrievalHit(**item) for item in bundle.metadata.get("top_hits", [])
    ] or bundle.citations
    if chunking_visibility != "auto/current":
        hits_for_view = [
            hit for hit in hits_for_view if hit.chunking_method == chunking_visibility
        ]
    render_hits(hits_for_view, question, prefix="answer")

    if evaluation_enabled:
        st.subheader("Evaluation Summary")
        st.json(bundle.evaluation_summary)

    if require_final_approval:
        approve_cols = st.columns(2)
        final_checkpoint = bundle.checkpoints[-1] if bundle.checkpoints else None
        if approve_cols[0].button("Approve Final Answer", type="primary"):
            if final_checkpoint:
                runtime.checkpoint_manager.approve(
                    final_checkpoint.checkpoint_id,
                    notes="Approved final answer in Chat page.",
                )
            st.success("Final answer approved.")
        if approve_cols[1].button("Reject Final Answer"):
            if final_checkpoint:
                runtime.checkpoint_manager.reject(
                    final_checkpoint.checkpoint_id,
                    notes="Rejected final answer in Chat page.",
                )
            st.warning("Final answer rejected.")
