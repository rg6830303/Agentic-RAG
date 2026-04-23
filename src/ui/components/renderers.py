from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.utils.models import AnswerBundle, CheckpointRecord, RetrievalHit
from src.utils.text import highlight_text, tokenize


def render_checkpoints(checkpoints: list[CheckpointRecord]) -> None:
    if not checkpoints:
        return
    frame = pd.DataFrame(
        [
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "stage": checkpoint.stage,
                "status": checkpoint.status,
                "created_at": checkpoint.created_at,
                "notes": checkpoint.notes or "",
            }
            for checkpoint in checkpoints
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def open_source_viewer(hit: RetrievalHit, query: str) -> None:
    st.session_state["source_viewer_request"] = {
        "file_name": hit.file_name,
        "chunk_ids": [hit.chunk_id],
        "query_terms": tokenize(query),
        "focus_chunk_id": hit.chunk_id,
    }
    st.switch_page("pages/06_Source_Viewer.py")


def render_hit(hit: RetrievalHit, query: str, prefix: str) -> None:
    with st.container(border=True):
        cols = st.columns([4, 1])
        with cols[0]:
            if st.button(hit.file_name, key=f"open-source-{prefix}-{hit.chunk_id}"):
                open_source_viewer(hit, query)
            st.caption(
                f"source={hit.source} | score={hit.score:.3f} | method={hit.chunking_method} | page={hit.page_number or '-'} | level={hit.level}"
            )
        with cols[1]:
            st.metric("Rank", hit.rank)
        st.write(hit.text[:900] + ("..." if len(hit.text) > 900 else ""))
        if hit.sentence_attention:
            st.caption("Sentence attention")
            for item in hit.sentence_attention[:3]:
                st.write(f"- {item['score']}: {item['sentence']}")


def render_hits(hits: list[RetrievalHit], query: str, prefix: str = "hit") -> None:
    for hit in hits:
        render_hit(hit, query, prefix)


def render_guardrails(bundle: AnswerBundle) -> None:
    status = "Pass" if bundle.guardrails.passed else "Needs Review"
    st.subheader("Guardrails")
    cols = st.columns(4)
    cols[0].metric("Status", status)
    cols[1].metric("Confidence", f"{bundle.guardrails.confidence:.2f}")
    cols[2].metric("Citation Coverage", f"{bundle.guardrails.citation_coverage:.2f}")
    cols[3].metric("Retrieval Floor", "Met" if bundle.guardrails.retrieval_floor_met else "Missed")
    if bundle.guardrails.risk_flags:
        for flag in bundle.guardrails.risk_flags:
            st.warning(flag)


def render_source_chunks(
    hits: list[RetrievalHit],
    focus_chunk_ids: set[str],
    query_terms: list[str],
) -> None:
    for hit in hits:
        with st.container(border=True):
            st.caption(
                f"{hit.file_name} | chunk_id={hit.chunk_id} | page={hit.page_number or '-'} | method={hit.chunking_method}"
            )
            body = highlight_text(hit.text, query_terms)
            if hit.chunk_id in focus_chunk_ids:
                st.markdown(
                    f"<div style='border:2px solid #ffb300;padding:0.75rem;border-radius:0.5rem'>{body}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='padding:0.75rem;border-radius:0.5rem;background:#fafafa'>{body}</div>",
                    unsafe_allow_html=True,
                )
            if hit.sentence_attention:
                st.caption("Top sentence matches")
                for item in hit.sentence_attention[:3]:
                    st.write(f"- {item['score']}: {item['sentence']}")
