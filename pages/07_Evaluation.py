from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ui.helpers.bootstrap import get_runtime
from src.ui.state.session import initialize_session
from src.utils.models import QueryOptions, RetrievalMode


initialize_session()
runtime = get_runtime()
default_dataset = runtime.settings.golden_eval_dir / "ncert_physics_golden.json"

st.title("Evaluation")
st.caption("Run local-first heuristic evaluations now, and keep the optional RAGAS adapter visible but safely isolated.")

dataset_path = st.text_input("Golden dataset path", value=str(default_dataset))
include_ragas = st.checkbox("Attempt optional RAGAS + LiteLLM adapter", value=False)
top_k = st.slider("Top K", min_value=2, max_value=10, value=6)
retrieval_mode = st.selectbox(
    "Retrieval Mode",
    options=[RetrievalMode.FIXED.value, RetrievalMode.HIERARCHICAL.value],
)

options = QueryOptions(
    top_k=top_k,
    use_vector=True,
    use_bm25=True,
    use_reranking=True,
    self_rag=True,
    checkpoints_enabled=False,
    require_context_review=False,
    require_final_approval=False,
    evaluation_enabled=False,
    sentence_attention=True,
    citation_display=True,
    retrieval_mode=retrieval_mode,
    parallel_enabled=True,
)

if st.button("Run Evaluation", type="primary", disabled=not Path(dataset_path).exists()):
    result = runtime.evaluation_service.run(Path(dataset_path), options, include_ragas=include_ragas)
    heuristic = result["heuristic"]
    st.subheader("Heuristic Report")
    st.json(heuristic.summary)
    st.dataframe(heuristic.rows, use_container_width=True, hide_index=True)
    st.write(heuristic.artifacts)
    if result["ragas"]:
        st.subheader("RAGAS Adapter Status")
        st.json(result["ragas"])

reports = sorted(runtime.settings.evaluations_dir.glob("*"), reverse=True)
if reports:
    st.subheader("Saved Reports")
    for report_dir in reports[:10]:
        st.write(str(report_dir))
