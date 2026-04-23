from __future__ import annotations

import streamlit as st

from src.ui.helpers.bootstrap import get_runtime
from src.ui.state.session import initialize_session


initialize_session()
runtime = get_runtime()

st.title("Settings")
st.caption("Inspect environment validation, local diagnostics, and tune guardrail thresholds for the current runtime.")

st.subheader("Environment Validation")
issues = runtime.settings.validate()
if issues:
    for issue in issues:
        if "missing" in issue.lower():
            st.warning(issue)
        else:
            st.info(issue)
else:
    st.success("Azure OpenAI configuration appears complete for chat and embeddings.")

st.subheader("Guardrails")
runtime.settings.confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.95,
    value=float(runtime.settings.confidence_threshold),
    step=0.05,
)
runtime.settings.citation_coverage_threshold = st.slider(
    "Citation Coverage Threshold",
    min_value=0.1,
    max_value=1.0,
    value=float(runtime.settings.citation_coverage_threshold),
    step=0.05,
)
runtime.settings.retrieval_score_floor = st.slider(
    "Retrieval Score Floor",
    min_value=0.05,
    max_value=0.95,
    value=float(runtime.settings.retrieval_score_floor),
    step=0.05,
)

st.subheader("Diagnostics")
st.json(runtime.settings.diagnostics())
