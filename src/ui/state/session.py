from __future__ import annotations

import streamlit as st


def initialize_session() -> None:
    defaults = {
        "prepared_ingestion": None,
        "last_ingestion_result": None,
        "source_viewer_request": {},
        "pending_context_review": None,
        "pending_answer_bundle": None,
        "pending_admin_action": None,
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
