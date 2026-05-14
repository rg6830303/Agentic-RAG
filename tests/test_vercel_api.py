from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import app as root_app
from api.index import (
    ChatRequest,
    ChatMessageEditRequest,
    ChatRegenerateRequest,
    EvaluationRequest,
    append_chat_message,
    capabilities_info,
    chat_completion,
    chat_history,
    chat_session,
    clone_chat_session,
    corpus_info,
    evaluate,
    edit_chat_message,
    health_check,
    read_root,
    regenerate_chat_response,
    runtime_info,
)


class VercelApiTests(unittest.TestCase):
    def test_health_and_root_metadata(self) -> None:
        self.assertIsInstance(root_app, FastAPI)
        root = read_root()
        self.assertIn(b"Advanced Agentic RAG", root.body)
        self.assertIn(b"requestSubmit", root.body)
        self.assertIn(b"data-pending-response", root.body)
        self.assertIn(b"Knowledge Base", root.body)
        self.assertIn(b"Evaluation &amp; Diagnostics", root.body)
        self.assertIn(b"localStorage", root.body)
        self.assertIn(b"drawerToggle", root.body)
        self.assertIn(b"composerOptions", root.body)
        self.assertIn(b"responsive-table", root.body)
        self.assertIn(b"Regenerate", root.body)
        self.assertIn(b"Save & regenerate", root.body)
        self.assertIn(b"buildThreadMemory", root.body)
        self.assertIn(b"ensureActiveThread", root.body)
        self.assertIn(b"/api/auth/login", root.body)
        self.assertIn(b"Response Details", root.body)
        self.assertIn(b"data-action=\"details\"", root.body)
        self.assertIn(b"authPasswordConfirm", root.body)
        self.assertIn(b"validateAuthForm", root.body)
        self.assertIn(b"authModeSwitch", root.body)
        self.assertIn(b"Create an account", root.body)
        self.assertEqual(health_check()["status"], "ok")

    def test_corpus_is_available_to_vercel_api(self) -> None:
        corpus = corpus_info()
        self.assertGreater(corpus["source_count"], 0)
        self.assertGreater(corpus["chunk_count"], 0)
        self.assertTrue(corpus["indexes"]["hybrid"])

    def test_capabilities_are_exposed(self) -> None:
        capabilities = capabilities_info()
        self.assertIn("hybrid", capabilities["retrieval"])
        self.assertIn("wikipedia_text", capabilities["retrieval"])
        self.assertIn("post-generation_pre-final-answer", capabilities["hitl"])

    def test_runtime_does_not_expose_secret_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "super-secret",
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
                "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
                "AZURE_OPENAI_CHAT_DEPLOYMENT": "chat",
            },
            clear=True,
        ):
            runtime = runtime_info()

        self.assertTrue(runtime["azure_openai"]["chat_configured"])
        self.assertNotIn("super-secret", str(runtime))

    def test_production_missing_env_uses_temporary_auth_store(self) -> None:
        original_store = app_module.account_store
        original_error = app_module.ACCOUNT_STORE_ERROR
        original_ephemeral_path = app_module.EPHEMERAL_APP_DB_PATH
        app_module.account_store = None
        app_module.ACCOUNT_STORE_ERROR = ""
        with tempfile.TemporaryDirectory() as tmpdir:
            app_module.EPHEMERAL_APP_DB_PATH = Path(tmpdir) / "auth.db"
            try:
                with patch.dict(os.environ, {"VERCEL": "1"}, clear=True):
                    client = TestClient(root_app, base_url="https://testserver")
                    self.assertEqual(client.get("/").status_code, 200)
                    self.assertEqual(client.get("/api/health").status_code, 200)

                    runtime = client.get("/api/runtime")
                    self.assertEqual(runtime.status_code, 200)
                    self.assertFalse(runtime.json()["persistence"]["database_url_configured"])
                    self.assertTrue(runtime.json()["persistence"]["ephemeral"])
                    self.assertIn("temporary", runtime.json()["persistence"]["warning"])

                    signup = client.post(
                        "/api/auth/signup",
                        json={
                            "email": "temp-auth@example.com",
                            "password": "correct horse battery staple",
                            "display_name": "Temp Auth",
                        },
                    )
                    self.assertEqual(signup.status_code, 200)
                    self.assertTrue(signup.json()["authenticated"])

                    me = client.get("/api/auth/me")
                    self.assertEqual(me.status_code, 200)
                    self.assertTrue(me.json()["authenticated"])

                    created = client.post("/api/chats", json={"session_name": "Temporary thread"})
                    self.assertEqual(created.status_code, 200)
            finally:
                app_module.account_store = original_store
                app_module.ACCOUNT_STORE_ERROR = original_error
                app_module.EPHEMERAL_APP_DB_PATH = original_ephemeral_path

    def test_auth_buttons_report_backend_setup_errors(self) -> None:
        original_store = app_module.account_store
        original_error = app_module.ACCOUNT_STORE_ERROR
        app_module.account_store = None
        app_module.ACCOUNT_STORE_ERROR = "Database unavailable"
        try:
            with patch.dict(os.environ, {"VERCEL": "1"}, clear=True), patch.object(
                app_module,
                "AccountStore",
                side_effect=RuntimeError("boom"),
            ):
                client = TestClient(root_app)
                protected = client.get("/api/chats")
                self.assertEqual(protected.status_code, 503)
                self.assertIn("Account persistence is unavailable", protected.json()["detail"])
        finally:
            app_module.account_store = original_store
            app_module.ACCOUNT_STORE_ERROR = original_error

    def test_chat_uses_local_rag_without_azure_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            response = chat_completion(ChatRequest(message="How is resistance related to length?", use_wikipedia=False))

        self.assertEqual(response.provider, "local_agentic_rag")
        self.assertTrue(response.finalized_answer)
        self.assertTrue(response.citations)
        self.assertTrue(response.retrieved_chunks)
        self.assertTrue(response.pipeline)
        self.assertTrue(response.checkpoints)

    def test_chat_history_persists_resume_and_clone(self) -> None:
        original_service = app_module.chat_history_service
        with tempfile.TemporaryDirectory() as tmpdir:
            app_module.chat_history_service = app_module.ChatHistoryService(Path(tmpdir))
            try:
                with patch.dict(os.environ, {}, clear=True):
                    first = chat_completion(ChatRequest(message="Help me study resistance and current.", use_wikipedia=False))
                    second = chat_completion(
                        ChatRequest(
                            message="What should I review next?",
                            session_id=first.session_id,
                            use_wikipedia=False,
                        )
                    )

                self.assertTrue(first.chat_saved)
                self.assertTrue(first.session_id)
                self.assertTrue(first.agenda_summary)
                self.assertTrue(first.suggestions)
                self.assertEqual(second.session_id, first.session_id)
                self.assertEqual(len(second.history), 2)

                listing = chat_history()
                self.assertEqual(len(listing.sessions), 1)
                loaded = chat_session(first.session_id or "")
                self.assertEqual(loaded.exchange_count, 2)
                self.assertEqual(loaded.history[0].user_prompt, "Help me study resistance and current.")

                cloned = clone_chat_session(first.session_id or "")
                self.assertNotEqual(cloned.session_id, first.session_id)
                self.assertEqual(len(cloned.history), 2)
            finally:
                app_module.chat_history_service = original_service

    def test_threads_continue_edit_and_regenerate(self) -> None:
        original_service = app_module.chat_history_service
        with tempfile.TemporaryDirectory() as tmpdir:
            app_module.chat_history_service = app_module.ChatHistoryService(Path(tmpdir))
            try:
                with patch.dict(os.environ, {}, clear=True):
                    first = chat_completion(
                        ChatRequest(
                            message="Explain RAG simply.",
                            use_wikipedia=False,
                        )
                    )
                    second = append_chat_message(
                        first.session_id or "",
                        ChatRequest(
                            message="Now explain how my project uses this.",
                            use_wikipedia=False,
                        ),
                    )

                    self.assertEqual(second.session_id, first.session_id)
                    self.assertEqual(len(second.history), 2)
                    self.assertEqual(len(second.messages), 4)
                    self.assertEqual(second.history[0].user_prompt, "Explain RAG simply.")
                    self.assertEqual(second.history[1].user_prompt, "Now explain how my project uses this.")

                    edited = edit_chat_message(
                        second.session_id or "",
                        second.history[0].user_message_id,
                        ChatMessageEditRequest(
                            message="Explain vector retrieval simply.",
                            use_wikipedia=False,
                        ),
                    )
                    self.assertEqual(len(edited.history), 1)
                    self.assertEqual(len(edited.messages), 2)
                    self.assertTrue(edited.history[0].edited)
                    self.assertEqual(edited.history[0].user_prompt, "Explain vector retrieval simply.")

                    assistant_before = edited.history[0].assistant_message_id
                    regenerated = regenerate_chat_response(
                        edited.session_id,
                        ChatRegenerateRequest(use_wikipedia=False),
                    )
                    self.assertEqual(len(regenerated.history), 1)
                    self.assertEqual(regenerated.history[0].user_prompt, "Explain vector retrieval simply.")
                    self.assertNotEqual(regenerated.history[0].assistant_message_id, assistant_before)
            finally:
                app_module.chat_history_service = original_service

    def test_client_owned_session_id_is_preserved_for_multi_turn_threads(self) -> None:
        original_service = app_module.chat_history_service
        with tempfile.TemporaryDirectory() as tmpdir:
            app_module.chat_history_service = app_module.ChatHistoryService(Path(tmpdir))
            try:
                requested_session_id = "chat_client_owned_thread"
                with patch.dict(os.environ, {}, clear=True):
                    first = chat_completion(
                        ChatRequest(
                            message="Explain RAG simply.",
                            session_id=requested_session_id,
                            session_name="Client-owned thread",
                            use_wikipedia=False,
                        )
                    )
                    second = append_chat_message(
                        requested_session_id,
                        ChatRequest(
                            message="Now explain how this applies to my app.",
                            session_id=requested_session_id,
                            session_name="Client-owned thread",
                            memory_context=(
                                "Recent turns:\n"
                                "User: Explain RAG simply.\n"
                                "Assistant: RAG combines retrieval and generation."
                            ),
                            use_wikipedia=False,
                        ),
                    )

                self.assertEqual(first.session_id, requested_session_id)
                self.assertEqual(second.session_id, requested_session_id)
                self.assertEqual(len(second.history), 2)
                self.assertEqual(second.history[0].user_prompt, "Explain RAG simply.")
                self.assertEqual(second.history[1].user_prompt, "Now explain how this applies to my app.")
            finally:
                app_module.chat_history_service = original_service

    def test_authenticated_chats_are_user_scoped_and_multi_turn(self) -> None:
        original_store = app_module.account_store
        with tempfile.TemporaryDirectory() as tmpdir:
            app_module.account_store = app_module.AccountStore(f"sqlite:///{Path(tmpdir) / 'auth.db'}")
            try:
                alice = TestClient(root_app)
                bob = TestClient(root_app)

                unauthenticated = TestClient(root_app).get("/api/chats")
                self.assertEqual(unauthenticated.status_code, 401)
                bad_login = TestClient(root_app).post(
                    "/api/auth/login",
                    json={"email": "alice@example.com", "password": "wrong-password"},
                )
                self.assertEqual(bad_login.status_code, 401)
                self.assertIn("incorrect", bad_login.json()["detail"])

                signup = alice.post(
                    "/api/auth/signup",
                    json={"email": "alice@example.com", "password": "correct horse battery staple", "display_name": "Alice"},
                )
                self.assertEqual(signup.status_code, 200)
                self.assertTrue(signup.json()["authenticated"])
                stored_user = app_module.account_store.get_user_by_email("alice@example.com")
                self.assertIsNotNone(stored_user)
                self.assertNotIn("correct horse battery staple", str(stored_user))

                created = alice.post("/api/chats", json={"session_name": "Alice thread"})
                self.assertEqual(created.status_code, 200)
                thread_id = created.json()["session_id"]

                first = alice.post(
                    f"/api/chats/{thread_id}/messages",
                    json={"message": "Explain RAG simply.", "use_wikipedia": False},
                )
                self.assertEqual(first.status_code, 200)
                second = alice.post(
                    f"/api/chats/{thread_id}/messages",
                    json={
                        "message": "Now explain how this applies to my app.",
                        "memory_context": "User previously asked: Explain RAG simply.",
                        "use_wikipedia": False,
                    },
                )
                self.assertEqual(second.status_code, 200)
                self.assertEqual(len(second.json()["history"]), 2)

                bob_signup = bob.post(
                    "/api/auth/signup",
                    json={"email": "bob@example.com", "password": "correct horse battery staple"},
                )
                self.assertEqual(bob_signup.status_code, 200)
                self.assertEqual(bob.get("/api/chats").json()["sessions"], [])
                self.assertEqual(bob.get(f"/api/chats/{thread_id}").status_code, 404)

                alice_threads = alice.get("/api/chats").json()["sessions"]
                self.assertEqual(len(alice_threads), 1)
                self.assertEqual(alice.get(f"/api/chats/{thread_id}").json()["exchange_count"], 2)
            finally:
                app_module.account_store = original_store

    def test_wikipedia_context_returns_url_backed_citations(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self.status_code = 200
                self._payload = payload

            def json(self) -> dict[str, object]:
                return self._payload

        def fake_get(_url: str, params: dict[str, object], **_kwargs: object) -> FakeResponse:
            if params.get("list") == "search":
                return FakeResponse(
                    {
                        "query": {
                            "search": [
                                {
                                    "title": "Ada Lovelace",
                                    "snippet": "English mathematician and writer",
                                    "size": 100,
                                }
                            ]
                        }
                    }
                )
            return FakeResponse(
                {
                    "query": {
                        "pages": [
                            {
                                "pageid": 974,
                                "title": "Ada Lovelace",
                                "extract": (
                                    "Ada Lovelace was an English mathematician and writer. "
                                    "She is chiefly known for her work on Charles Babbage's Analytical Engine."
                                ),
                                "fullurl": "https://en.wikipedia.org/wiki/Ada_Lovelace",
                            }
                        ]
                    }
                }
            )

        original_service = app_module.chat_history_service
        app_module.WIKIPEDIA_CACHE.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            app_module.chat_history_service = app_module.ChatHistoryService(Path(tmpdir))
            try:
                with patch.dict(os.environ, {}, clear=True), patch.object(app_module.requests, "get", side_effect=fake_get):
                    response = chat_completion(
                        ChatRequest(
                            message="Who was Ada Lovelace?",
                            use_generation=False,
                            top_k=3,
                        )
                    )

                wikipedia_citations = [
                    citation for citation in response.citations if citation.source_type == "wikipedia"
                ]
                self.assertTrue(wikipedia_citations)
                self.assertEqual(wikipedia_citations[0].source_url, "https://en.wikipedia.org/wiki/Ada_Lovelace")
                self.assertIn("wikipedia", response.used_methods)
            finally:
                app_module.chat_history_service = original_service
                app_module.WIKIPEDIA_CACHE.clear()

    def test_evaluation_endpoint_returns_rows(self) -> None:
        report = evaluate(EvaluationRequest())
        self.assertGreater(report.summary["sample_count"], 0)
        self.assertTrue(report.rows)


if __name__ == "__main__":
    unittest.main()
