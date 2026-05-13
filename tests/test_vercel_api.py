from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from fastapi import FastAPI

from app import app as root_app
from api.index import (
    ChatRequest,
    EvaluationRequest,
    capabilities_info,
    chat_completion,
    chat_history,
    chat_session,
    clone_chat_session,
    corpus_info,
    evaluate,
    health_check,
    read_root,
    runtime_info,
)


class VercelApiTests(unittest.TestCase):
    def test_health_and_root_metadata(self) -> None:
        self.assertIsInstance(root_app, FastAPI)
        root = read_root()
        self.assertIn(b"Advanced Agentic RAG", root.body)
        self.assertIn(b"requestSubmit", root.body)
        self.assertIn(b"data-pending-response", root.body)
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
