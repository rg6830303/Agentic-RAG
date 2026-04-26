from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI

from app import app as root_app
from api.index import (
    ChatRequest,
    EvaluationRequest,
    capabilities_info,
    chat_completion,
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
        self.assertEqual(health_check()["status"], "ok")

    def test_corpus_is_available_to_vercel_api(self) -> None:
        corpus = corpus_info()
        self.assertGreater(corpus["source_count"], 0)
        self.assertGreater(corpus["chunk_count"], 0)
        self.assertTrue(corpus["indexes"]["hybrid"])

    def test_capabilities_are_exposed(self) -> None:
        capabilities = capabilities_info()
        self.assertIn("hybrid", capabilities["retrieval"])
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
            response = chat_completion(ChatRequest(message="How is resistance related to length?"))

        self.assertEqual(response.provider, "local_agentic_rag")
        self.assertTrue(response.finalized_answer)
        self.assertTrue(response.citations)
        self.assertTrue(response.retrieved_chunks)
        self.assertTrue(response.pipeline)
        self.assertTrue(response.checkpoints)

    def test_evaluation_endpoint_returns_rows(self) -> None:
        report = evaluate(EvaluationRequest())
        self.assertGreater(report.summary["sample_count"], 0)
        self.assertTrue(report.rows)


if __name__ == "__main__":
    unittest.main()
