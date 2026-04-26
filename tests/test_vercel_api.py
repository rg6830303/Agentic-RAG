from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from api.index import ChatRequest, chat_completion, health_check, read_root, runtime_info


class VercelApiTests(unittest.TestCase):
    def test_health_and_root_metadata(self) -> None:
        root = read_root()
        self.assertEqual(root["status"], "ok")
        self.assertEqual(root["health"], "/api/health")
        self.assertEqual(health_check()["status"], "ok")

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

    def test_chat_requires_vercel_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as raised:
                chat_completion(ChatRequest(message="hello"))

        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
