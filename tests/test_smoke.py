from __future__ import annotations

import unittest
import uuid
from pathlib import Path
import shutil
import os
from unittest.mock import patch

from src.chunking.strategies import ChunkingService
from src.config.settings import AppSettings
from src.docstore.sqlite_store import SQLiteDocstore
from src.indexing.manager import IndexManager
from src.providers.azure_openai import AzureOpenAIProvider
from src.utils.hashing import checksum_text
from src.utils.models import DocumentSection, LoadedDocument


class DummyProvider(AzureOpenAIProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def chat_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 700) -> str:  # type: ignore[override]
        return "dummy response"


class SmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patcher = patch.dict(os.environ, {}, clear=True)
        self.env_patcher.start()
        temp_root = Path.cwd() / "artifacts" / "test_workspaces"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.root = temp_root / f"workspace_{uuid.uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=False)
        root = self.root
        (root / ".env").write_text(
            "\n".join(
                [
                    "AZURE_OPENAI_API_KEY=dummy",
                    "AZURE_OPENAI_ENDPOINT=https://example.openai.azure.com/",
                    "AZURE_OPENAI_API_VERSION=2024-12-01-preview",
                    "AZURE_OPENAI_CHAT_DEPLOYMENT=chat",
                    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=embed",
                ]
            ),
            encoding="utf-8",
        )
        self.settings = AppSettings.from_env(root)
        self.settings.ensure_directories()

    def tearDown(self) -> None:
        self.env_patcher.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_settings_detect_configuration(self) -> None:
        self.assertTrue(self.settings.chat_available)
        self.assertTrue(self.settings.embeddings_available)
        self.assertEqual(self.settings.azure_api_version, "2024-12-01-preview")
        self.assertEqual(self.settings.config_source, "dotenv")

    def test_environment_values_take_precedence_over_dotenv(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "environment-key",
                "AZURE_OPENAI_ENDPOINT": "https://environment.openai.azure.com/",
                "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
                "AZURE_OPENAI_CHAT_DEPLOYMENT": "environment-chat",
                "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT": "environment-embed",
            },
        ):
            settings = AppSettings.from_env(self.root)

        self.assertEqual(settings.config_source, "environment")
        self.assertEqual(settings.azure_api_key, "environment-key")
        self.assertEqual(
            settings.azure_endpoint,
            "https://environment.openai.azure.com/",
        )
        self.assertEqual(settings.azure_chat_deployment, "environment-chat")
        self.assertFalse(
            any("Missing .env file" in issue for issue in settings.validate())
        )

    def test_docstore_chunking_and_bm25(self) -> None:
        docstore = SQLiteDocstore(self.settings)
        chunking = ChunkingService()
        provider = DummyProvider(self.settings)
        index_manager = IndexManager(self.settings, provider, docstore)

        document = LoadedDocument(
            document_id=checksum_text("doc-1")[:24],
            file_path=str(self.root / "physics.txt"),
            file_name="physics.txt",
            extension=".txt",
            checksum=checksum_text("physics"),
            ingested_at="2026-01-01T00:00:00+00:00",
            sections=[
                DocumentSection(
                    section_id=checksum_text("sec-1")[:24],
                    text="Resistance depends on resistivity, length, and area. Current is rate of flow of charge.",
                )
            ],
            metadata={},
        )

        chunks, _ = chunking.chunk_document(document, requested_strategy="fixed", auto_mode=False)
        self.assertGreaterEqual(len(chunks), 1)
        docstore.upsert_document(document, chunks)
        bm25_result = index_manager.rebuild_bm25()
        self.assertTrue(bm25_result["ok"])
        results = index_manager.search_bm25("How is resistance related to length and area?", top_k=3)
        self.assertTrue(results)


if __name__ == "__main__":
    unittest.main()
