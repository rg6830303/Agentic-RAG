from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


@dataclass(slots=True)
class AppSettings:
    root_dir: Path
    env_path: Path
    data_dir: Path
    sample_corpus_dir: Path
    golden_eval_dir: Path
    artifacts_dir: Path
    docstore_dir: Path
    indexes_dir: Path
    evaluations_dir: Path
    sessions_dir: Path
    sqlite_path: Path
    faiss_index_path: Path
    faiss_meta_path: Path
    bm25_path: Path
    chunk_artifact_path: Path
    checkpoint_artifact_path: Path
    azure_endpoint: str
    azure_api_version: str
    azure_api_key: str
    azure_chat_deployment: str
    azure_embeddings_deployment: str
    embedding_batch_size: int = 8
    embedding_timeout_seconds: int = 60
    chat_timeout_seconds: int = 90
    max_workers: int = 4
    default_top_k: int = 6
    confidence_threshold: float = 0.55
    citation_coverage_threshold: float = 0.5
    retrieval_score_floor: float = 0.2

    @classmethod
    def from_env(cls, root_dir: Path | None = None) -> "AppSettings":
        root = root_dir or Path.cwd()
        env_path = root / ".env"
        values = {key: value or "" for key, value in dotenv_values(env_path).items()}
        artifacts_dir = root / "artifacts"
        docstore_dir = artifacts_dir / "docstore"
        indexes_dir = artifacts_dir / "indexes"
        evaluations_dir = artifacts_dir / "evaluations"
        sessions_dir = artifacts_dir / "sessions"
        return cls(
            root_dir=root,
            env_path=env_path,
            data_dir=root / "data",
            sample_corpus_dir=root / "data" / "sample_corpus",
            golden_eval_dir=root / "data" / "golden_eval",
            artifacts_dir=artifacts_dir,
            docstore_dir=docstore_dir,
            indexes_dir=indexes_dir,
            evaluations_dir=evaluations_dir,
            sessions_dir=sessions_dir,
            sqlite_path=docstore_dir / "docstore.sqlite3",
            faiss_index_path=indexes_dir / "faiss.index",
            faiss_meta_path=indexes_dir / "faiss_meta.json",
            bm25_path=indexes_dir / "bm25_index.json",
            chunk_artifact_path=docstore_dir / "chunks.jsonl",
            checkpoint_artifact_path=sessions_dir / "checkpoints.jsonl",
            azure_endpoint=values.get("AZURE_OPENAI_ENDPOINT", "").strip(),
            azure_api_version=values.get("AZURE_OPENAI_API_VERSION", "").strip(),
            azure_api_key=values.get("AZURE_OPENAI_API_KEY", "").strip(),
            azure_chat_deployment=values.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip(),
            azure_embeddings_deployment=values.get(
                "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", ""
            ).strip(),
        )

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.sample_corpus_dir,
            self.golden_eval_dir,
            self.artifacts_dir,
            self.docstore_dir,
            self.indexes_dir,
            self.evaluations_dir,
            self.sessions_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def chat_available(self) -> bool:
        return bool(
            self.azure_endpoint
            and self.azure_api_version
            and self.azure_api_key
            and self.azure_chat_deployment
        )

    @property
    def embeddings_available(self) -> bool:
        return bool(
            self.azure_endpoint
            and self.azure_api_version
            and self.azure_api_key
            and self.azure_embeddings_deployment
        )

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.env_path.exists():
            issues.append("Missing .env file in the repository root.")
        if not self.azure_endpoint:
            issues.append("AZURE_OPENAI_ENDPOINT is missing.")
        if not self.azure_api_version:
            issues.append("AZURE_OPENAI_API_VERSION is missing.")
        if not self.azure_api_key:
            issues.append("AZURE_OPENAI_API_KEY is missing.")
        if not self.azure_chat_deployment:
            issues.append(
                "AZURE_OPENAI_CHAT_DEPLOYMENT is missing, so generative answers will use local extractive fallback."
            )
        if not self.azure_embeddings_deployment:
            issues.append(
                "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is missing, so FAISS indexing will remain unavailable."
            )
        return issues

    def package_flags(self) -> dict[str, bool]:
        return {
            "faiss": importlib.util.find_spec("faiss") is not None,
            "pypdf": importlib.util.find_spec("pypdf") is not None,
            "python_docx": importlib.util.find_spec("docx") is not None,
            "litellm": importlib.util.find_spec("litellm") is not None,
            "ragas": importlib.util.find_spec("ragas") is not None,
        }

    def diagnostics(self) -> dict[str, Any]:
        host = ""
        if self.azure_endpoint:
            host = self.azure_endpoint.replace("https://", "").replace("http://", "").strip("/")
        return {
            "env_path": str(self.env_path),
            "endpoint_host": host,
            "api_version": self.azure_api_version,
            "chat_configured": self.chat_available,
            "embeddings_configured": self.embeddings_available,
            "sqlite_path": str(self.sqlite_path),
            "faiss_index_path": str(self.faiss_index_path),
            "bm25_path": str(self.bm25_path),
            "package_flags": self.package_flags(),
        }
