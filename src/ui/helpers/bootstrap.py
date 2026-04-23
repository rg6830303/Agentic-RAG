from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from src.agentic.service import AgenticRAGService
from src.checkpoints.service import CheckpointManager
from src.chunking.strategies import ChunkingService
from src.config.settings import AppSettings
from src.docstore.sqlite_store import SQLiteDocstore
from src.evaluation.service import EvaluationService
from src.indexing.manager import IndexManager
from src.ingestion.pipeline import IngestionService
from src.providers.azure_openai import AzureOpenAIProvider
from src.retrieval.engine import RetrievalEngine
from src.reranking.heuristic import HeuristicReranker


@dataclass(slots=True)
class AppRuntime:
    settings: AppSettings
    provider: AzureOpenAIProvider
    docstore: SQLiteDocstore
    checkpoint_manager: CheckpointManager
    chunking_service: ChunkingService
    index_manager: IndexManager
    ingestion_service: IngestionService
    retrieval_engine: RetrievalEngine
    rag_service: AgenticRAGService
    evaluation_service: EvaluationService


@st.cache_resource(show_spinner=False)
def get_runtime(root_dir: str | None = None) -> AppRuntime:
    root_path = Path(root_dir or Path.cwd())
    settings = AppSettings.from_env(root_path)
    settings.ensure_directories()
    provider = AzureOpenAIProvider(settings)
    docstore = SQLiteDocstore(settings)
    checkpoint_manager = CheckpointManager(docstore)
    chunking_service = ChunkingService()
    index_manager = IndexManager(settings, provider, docstore)
    retrieval_engine = RetrievalEngine(docstore, index_manager, reranker=HeuristicReranker())
    ingestion_service = IngestionService(
        docstore=docstore,
        chunking_service=chunking_service,
        index_manager=index_manager,
        checkpoint_manager=checkpoint_manager,
        max_workers=settings.max_workers,
    )
    rag_service = AgenticRAGService(settings, provider, retrieval_engine, checkpoint_manager)
    evaluation_service = EvaluationService(settings, rag_service)
    return AppRuntime(
        settings=settings,
        provider=provider,
        docstore=docstore,
        checkpoint_manager=checkpoint_manager,
        chunking_service=chunking_service,
        index_manager=index_manager,
        ingestion_service=ingestion_service,
        retrieval_engine=retrieval_engine,
        rag_service=rag_service,
        evaluation_service=evaluation_service,
    )
