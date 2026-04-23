from __future__ import annotations

from pathlib import Path

from src.agentic.service import AgenticRAGService
from src.config.settings import AppSettings
from src.evaluation.heuristic import HeuristicEvaluator
from src.evaluation.ragas_runner import RagasLiteLLMEvaluator
from src.utils.models import QueryOptions


class EvaluationService:
    def __init__(self, settings: AppSettings, rag_service: AgenticRAGService) -> None:
        self.settings = settings
        self.heuristic = HeuristicEvaluator(settings, rag_service)
        self.ragas = RagasLiteLLMEvaluator(settings)

    def run(
        self,
        dataset_path: Path,
        options: QueryOptions,
        include_ragas: bool = False,
    ) -> dict[str, object]:
        heuristic_report = self.heuristic.run(dataset_path, options)
        ragas_report = None
        if include_ragas:
            ragas_report = self.ragas.run(dataset_path, heuristic_report.rows)
        return {"heuristic": heuristic_report, "ragas": ragas_report}
