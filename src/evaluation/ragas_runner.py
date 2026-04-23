from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.config.settings import AppSettings


class RagasLiteLLMEvaluator:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def available(self) -> bool:
        package_flags = self.settings.package_flags()
        return (
            package_flags.get("ragas", False)
            and package_flags.get("litellm", False)
            and self.settings.chat_available
            and self.settings.embeddings_available
        )

    def run(self, dataset_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.available():
            return {
                "ok": False,
                "message": "RAGAS + LiteLLM optional stack is unavailable, so only local heuristic evaluation was executed.",
            }
        try:
            from datasets import Dataset
            from langchain_openai import AzureChatOpenAI
            from ragas import evaluate
            from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
            from ragas.metrics._answer_relevance import ResponseRelevancy
            from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference
            from ragas.metrics._context_recall import LLMContextRecall
            from ragas.metrics._faithfulness import Faithfulness
        except ImportError as exc:
            return {
                "ok": False,
                "message": f"RAGAS or LiteLLM import failed after availability check: {exc}",
            }

        class _LiteLLMEmbeddingsAdapter(LiteLLMEmbeddings):
            def embed_query(self, text: str) -> list[float]:
                return self.embed_text(text)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.embed_texts(texts)

        if not rows:
            return {
                "ok": False,
                "message": "No evaluation rows were available for RAGAS scoring.",
                "dataset_path": str(dataset_path),
                "row_count": 0,
            }

        llm = AzureChatOpenAI(
            azure_endpoint=self.settings.azure_endpoint,
            api_key=self.settings.azure_api_key,
            api_version=self.settings.azure_api_version,
            azure_deployment=self.settings.azure_chat_deployment,
            temperature=0.0,
        )
        embeddings = _LiteLLMEmbeddingsAdapter(
            model=f"azure/{self.settings.azure_embeddings_deployment}",
            api_key=self.settings.azure_api_key,
            api_base=self.settings.azure_endpoint,
            api_version=self.settings.azure_api_version,
        )
        dataset = Dataset.from_list(
            [
                {
                    "user_input": row["question"],
                    "response": row["response"],
                    "retrieved_contexts": row["retrieved_contexts"],
                    "reference": row["reference"],
                }
                for row in rows
            ]
        )
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[
                    Faithfulness(),
                    ResponseRelevancy(strictness=1),
                    LLMContextPrecisionWithoutReference(),
                    LLMContextRecall(),
                ],
                llm=llm,
                embeddings=embeddings,
                show_progress=False,
                raise_exceptions=False,
            )
            frame = result.to_pandas()
        except Exception as exc:
            return {
                "ok": False,
                "message": f"RAGAS evaluation failed: {exc}",
                "dataset_path": str(dataset_path),
                "row_count": len(rows),
            }

        records = frame.to_dict(orient="records")
        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "llm_context_precision_without_reference",
            "context_recall",
        ]
        summary: dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "row_count": len(records),
        }
        for metric_name in metric_names:
            values = [
                float(record[metric_name])
                for record in records
                if record.get(metric_name) is not None
                and not math.isnan(float(record[metric_name]))
            ]
            summary[f"avg_{metric_name}"] = round(sum(values) / max(len(values), 1), 4) if values else None

        return {
            "ok": True,
            "message": "RAGAS evaluation completed.",
            "summary": summary,
            "rows": records,
        }
