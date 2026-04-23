from __future__ import annotations

import csv
import json
from pathlib import Path

from src.agentic.service import AgenticRAGService
from src.config.settings import AppSettings
from src.evaluation.dataset import load_golden_samples
from src.utils.hashing import checksum_text
from src.utils.models import EvaluationReport, QueryOptions
from src.utils.text import tokenize
from src.utils.time import utc_now_iso


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    ref_pool = ref_tokens.copy()
    for token in pred_tokens:
        if token in ref_pool:
            common += 1
            ref_pool.remove(token)
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class HeuristicEvaluator:
    def __init__(self, settings: AppSettings, rag_service: AgenticRAGService) -> None:
        self.settings = settings
        self.rag_service = rag_service

    def run(self, dataset_path: Path, options: QueryOptions) -> EvaluationReport:
        samples = load_golden_samples(dataset_path)
        rows: list[dict[str, object]] = []
        for sample in samples:
            bundle = self.rag_service.answer(
                sample.question,
                QueryOptions(
                    top_k=options.top_k,
                    use_vector=options.use_vector,
                    use_bm25=options.use_bm25,
                    use_reranking=options.use_reranking,
                    self_rag=options.self_rag,
                    checkpoints_enabled=False,
                    require_context_review=False,
                    require_final_approval=False,
                    evaluation_enabled=False,
                    sentence_attention=options.sentence_attention,
                    citation_display=options.citation_display,
                    retrieval_mode=options.retrieval_mode,
                    parallel_enabled=options.parallel_enabled,
                ),
            )
            citation_files = {citation.file_name for citation in bundle.citations}
            expected_files = set(sample.expected_files)
            context_match = len(citation_files & expected_files) / max(len(expected_files), 1)
            f1 = token_f1(bundle.answer, sample.reference_answer)
            citation_hit_rate = len(citation_files & expected_files) / max(len(citation_files), 1)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "question": sample.question,
                    "response": bundle.answer,
                    "reference": sample.reference_answer,
                    "retrieved_contexts": [citation.text for citation in bundle.citations],
                    "citation_files": sorted(citation_files),
                    "answer_f1": round(f1, 4),
                    "context_recall": round(context_match, 4),
                    "citation_hit_rate": round(citation_hit_rate, 4),
                    "confidence": round(bundle.confidence, 4),
                    "needs_review": bundle.needs_review,
                }
            )
        summary = {
            "sample_count": len(rows),
            "avg_answer_f1": round(sum(float(row["answer_f1"]) for row in rows) / max(len(rows), 1), 4),
            "avg_context_recall": round(sum(float(row["context_recall"]) for row in rows) / max(len(rows), 1), 4),
            "avg_citation_hit_rate": round(sum(float(row["citation_hit_rate"]) for row in rows) / max(len(rows), 1), 4),
            "avg_confidence": round(sum(float(row["confidence"]) for row in rows) / max(len(rows), 1), 4),
        }
        return self._persist_report("heuristic", summary, rows)

    def _persist_report(
        self, mode: str, summary: dict[str, object], rows: list[dict[str, object]]
    ) -> EvaluationReport:
        created_at = utc_now_iso()
        report_id = checksum_text(f"{mode}:{created_at}")[:24]
        base_path = self.settings.evaluations_dir / report_id
        base_path.mkdir(parents=True, exist_ok=True)
        json_path = base_path / "report.json"
        csv_path = base_path / "report.csv"
        md_path = base_path / "report.md"

        json_path.write_text(
            json.dumps(
                {"report_id": report_id, "created_at": created_at, "mode": mode, "summary": summary, "rows": rows},
                indent=2,
            ),
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            f"# Evaluation Report: {mode}",
            "",
            f"- Report ID: `{report_id}`",
            f"- Created at: `{created_at}`",
            "",
            "## Summary",
            "",
        ]
        for key, value in summary.items():
            lines.append(f"- {key}: {value}")
        lines.extend(["", "## Rows", "", "| sample_id | answer_f1 | context_recall | citation_hit_rate | confidence | needs_review |", "| --- | --- | --- | --- | --- | --- |"])
        for row in rows:
            lines.append(
                f"| {row['sample_id']} | {row['answer_f1']} | {row['context_recall']} | {row['citation_hit_rate']} | {row['confidence']} | {row['needs_review']} |"
            )
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return EvaluationReport(
            report_id=report_id,
            created_at=created_at,
            mode=mode,
            summary=summary,
            rows=rows,
            artifacts={
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
            },
        )
