from __future__ import annotations

from src.config.settings import AppSettings
from src.utils.models import GuardrailResult, RetrievalHit
from src.utils.text import word_overlap_score


def evaluate_guardrails(
    settings: AppSettings,
    question: str,
    answer: str,
    citations: list[RetrievalHit],
    retrieval_hits: list[RetrievalHit],
    reflection_confidence: float,
) -> GuardrailResult:
    citation_coverage = min(len(citations) / max(min(len(retrieval_hits), 4), 1), 1.0)
    best_retrieval_score = retrieval_hits[0].score if retrieval_hits else 0.0
    answer_grounding = max((word_overlap_score(answer, hit.text) for hit in citations), default=0.0)
    confidence = round(
        0.45 * reflection_confidence
        + 0.35 * best_retrieval_score
        + 0.2 * answer_grounding,
        4,
    )
    flags: list[str] = []
    retrieval_floor_met = best_retrieval_score >= settings.retrieval_score_floor
    if confidence < settings.confidence_threshold:
        flags.append("Confidence below configured threshold.")
    if citation_coverage < settings.citation_coverage_threshold:
        flags.append("Citation coverage below configured threshold.")
    if not retrieval_floor_met:
        flags.append("Top retrieval score is below the configured floor.")
    passed = not flags
    return GuardrailResult(
        passed=passed,
        confidence=confidence,
        citation_coverage=round(citation_coverage, 4),
        retrieval_floor_met=retrieval_floor_met,
        risk_flags=flags,
    )
