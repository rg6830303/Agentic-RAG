from __future__ import annotations

import json
import re
from dataclasses import asdict

from src.agentic.guardrails import evaluate_guardrails
from src.checkpoints.service import CheckpointManager
from src.config.settings import AppSettings
from src.providers.azure_openai import AzureOpenAIProvider, ProviderError
from src.retrieval.engine import RetrievalEngine
from src.utils.models import AnswerBundle, CheckpointRecord, QueryOptions, RetrievalHit
from src.utils.text import sentence_relevance, split_sentences


class AgenticRAGService:
    def __init__(
        self,
        settings: AppSettings,
        provider: AzureOpenAIProvider,
        retrieval_engine: RetrievalEngine,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.retrieval_engine = retrieval_engine
        self.checkpoint_manager = checkpoint_manager

    def retrieve_contexts(
        self,
        question: str,
        options: QueryOptions,
        require_human_review: bool = False,
    ) -> tuple[list[RetrievalHit], list[CheckpointRecord]]:
        hits = self.retrieval_engine.retrieve(question, options)
        checkpoint = self.checkpoint_manager.create(
            stage="post-retrieval_pre-generation",
            payload={
                "question": question,
                "retrieved_count": len(hits),
                "retrieval_mode": options.retrieval_mode,
                "sources": [hit.file_name for hit in hits[:5]],
            },
            enabled=options.checkpoints_enabled,
            requires_human=require_human_review,
        )
        return hits, [checkpoint]

    def answer(self, question: str, options: QueryOptions) -> AnswerBundle:
        hits, checkpoints = self.retrieve_contexts(
            question,
            options,
            require_human_review=options.require_context_review,
        )
        return self.generate_from_hits(question, hits, options, checkpoints=checkpoints)

    def generate_from_hits(
        self,
        question: str,
        hits: list[RetrievalHit],
        options: QueryOptions,
        checkpoints: list[CheckpointRecord] | None = None,
    ) -> AnswerBundle:
        checkpoints = checkpoints or []
        if not hits:
            guardrails = evaluate_guardrails(
                self.settings,
                question=question,
                answer="",
                citations=[],
                retrieval_hits=[],
                reflection_confidence=0.0,
            )
            return AnswerBundle(
                question=question,
                answer="No relevant context was found in the local indexes. Try ingesting more files or enabling another retrieval branch.",
                citations=[],
                used_methods=[],
                retrieval_mode=options.retrieval_mode,
                self_rag_enabled=options.self_rag,
                confidence=0.0,
                needs_review=True,
                guardrails=guardrails,
                reflection="No retrieval results were available.",
                checkpoints=checkpoints,
            )

        draft_answer = self._compose_answer(question, hits)
        reflection, reflection_confidence, needs_more_context = self._reflect(
            question, draft_answer, hits
        )

        refined_hits = hits
        if options.self_rag and needs_more_context:
            expanded_query = self._expand_query(question, hits)
            second_pass = self.retrieval_engine.retrieve(expanded_query, options)
            if second_pass:
                seen = {hit.chunk_id for hit in hits}
                refined_hits = hits + [hit for hit in second_pass if hit.chunk_id not in seen]
                refined_hits.sort(key=lambda item: item.score, reverse=True)
                refined_hits = refined_hits[: max(options.top_k, 6)]
                draft_answer = self._compose_answer(question, refined_hits)
                reflection, reflection_confidence, _ = self._reflect(
                    question, draft_answer, refined_hits
                )

        citations = refined_hits[: min(5, len(refined_hits))]
        final_checkpoint = self.checkpoint_manager.create(
            stage="post-generation_pre-final-answer",
            payload={
                "question": question,
                "citation_count": len(citations),
                "self_rag": options.self_rag,
            },
            enabled=options.checkpoints_enabled,
            requires_human=options.require_final_approval,
        )
        checkpoints.append(final_checkpoint)
        guardrails = evaluate_guardrails(
            self.settings,
            question=question,
            answer=draft_answer,
            citations=citations,
            retrieval_hits=refined_hits,
            reflection_confidence=reflection_confidence,
        )
        needs_review = options.require_final_approval or not guardrails.passed
        return AnswerBundle(
            question=question,
            answer=draft_answer,
            citations=citations,
            used_methods=sorted({hit.source for hit in refined_hits}),
            retrieval_mode=options.retrieval_mode,
            self_rag_enabled=options.self_rag,
            confidence=guardrails.confidence,
            needs_review=needs_review,
            guardrails=guardrails,
            reflection=reflection,
            evaluation_summary={
                "sentence_attention_enabled": options.sentence_attention,
                "retrieved_chunks": len(refined_hits),
            },
            checkpoints=checkpoints,
            metadata={
                "top_hits": [asdict(hit) for hit in refined_hits[:5]],
                "manual_finalization_required": options.require_final_approval,
            },
        )

    def _compose_answer(self, question: str, hits: list[RetrievalHit]) -> str:
        if self.settings.chat_available:
            context_lines = []
            for index, hit in enumerate(hits[:6], start=1):
                context_lines.append(
                    f"[{index}] file={hit.file_name} score={hit.score:.3f} chunk_id={hit.chunk_id}\n{hit.text}"
                )
            system_prompt = (
                "You answer using only the provided context. If context is insufficient, say so plainly. "
                "Prefer concise, citation-aware answers and mention file names inline when useful."
            )
            user_prompt = (
                f"Question:\n{question}\n\n"
                "Context:\n"
                + "\n\n".join(context_lines)
                + "\n\nReturn a grounded answer with brief source-aware wording."
            )
            try:
                return self.provider.chat_completion(system_prompt, user_prompt)
            except ProviderError:
                pass
        return self._extractive_answer(question, hits)

    def _extractive_answer(self, question: str, hits: list[RetrievalHit]) -> str:
        selected_sentences: list[str] = []
        for hit in hits[:4]:
            for item in sentence_relevance(question, hit.text, limit=2):
                sentence = str(item["sentence"])
                if sentence not in selected_sentences:
                    selected_sentences.append(sentence)
        if not selected_sentences:
            selected_sentences = [split_sentences(hits[0].text)[0] if split_sentences(hits[0].text) else hits[0].text[:300]]
        answer = " ".join(selected_sentences[:5]).strip()
        return (
            f"Local extractive answer: {answer} "
            "This draft uses retrieved evidence only because Azure chat generation is unavailable or declined."
        )

    def _reflect(
        self,
        question: str,
        answer: str,
        hits: list[RetrievalHit],
    ) -> tuple[str, float, bool]:
        if self.settings.chat_available:
            prompt = (
                "Review the groundedness of this answer. Return JSON with keys confidence (0-1), "
                "needs_more_context (true/false), and reasoning.\n\n"
                f"Question: {question}\n\nAnswer: {answer}\n\n"
                + "\n\n".join(f"- {hit.file_name}: {hit.text[:500]}" for hit in hits[:4])
            )
            try:
                raw = self.provider.chat_completion(
                    "You are a retrieval critic. Respond with JSON only.",
                    prompt,
                    temperature=0.0,
                    max_tokens=220,
                )
                parsed = json.loads(self._extract_json(raw))
                confidence = float(parsed.get("confidence", 0.5))
                needs_more = bool(parsed.get("needs_more_context", False))
                reasoning = str(parsed.get("reasoning", ""))
                return reasoning, confidence, needs_more
            except (ProviderError, ValueError, json.JSONDecodeError):
                pass
        best_score = hits[0].score if hits else 0.0
        confidence = min(0.95, max(0.2, best_score + 0.1))
        needs_more = confidence < 0.55
        reasoning = (
            "Heuristic reflection inferred confidence from retrieval strength because a model-based reflection pass was unavailable."
        )
        return reasoning, confidence, needs_more

    def _expand_query(self, question: str, hits: list[RetrievalHit]) -> str:
        key_phrases = []
        for hit in hits[:3]:
            for sentence in split_sentences(hit.text)[:2]:
                if len(sentence.split()) >= 4:
                    key_phrases.append(sentence)
        suffix = " ".join(key_phrases[:2])
        return f"{question}\nFocus on: {suffix}"

    @staticmethod
    def _extract_json(raw: str) -> str:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        return match.group(0) if match else raw
