from __future__ import annotations

import html
import re
from collections import Counter
from typing import Iterable


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufeff", " ").replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def approximate_token_count(text: str) -> int:
    return len(tokenize(text))


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    sentences = [piece.strip() for piece in SENTENCE_RE.split(normalized) if piece.strip()]
    if len(sentences) <= 1:
        return [line.strip() for line in normalized.splitlines() if line.strip()]
    return sentences


def split_paragraphs(text: str) -> list[str]:
    return [piece.strip() for piece in normalize_text(text).split("\n\n") if piece.strip()]


def heading_density(text: str) -> float:
    lines = [line.strip() for line in normalize_text(text).splitlines() if line.strip()]
    if not lines:
        return 0.0
    heading_like = [
        line
        for line in lines
        if line.endswith(":")
        or line.startswith("#")
        or (line.istitle() and len(line.split()) <= 8)
        or re.match(r"^\d+(\.\d+)*\s+[A-Z]", line)
    ]
    return len(heading_like) / len(lines)


def line_density(text: str) -> float:
    lines = [line for line in normalize_text(text).splitlines() if line.strip()]
    if not lines:
        return 0.0
    total_chars = sum(len(line) for line in lines)
    return total_chars / max(len(lines), 1)


def delimiter_profile(text: str) -> dict[str, int]:
    normalized = normalize_text(text)
    return {
        "blank_lines": normalized.count("\n\n"),
        "bullet_lines": len(re.findall(r"(?m)^\s*[-*]\s+", normalized)),
        "code_braces": normalized.count("{") + normalized.count("}"),
        "commas": normalized.count(","),
        "pipes": normalized.count("|"),
        "colons": normalized.count(":"),
    }


def word_overlap_score(query: str, text: str) -> float:
    query_terms = Counter(tokenize(query))
    text_terms = Counter(tokenize(text))
    if not query_terms or not text_terms:
        return 0.0
    overlap = sum(min(count, text_terms[token]) for token, count in query_terms.items())
    return overlap / max(sum(query_terms.values()), 1)


def sentence_relevance(query: str, text: str, limit: int = 5) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for sentence in split_sentences(text):
        score = word_overlap_score(query, sentence)
        if score > 0:
            results.append({"sentence": sentence, "score": round(score, 4)})
    results.sort(key=lambda item: float(item["score"]), reverse=True)
    return results[:limit]


def highlight_text(text: str, terms: Iterable[str]) -> str:
    safe = html.escape(text)
    unique_terms = sorted({term for term in terms if term}, key=len, reverse=True)
    for term in unique_terms:
        pattern = re.compile(re.escape(html.escape(term)), re.IGNORECASE)
        safe = pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", safe)
    return safe
