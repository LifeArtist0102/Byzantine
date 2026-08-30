"""Low-cost query analysis based on reviewed local metadata and rules."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from byzantine.metadata.enrichment import extract_date_range, load_seed
from byzantine.models.retrieval import QueryPlan

_COMPLEX = re.compile(
    r"(?:why|how|compare|comparison|impact|effect|cause|process|evolution|respectively|"
    r"为什么|如何|比较|影响|原因|过程|演变|分别|关系|之后|后来|这些|那[些个])",
    re.IGNORECASE,
)
_CAUSAL = re.compile(r"(?:why|cause|impact|effect|because|为什么|原因|影响|导致)", re.IGNORECASE)
_COMPARISON = re.compile(r"(?:compare|comparison|versus|difference|比较|差异|分别)", re.IGNORECASE)
_PROCESS = re.compile(r"(?:how|process|evolution|过程|如何|演变|发展)", re.IGNORECASE)
_PRONOUN = re.compile(r"(?:\b(?:he|she|they|it|this|that|these|those)\b|这|那|其|该)", re.IGNORECASE)


def _matches(question: str, seed: dict[str, object], category: str) -> list[str]:
    matched: list[str] = []
    lowered = question.casefold()
    for record in seed.get(category, []):
        if not isinstance(record, dict):
            continue
        aliases = [str(value) for value in record.get("aliases", [])]
        if any(alias.casefold() in lowered for alias in aliases if alias):
            canonical = str(record.get("canonical") or "").strip()
            if canonical:
                matched.append(canonical)
    return sorted(set(matched))


def _intent(question: str) -> str:
    if _COMPARISON.search(question):
        return "comparison"
    if _CAUSAL.search(question):
        return "causal_analysis"
    if _PROCESS.search(question):
        return "process_analysis"
    return "fact_lookup"


def analyse_query(
    question: str,
    *,
    seed_path: Path | None = None,
    conversation_context: Sequence[dict[str, str]] = (),
) -> QueryPlan:
    """Build a deterministic first-pass plan; it never calls an LLM."""
    seed = load_seed(seed_path) if seed_path and seed_path.is_file() else {}
    start, end, _ = extract_date_range(question)
    topics = [
        name
        for name, words in dict(seed.get("topics", {})).items()
        if any(str(word).casefold() in question.casefold() for word in words)
    ]
    intent = _intent(question)
    needs_context = bool(conversation_context and _PRONOUN.search(question))
    needs_agent = bool(
        needs_context
        or _COMPLEX.search(question)
        or intent in {"comparison", "causal_analysis", "process_analysis"}
        or (len(question.strip()) < 10 and conversation_context)
    )
    return QueryPlan(
        original_query=question,
        rewritten_query=question,
        intent=intent,
        people=_matches(question, seed, "people"),
        places=_matches(question, seed, "places"),
        topics=sorted(set(topics)),
        date_start=start,
        date_end=end,
        needs_agent_planning=needs_agent,
        used_conversation_context=needs_context,
    )
