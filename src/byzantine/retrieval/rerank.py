"""Pluggable local reranking with a safe deterministic fallback."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from byzantine.models.evidence import Evidence


def rerank_candidates(
    query: str,
    candidates: Sequence[Evidence],
    *,
    reranker: Callable[[str, Sequence[Evidence]], Sequence[Evidence]] | None = None,
    limit: int = 8,
) -> list[Evidence]:
    """Apply an optional local reranker only to fused candidates, never fail closed."""
    if not candidates:
        return []
    if reranker is None:
        return list(candidates)[:limit]
    try:
        return list(reranker(query, candidates[:30]))[:limit]
    except (ImportError, RuntimeError, ValueError):
        return list(candidates)[:limit]
