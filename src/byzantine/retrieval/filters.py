"""One metadata-filter definition shared by Qdrant and local fallback paths."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from byzantine.models.evidence import Evidence
from byzantine.models.retrieval import QueryPlan


def metadata_values(plan: QueryPlan) -> dict[str, object]:
    return {
        "people": plan.people,
        "places": plan.places,
        "topics": plan.topics,
        "date_start": plan.date_start,
        "date_end": plan.date_end,
    }


def build_metadata_filter(
    *,
    people: Sequence[str] = (),
    places: Sequence[str] = (),
    topics: Sequence[str] = (),
    date_start: int | None = None,
    date_end: int | None = None,
    document_ids: Sequence[str] = (),
    collection_ids: Sequence[str] = (),
) -> Any | None:
    """Build Qdrant overlap filters with OR-within / AND-across semantics."""
    try:
        from qdrant_client import models
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('Install retrieval dependencies with `pip install -e ".[rag]".') from exc

    must: list[Any] = []
    for key, values in (
        ("document_id", document_ids),
        ("collection_id", collection_ids),
        ("metadata.people", people),
        ("metadata.places", places),
        ("metadata.topics", topics),
    ):
        cleaned = [value.strip() for value in values if value and value.strip()]
        if cleaned:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=cleaned)))
    if date_start is not None:
        must.append(models.FieldCondition(key="metadata.date_end", range=models.Range(gte=date_start)))
    if date_end is not None:
        must.append(models.FieldCondition(key="metadata.date_start", range=models.Range(lte=date_end)))
    return models.Filter(must=must) if must else None


def matches_metadata(item: Evidence, plan: QueryPlan) -> bool:
    """Same filter rules for FTS/local fallback, without assuming metadata is complete."""
    metadata = item.metadata
    people = set(metadata.get("people", []))
    places = set(metadata.get("places", []))
    topics = set(metadata.get("topics", []))
    if plan.people and not people.intersection(plan.people):
        return False
    if plan.places and not places.intersection(plan.places):
        return False
    if plan.topics and not topics.intersection(plan.topics):
        return False
    start, end = metadata.get("date_start"), metadata.get("date_end")
    if plan.date_start is not None and (end is None or int(end) < plan.date_start):
        return False
    return not (plan.date_end is not None and (start is None or int(start) > plan.date_end))
