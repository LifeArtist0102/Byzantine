"""Semantic evidence retrieval from the local Byzantine Qdrant collection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def build_metadata_filter(
    *,
    people: Sequence[str] = (),
    places: Sequence[str] = (),
    topics: Sequence[str] = (),
    date_start: int | None = None,
    date_end: int | None = None,
) -> Any | None:
    """Build an AND filter, with OR semantics inside each label type."""
    try:
        from qdrant_client import models
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('Install retrieval dependencies with `pip install -e ".[rag]"`.') from exc

    must: list[Any] = []
    for key, values in (
        ("metadata.people", people),
        ("metadata.places", places),
        ("metadata.topics", topics),
    ):
        cleaned = [value.strip() for value in values if value.strip()]
        if cleaned:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=cleaned)))

    # A date filter means overlap with the requested historical range.
    if date_start is not None:
        must.append(models.FieldCondition(key="metadata.date_end", range=models.Range(gte=date_start)))
    if date_end is not None:
        must.append(models.FieldCondition(key="metadata.date_start", range=models.Range(lte=date_end)))
    return models.Filter(must=must) if must else None


def load_embedding_model(model_name: str) -> tuple[Any, str]:
    """Load BGE-M3 on the best locally available device."""
    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('Install retrieval dependencies with `pip install -e ".[rag]"`.') from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BGEM3FlagModel(model_name, use_fp16=device == "cuda", device=device), device


def search_local_index(
    query: str,
    *,
    qdrant_path: str,
    collection_name: str,
    model_name: str,
    max_length: int,
    limit: int = 5,
    people: Sequence[str] = (),
    places: Sequence[str] = (),
    topics: Sequence[str] = (),
    date_start: int | None = None,
    date_end: int | None = None,
) -> dict[str, Any]:
    """Encode a user question and return its most relevant source evidence."""
    if not query.strip():
        raise ValueError("Question cannot be empty")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if date_start is not None and date_end is not None and date_start > date_end:
        raise ValueError("date_start must not be later than date_end")

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('Install retrieval dependencies with `pip install -e ".[rag]"`.') from exc

    model, device = load_embedding_model(model_name)
    encoded = model.encode(
        [query],
        batch_size=1,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    query_vector = encoded["dense_vecs"][0].tolist()
    query_filter = build_metadata_filter(
        people=people,
        places=places,
        topics=topics,
        date_start=date_start,
        date_end=date_end,
    )

    client = QdrantClient(path=qdrant_path)
    try:
        if not client.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name!r} does not exist. Run byzantine-index first.")
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits = [
            {
                "score": round(float(point.score), 6),
                "chunk_id": point.payload["chunk_id"],
                "section_path": point.payload["section_path"],
                "page_start": point.payload["page_start"],
                "page_end": point.payload["page_end"],
                "metadata": point.payload["metadata"],
                "text": point.payload["text"],
            }
            for point in response.points
        ]
    finally:
        client.close()

    return {
        "query": query,
        "device": device,
        "collection_name": collection_name,
        "filters": {
            "people": list(people),
            "places": list(places),
            "topics": list(topics),
            "date_start": date_start,
            "date_end": date_end,
        },
        "hits": hits,
    }


def render_evidence(result: dict[str, Any], *, text_limit: int = 800) -> str:
    """Render source evidence for terminal inspection without pretending to answer."""
    lines = [f"Question: {result['query']}", f"Matches: {len(result['hits'])}"]
    for index, hit in enumerate(result["hits"], start=1):
        pages = f"p. {hit['page_start']}" if hit["page_start"] == hit["page_end"] else (
            f"pp. {hit['page_start']}-{hit['page_end']}"
        )
        section = " > ".join(hit["section_path"])
        text = " ".join(hit["text"].split())
        if len(text) > text_limit:
            text = f"{text[:text_limit].rstrip()}…"
        metadata = hit["metadata"]
        labels = []
        for label, key in (("people", "people"), ("places", "places"), ("topics", "topics")):
            values = metadata.get(key, [])
            if values:
                labels.append(f"{label}: {', '.join(values)}")
        if metadata.get("date_start") is not None:
            labels.append(f"dates: {metadata['date_start']}–{metadata['date_end']}")
        lines.extend(
            [
                "",
                f"[{index}] score={hit['score']:.4f} | {pages}",
                f"Section: {section}",
                *(["Metadata: " + " | ".join(labels)] if labels else []),
                text,
            ]
        )
    return "\n".join(lines)
