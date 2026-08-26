"""Keyword/vector retrieval fusion for the local multi-document library."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from byzantine.models.evidence import Evidence
from byzantine.storage.database import LibraryDatabase


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[str]:
    """Fuse ranked IDs without comparing incompatible keyword/vector scores."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
    return [identifier for identifier, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def keyword_evidence(database: LibraryDatabase, query: str, *, document_ids: Sequence[str] = (), collection_ids: Sequence[str] = (), limit: int = 20) -> list[Evidence]:
    """Return FTS5 results. SQLite's default tokenizer has limited Chinese segmentation."""
    try:
        rows = database.fts_search(query, document_ids=document_ids, collection_ids=collection_ids, limit=limit)
    except Exception as exc:
        raise ValueError("关键词检索无法解析此查询；请尝试更短的关键词或英文术语。") from exc
    return [database.evidence_from_row(row) for row in rows]


def hybrid_search(query: str, *, database: LibraryDatabase, vector_search: Callable[..., Sequence[Evidence]] | None = None, document_ids: Sequence[str] = (), collection_ids: Sequence[str] = (), top_k: int = 5) -> list[Evidence]:
    """Use FTS5 and optional BGE/Qdrant hits, then return canonical Evidence objects."""
    keywords = keyword_evidence(database, query, document_ids=document_ids, collection_ids=collection_ids, limit=top_k * 4)
    vectors = list(vector_search(query, document_ids=document_ids, collection_ids=collection_ids, limit=top_k * 4)) if vector_search else []
    merged = reciprocal_rank_fusion([[item.chunk_id for item in keywords], [item.chunk_id for item in vectors]])
    lookup = {item.chunk_id: item for item in [*keywords, *vectors]}
    return [lookup[chunk_id] for chunk_id in merged[:top_k]]
