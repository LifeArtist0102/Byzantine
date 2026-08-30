"""Keyword/vector retrieval fusion for the local multi-document library."""

from __future__ import annotations

import os
import re
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from byzantine.chunking.semantic import estimate_tokens
from byzantine.metadata.enrichment import extract_date_range, load_seed
from byzantine.models.evidence import Evidence
from byzantine.models.retrieval import QueryPlan
from byzantine.storage.database import LibraryDatabase


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[str]:
    """Fuse ranked IDs without comparing incompatible keyword/vector scores."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
    return [
        identifier for identifier, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def keyword_evidence(
    database: LibraryDatabase,
    query: str,
    *,
    document_ids: Sequence[str] = (),
    collection_ids: Sequence[str] = (),
    limit: int = 20,
) -> list[Evidence]:
    """Return FTS5 results. SQLite's default tokenizer has limited Chinese segmentation."""
    try:
        rows = database.fts_search(
            query, document_ids=document_ids, collection_ids=collection_ids, limit=limit
        )
    except Exception as exc:
        raise ValueError("关键词检索无法解析此查询；请尝试更短的关键词或英文术语。") from exc
    return [database.evidence_from_row(row) for row in rows]


def hybrid_search(
    query: str,
    *,
    database: LibraryDatabase,
    vector_search: Callable[..., Sequence[Evidence]] | None = None,
    document_ids: Sequence[str] = (),
    collection_ids: Sequence[str] = (),
    top_k: int = 5,
) -> list[Evidence]:
    """Use FTS5 and optional BGE/Qdrant hits, then return canonical Evidence objects."""
    keywords = keyword_evidence(
        database, query, document_ids=document_ids, collection_ids=collection_ids, limit=top_k * 4
    )
    vectors = (
        list(
            vector_search(
                query, document_ids=document_ids, collection_ids=collection_ids, limit=top_k * 4
            )
        )
        if vector_search
        else []
    )
    merged = reciprocal_rank_fusion(
        [[item.chunk_id for item in keywords], [item.chunk_id for item in vectors]]
    )
    lookup = {item.chunk_id: item for item in [*keywords, *vectors]}
    return [lookup[chunk_id] for chunk_id in merged[:top_k]]


@dataclass(frozen=True)
class ParsedQuery:
    original: str
    people: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    date_start: int | None = None
    date_end: int | None = None


def parse_local_query(question: str, seed_path: Path | None = None) -> ParsedQuery:
    seed = load_seed(seed_path) if seed_path and seed_path.is_file() else {}

    def matched(category: str) -> tuple[list[str], list[str]]:
        canonical, aliases = [], []
        for record in seed.get(category, []):
            if not isinstance(record, dict):
                continue
            names = [str(value) for value in record.get("aliases", [])]
            if any(name.casefold() in question.casefold() for name in names):
                canonical.append(str(record.get("canonical")))
                aliases.extend(names)
        return sorted(set(canonical)), sorted(set(aliases))

    people, person_aliases = matched("people")
    places, place_aliases = matched("places")
    events, event_aliases = matched("events")
    date_start, date_end, _ = extract_date_range(question)
    themes = [
        name
        for name, words in seed.get("topics", {}).items()
        if any(word.casefold() in question.casefold() for word in words)
    ]
    return ParsedQuery(
        question,
        people,
        places,
        events,
        themes,
        sorted(set(person_aliases + place_aliases + event_aliases)),
        date_start,
        date_end,
    )


def _fts_query(parsed: ParsedQuery) -> str:
    terms = [parsed.original, *parsed.people, *parsed.places, *parsed.events, *parsed.aliases]
    tokens = re.findall(r"[\w'-]+", " ".join(terms))
    return " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens) if token) or parsed.original


def _boost(item: Evidence, parsed: ParsedQuery) -> float:
    metadata = item.metadata
    trusted = dict(metadata.get("trusted", {}))
    candidate = dict(metadata.get("candidate", {}))
    score = 0.0
    for wanted, key in (
        (parsed.people, "people"),
        (parsed.places, "places"),
        (parsed.themes, "themes"),
    ):
        score += 2.0 * len(set(wanted) & set(trusted.get(key, [])))
    score += 0.65 * len(set(parsed.events) & set(candidate.get("events", [])))
    score += 0.25 * len(set(parsed.aliases) & set(candidate.get("aliases", [])))
    start, end = trusted.get("date_start"), trusted.get("date_end")
    if (
        parsed.date_start is not None
        and start is not None
        and end is not None
        and start <= parsed.date_end <= end
    ):
        score += 1.5
    return score


def _dedupe_ranked(ranked: list[tuple[Evidence, float]], limit: int) -> list[Evidence]:
    output: list[Evidence] = []
    parents, pages, signatures = defaultdict(int), defaultdict(int), set()
    for item, _ in ranked:
        parent = item.parent_id or str(item.metadata.get("parent_id") or item.chunk_id)
        page = item.pdf_page_start
        signature = re.sub(r"\W+", "", (item.original_text or item.text).lower())[:180]
        if (
            parents[parent] >= 2
            or (page is not None and pages[page] >= 2)
            or signature in signatures
        ):
            continue
        output.append(item)
        parents[parent] += 1
        pages[page] += 1
        signatures.add(signature)
        if len(output) >= limit:
            break
    return output


def retrieve_evidence(
    question: str,
    *,
    database: LibraryDatabase,
    vector_search: Callable[..., Sequence[Evidence]] | None = None,
    sparse_search: Callable[..., Sequence[Evidence]] | None = None,
    document_ids: Sequence[str] = (),
    collection_ids: Sequence[str] = (),
    seed_path: Path | None = None,
    top_k: int = 8,
    colbert_reranker: Callable[[str, Sequence[Evidence]], Sequence[Evidence]] | None = None,
    query_plan: QueryPlan | None = None,
) -> list[Evidence]:
    """Dense + sparse + FTS recall, RRF fusion and metadata-aware ranking."""
    if query_plan is None:
        parsed = parse_local_query(question, seed_path)
        retrieval_query = question
    else:
        parsed = ParsedQuery(
            original=query_plan.rewritten_query,
            people=query_plan.people,
            places=query_plan.places,
            themes=query_plan.topics,
            date_start=query_plan.date_start,
            date_end=query_plan.date_end,
        )
        retrieval_query = query_plan.rewritten_query
    metadata_kwargs = {
        "people": parsed.people,
        "places": parsed.places,
        "topics": parsed.themes,
        "date_start": parsed.date_start,
        "date_end": parsed.date_end,
    }
    dense = (
        list(
            vector_search(
                retrieval_query,
                document_ids=document_ids,
                collection_ids=collection_ids,
                limit=40,
                **metadata_kwargs,
            )
        )
        if vector_search
        else []
    )
    sparse_query = " ".join(
        [retrieval_query, *parsed.people, *parsed.places, *parsed.events, *parsed.aliases]
    )
    sparse = (
        list(
            sparse_search(
                sparse_query,
                document_ids=document_ids,
                collection_ids=collection_ids,
                limit=40,
                **metadata_kwargs,
            )
        )
        if sparse_search
        else []
    )
    try:
        fts = [
            database.evidence_from_row(row)
            for row in database.fts_search(
                _fts_query(parsed),
                document_ids=document_ids,
                collection_ids=collection_ids,
                limit=40,
            )
        ]
    except (sqlite3.OperationalError, ValueError):
        fts = keyword_evidence(
            database,
            retrieval_query,
            document_ids=document_ids,
            collection_ids=collection_ids,
            limit=40,
        )
    rankings = [[item.chunk_id for item in ranking] for ranking in (dense, sparse, fts)]
    rrf = reciprocal_rank_fusion(rankings)
    lookup = {item.chunk_id: item for item in [*dense, *sparse, *fts]}
    base = {item_id: 1.0 / (60 + rank) for rank, item_id in enumerate(rrf, 1)}
    ranked = sorted(
        ((lookup[item_id], base[item_id] + _boost(lookup[item_id], parsed)) for item_id in rrf),
        key=lambda value: value[1],
        reverse=True,
    )
    selected = _dedupe_ranked(ranked, max(top_k, 30))
    if colbert_reranker and os.getenv("BYZANTINE_ENABLE_COLBERT", "0") == "1":
        selected = list(colbert_reranker(retrieval_query, selected[:30]))
    return selected[:top_k]


def expand_context(
    evidence: Sequence[Evidence],
    *,
    database: LibraryDatabase,
    question: str,
    token_budget: int = 4200,
) -> list[Evidence]:
    """Attach only needed neighbouring child text under a strict source budget."""
    expanded: list[Evidence] = []
    used = 0
    causal = bool(
        re.search(r"原因|过程|观点|because|why|process|argument", question, re.IGNORECASE)
    )
    for item in evidence:
        text = item.original_text or item.text
        needs_parent = causal or bool(re.search(r"\b(he|she|they|this|it)\b", text, re.IGNORECASE))
        if needs_parent and item.parent_id:
            siblings = database.parent_evidence(item.document_id, item.parent_id)
            selected = [
                value
                for value in siblings
                if abs(
                    int(value.chunk_id.rsplit("_", 1)[-1]) - int(item.chunk_id.rsplit("_", 1)[-1])
                )
                <= 1
            ]
            merged = "\n\n".join(value.original_text or value.text for value in selected)
            if merged and estimate_tokens(merged) + used <= token_budget:
                pages = [
                    page
                    for value in selected
                    for page in (value.pdf_page_start, value.pdf_page_end)
                    if page is not None
                ]
                regions = []
                seen_regions: set[str] = set()
                for value in selected:
                    for region in value.source_regions:
                        key = region.model_dump_json()
                        if key not in seen_regions:
                            seen_regions.add(key)
                            regions.append(region)
                item = item.model_copy(
                    update={
                        "text": merged,
                        "original_text": merged,
                        "pdf_page_start": min(pages) if pages else item.pdf_page_start,
                        "pdf_page_end": max(pages) if pages else item.pdf_page_end,
                        "source_regions": regions,
                        "metadata": {
                            **item.metadata,
                            "context_chunk_ids": [value.chunk_id for value in selected],
                        },
                    }
                )
                text = merged
        tokens = estimate_tokens(text)
        if used + tokens > token_budget:
            continue
        expanded.append(item)
        used += tokens
    return expanded
