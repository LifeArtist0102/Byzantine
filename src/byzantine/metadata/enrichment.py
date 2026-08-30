"""Conservative local enrichment for people, places, dates and topics.

This stage deliberately separates trusted dictionary matches from unresolved
capitalized phrases. It prevents an extraction heuristic from silently becoming
historical fact in the vector-store payload.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

YEAR_RANGE = re.compile(r"(?<!\d)(?:c\.\s*)?(\d{3,4})\s*(?:-|–|to)\s*(\d{1,4})(?!\d)")
YEAR = re.compile(r"(?<!\d)(?:c\.\s*)?(\d{3,4})(?!\d)")
PROPER_NAME = re.compile(r"\b(?:[A-Z][a-z]+|[IVX]+)(?:\s+(?:[A-Z][a-z]+|[IVX]+)){1,3}\b")


def _records(seed: dict[str, Any], category: str) -> list[dict[str, Any]]:
    return [
        item for item in seed.get(category, []) if isinstance(item, dict) and item.get("canonical")
    ]


def load_seed(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _alias_matches(text: str, records: Iterable[dict[str, Any]]) -> list[str]:
    matches: list[str] = []
    for record in records:
        aliases = record.get("aliases", [])
        if any(
            re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE) for alias in aliases
        ):
            matches.append(str(record["canonical"]))
    return sorted(set(matches))


def _resolve_short_year(start: int, raw_end: str) -> int:
    if len(raw_end) >= len(str(start)):
        return int(raw_end)
    prefix = str(start)[: len(str(start)) - len(raw_end)]
    return int(prefix + raw_end)


def extract_date_range(
    text: str,
    *,
    historical_year_min: int = 200,
    historical_year_max: int = 1500,
) -> tuple[int | None, int | None, list[str]]:
    values: list[int] = []
    evidence: list[str] = []
    for match in YEAR_RANGE.finditer(text):
        start = int(match.group(1))
        end = _resolve_short_year(start, match.group(2))
        if (
            historical_year_min <= start <= historical_year_max
            and historical_year_min <= end <= historical_year_max
        ):
            values.extend((start, end))
            evidence.append(match.group(0))
    for match in YEAR.finditer(text):
        year = int(match.group(1))
        if historical_year_min <= year <= historical_year_max:
            values.append(year)
            evidence.append(match.group(0))
    return (min(values), max(values), sorted(set(evidence))) if values else (None, None, [])


def extract_topics(text: str, topics: dict[str, list[str]]) -> list[str]:
    lowered = text.casefold()
    detected = [
        topic
        for topic, keywords in topics.items()
        if any(
            re.search(rf"(?<!\w){re.escape(keyword.casefold())}(?!\w)", lowered)
            for keyword in keywords
        )
    ]
    return sorted(detected)


def extract_person_candidates(
    text: str, known_people: list[str], known_places: list[str]
) -> list[str]:
    known = {value.casefold() for value in [*known_people, *known_places]}
    candidates = {
        match.group(0)
        for match in PROPER_NAME.finditer(text)
        if match.group(0).casefold() not in known
    }
    return sorted(candidates)[:20]


def extract_candidates(
    text: str,
    seed: dict[str, Any],
    *,
    people: list[str] | None = None,
    places: list[str] | None = None,
) -> dict[str, list[str]]:
    """Candidate metadata is useful for recall, never a hard historical fact."""
    result: dict[str, list[str]] = {}
    for category in (
        "events",
        "polities",
        "dynasties",
        "offices",
        "institutions",
        "military_groups",
        "religious_groups",
        "works",
    ):
        matches = _alias_matches(text, _records(seed, category))
        if matches:
            result[category] = matches
    aliases = []
    resolved = {"people": people or [], "places": places or [], **result}
    for category in ("people", "places", *result):
        for record in _records(seed, category):
            if record.get("canonical") in resolved.get(category, []):
                aliases.extend(str(item) for item in record.get("aliases", []))
    result["aliases"] = sorted(set(aliases))
    return result


def build_retrieval_text(
    original_text: str,
    *,
    bibliographic: dict[str, Any],
    section_path: list[str],
    trusted: dict[str, Any],
    candidates: dict[str, Any],
    llm: dict[str, Any] | None = None,
) -> str:
    """Build a search-only representation; citation code never reads this field."""
    fields = [
        str(bibliographic.get("title") or ""),
        str(bibliographic.get("author") or ""),
        " > ".join(section_path),
        " ".join(str(value) for value in trusted.get("people", []) + trusted.get("places", [])),
        " ".join(
            str(value) for value in candidates.get("events", []) + candidates.get("aliases", [])
        ),
        " ".join(str(value) for value in trusted.get("themes", [])),
        str((llm or {}).get("contextual_prefix") or ""),
        original_text,
    ]
    return "\n".join(value for value in fields if value.strip())


def enrich_chunk(
    chunk: dict[str, Any],
    seed: dict[str, Any],
    *,
    historical_year_min: int = 200,
    historical_year_max: int = 1500,
) -> dict[str, Any]:
    text = str(chunk.get("original_text", chunk["text"]))
    people = _alias_matches(text, seed.get("people", []))
    places = _alias_matches(text, seed.get("places", []))
    date_start, date_end, date_evidence = extract_date_range(
        text,
        historical_year_min=historical_year_min,
        historical_year_max=historical_year_max,
    )
    trusted = {
        "people": people,
        "places": places,
        "date_start": date_start,
        "date_end": date_end,
        "date_evidence": date_evidence,
        "themes": extract_topics(text, seed.get("topics", {})),
        "section_path": list(chunk.get("section_path", [])),
    }
    candidates = {
        **extract_candidates(text, seed, people=people, places=places),
        "person_candidates": extract_person_candidates(text, people, places),
        "date_ranges": [[date_start, date_end]] if date_start is not None else [],
    }
    metadata = {
        "trusted": trusted,
        "candidate": candidates,
        "llm_inference": dict(chunk.get("llm_inference", {})),
        # Legacy aliases preserve existing filters and user data.
        "people": people,
        "places": places,
        "date_start": date_start,
        "date_end": date_end,
        "date_evidence": date_evidence,
        "topics": trusted["themes"],
        "metadata_provenance": "trusted_local_and_candidate_local_v2",
    }
    return {
        **chunk,
        "original_text": text,
        "text": text,
        "metadata": metadata,
    }


def enrich_chunks(
    chunks_path: Path,
    seed_path: Path,
    output_path: Path,
    *,
    historical_year_min: int = 200,
    historical_year_max: int = 1500,
) -> dict[str, int]:
    seed = load_seed(seed_path)
    with chunks_path.open("r", encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    enriched = [
        enrich_chunk(
            chunk,
            seed,
            historical_year_min=historical_year_min,
            historical_year_max=historical_year_max,
        )
        for chunk in chunks
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in enriched:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return {
        "chunk_count": len(enriched),
        "chunks_with_people": sum(bool(item["metadata"]["people"]) for item in enriched),
        "chunks_with_places": sum(bool(item["metadata"]["places"]) for item in enriched),
        "chunks_with_dates": sum(item["metadata"]["date_start"] is not None for item in enriched),
        "chunks_with_topics": sum(bool(item["metadata"]["topics"]) for item in enriched),
    }
