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


def load_seed(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _alias_matches(text: str, records: Iterable[dict[str, Any]]) -> list[str]:
    matches: list[str] = []
    for record in records:
        aliases = record.get("aliases", [])
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE) for alias in aliases):
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
        if historical_year_min <= start <= historical_year_max and historical_year_min <= end <= historical_year_max:
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
        if any(re.search(rf"(?<!\w){re.escape(keyword.casefold())}(?!\w)", lowered) for keyword in keywords)
    ]
    return sorted(detected)


def extract_person_candidates(text: str, known_people: list[str], known_places: list[str]) -> list[str]:
    known = {value.casefold() for value in [*known_people, *known_places]}
    candidates = {
        match.group(0)
        for match in PROPER_NAME.finditer(text)
        if match.group(0).casefold() not in known
    }
    return sorted(candidates)[:20]


def enrich_chunk(
    chunk: dict[str, Any],
    seed: dict[str, Any],
    *,
    historical_year_min: int = 200,
    historical_year_max: int = 1500,
) -> dict[str, Any]:
    text = str(chunk["text"])
    people = _alias_matches(text, seed.get("people", []))
    places = _alias_matches(text, seed.get("places", []))
    date_start, date_end, date_evidence = extract_date_range(
        text,
        historical_year_min=historical_year_min,
        historical_year_max=historical_year_max,
    )
    return {
        **chunk,
        "metadata": {
            "people": people,
            "person_candidates": extract_person_candidates(text, people, places),
            "places": places,
            "date_start": date_start,
            "date_end": date_end,
            "date_evidence": date_evidence,
            "topics": extract_topics(text, seed.get("topics", {})),
            "metadata_provenance": "local_seed_and_regex_v1",
        },
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
