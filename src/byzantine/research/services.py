"""Small, evidence-first workflows behind the history-research features."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from byzantine.models.evidence import Evidence


def parallel_reading(
    question: str, evidence: Sequence[Evidence], dimensions: Sequence[str]
) -> dict[str, Any]:
    """Analyse each selected document independently before comparison."""
    by_document: dict[str, list[Evidence]] = {}
    for item in evidence:
        by_document.setdefault(item.document_id, []).append(item)
    cells = []
    for document_id, items in by_document.items():
        for dimension in dimensions:
            excerpts = [
                " ".join((item.original_text or item.text).split())[:260] for item in items[:2]
            ]
            cells.append(
                {
                    "document_id": document_id,
                    "dimension": dimension,
                    "analysis": "；".join(excerpts) or "该文献没有足以分析此维度的证据。",
                    "evidence_ids": [item.evidence_id for item in items[:2]],
                    "epistemic_type": items[0].epistemic_type if items else "unknown",
                }
            )
    return {
        "comparison_id": f"comparison_{uuid.uuid4().hex}",
        "question": question,
        "selected_document_ids": sorted(by_document),
        "dimensions": list(dimensions),
        "comparison_cells": cells,
        "summary": "每篇文献独立检索并保留其可追溯证据；差异需要研究者进一步判断。",
        "created_at": datetime.now(UTC).isoformat(),
    }


def split_claim_units(text: str) -> list[str]:
    return [
        sentence.strip() for sentence in re.split(r"(?<=[。！？.!?])\s*", text) if sentence.strip()
    ]


def audit_draft(text: str, search: Callable[[str], Sequence[Evidence]]) -> list[dict[str, Any]]:
    results = []
    for sentence in split_claim_units(text):
        evidence = list(search(sentence))
        overstated = bool(
            re.search(r"彻底|必然|完全|所有|从不|always|never|entirely", sentence, re.IGNORECASE)
        )
        status = (
            "overstated"
            if overstated and evidence
            else ("supported" if evidence else "unsupported")
        )
        results.append(
            {
                "sentence": sentence,
                "status": status,
                "supporting_evidence": [item.evidence_id for item in evidence],
                "opposing_evidence": [],
                "qualifying_evidence": [],
            }
        )
    return results


def comparable(left: Evidence, right: Evidence) -> bool:
    def terms(item: Evidence) -> set[str]:
        candidate = item.metadata.get("candidate", {})
        return (
            set(item.metadata.get("people", []))
            | set(item.metadata.get("places", []))
            | set(candidate.get("events", []))
        )

    return bool(terms(left) and terms(right) and terms(left) & terms(right))


def classify_difference(left: Evidence, right: Evidence) -> str:
    """Different years alone are not evidence of a contradiction."""
    if not comparable(left, right):
        return "not_comparable"
    if left.language != right.language:
        return "translation_difference"
    return "comparable_difference"


def counter_queries(claim: str) -> list[str]:
    return [claim, f"evidence against or limiting: {claim}", f"different explanation for: {claim}"]
