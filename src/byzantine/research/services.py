"""Small, testable workflows behind the five history-research features."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from byzantine.models.evidence import Evidence


def parallel_reading(question: str, evidence: Sequence[Evidence], dimensions: Sequence[str]) -> dict[str, Any]:
    """Keep every document's evidence separate instead of producing a blended summary."""
    cells = []
    for item in evidence:
        for dimension in dimensions:
            cells.append({"document_id": item.document_id, "dimension": dimension, "analysis": f"与“{dimension}”相关的原文证据。", "evidence_ids": [item.evidence_id], "epistemic_type": item.epistemic_type})
    return {"comparison_id": f"comparison_{uuid.uuid4().hex}", "question": question, "selected_document_ids": sorted({item.document_id for item in evidence}), "dimensions": list(dimensions), "comparison_cells": cells, "summary": "各文献证据分列展示；未找到证据的维度不作结论。", "created_at": datetime.now(UTC).isoformat()}


def split_claim_units(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[。！？.!?])\s*", text) if sentence.strip()]


def audit_draft(text: str, search: Callable[[str], Sequence[Evidence]]) -> list[dict[str, Any]]:
    """Audit sentences without altering the writer's draft."""
    results = []
    for sentence in split_claim_units(text):
        evidence = list(search(sentence))
        overstated = bool(re.search(r"彻底|必然|完全|所有|从不|always|never|entirely", sentence, re.IGNORECASE))
        status = "overstated" if overstated and evidence else ("supported" if evidence else "unsupported")
        results.append({"sentence": sentence, "status": status, "supporting_evidence": [item.evidence_id for item in evidence], "opposing_evidence": [], "qualifying_evidence": [], "issue": "表述强度可能超过现有证据。" if overstated else ("" if evidence else "当前资料未找到支持证据。"), "suggestion": "建议使用“现有资料表明”“在一定程度上”等限定语。" if overstated else ""})
    return results


def classify_difference(left: Evidence, right: Evidence) -> str:
    """A cautious first-pass classification: never call all variation contradiction."""
    left_years, right_years = set(re.findall(r"\b\d{3,4}\b", left.text)), set(re.findall(r"\b\d{3,4}\b", right.text))
    if left_years and right_years and left_years.isdisjoint(right_years):
        return "potential direct_contradiction"
    if left.language != right.language:
        return "translation_difference"
    return "different_perspective"


def counter_queries(claim: str) -> list[str]:
    return [claim, f"evidence against or limiting: {claim}", f"different explanation for: {claim}"]
