"""Evidence sufficiency assessment, deliberately separate from citations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from byzantine.models.evidence import Evidence
from byzantine.models.retrieval import QueryPlan, RetrievalAssessment


def assess_evidence(plan: QueryPlan, evidence: Sequence[Evidence]) -> RetrievalAssessment:
    """Fast deterministic quality gate based on coverage rather than count alone."""
    if not evidence:
        return RetrievalAssessment(
            sufficient=False,
            confidence=0.0,
            missing_aspects=["no_retrieved_evidence"],
            should_retry=True,
        )
    covered_people = set().union(*(set(item.metadata.get("people", [])) for item in evidence))
    covered_places = set().union(*(set(item.metadata.get("places", [])) for item in evidence))
    covered_topics = set().union(*(set(item.metadata.get("topics", [])) for item in evidence))
    missing = []
    if plan.people and not covered_people.intersection(plan.people):
        missing.append("people")
    if plan.places and not covered_places.intersection(plan.places):
        missing.append("places")
    if plan.topics and not covered_topics.intersection(plan.topics):
        missing.append("topics")
    sections = {tuple(item.section_path) for item in evidence}
    core_coverage = 1 - len(missing) / max(1, len(plan.people) + len(plan.places) + len(plan.topics))
    diversity = min(1.0, len(sections) / 2)
    volume = min(1.0, len(evidence) / 4)
    confidence = round(0.55 * core_coverage + 0.25 * diversity + 0.20 * volume, 2)
    complex_query = plan.intent in {"causal_analysis", "comparison", "process_analysis"}
    sufficient = not missing and len(evidence) >= (3 if complex_query else 1) and confidence >= 0.58
    return RetrievalAssessment(
        sufficient=sufficient,
        confidence=confidence,
        missing_aspects=missing,
        covered_aspects=sorted({*covered_people, *covered_places, *covered_topics}),
        should_retry=not sufficient,
    )


def grade_with_deepseek(
    plan: QueryPlan,
    evidence: Sequence[Evidence],
    *,
    client: Any | None = None,
    model: str = "deepseek-chat",
) -> RetrievalAssessment | None:
    """Optional one-shot grader for marginal complex questions; no source text dump."""
    if client is None or plan.intent not in {"causal_analysis", "comparison", "process_analysis"}:
        return None
    compact_evidence = [
        {
            "id": item.evidence_id,
            "section": item.section_path,
            "metadata": item.metadata,
            "excerpt": (item.original_text or item.text)[:280],
        }
        for item in evidence[:12]
    ]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only JSON. Do not answer the historical question."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": plan.original_query,
                        "plan": plan.model_dump(mode="json"),
                        "evidence_summaries": compact_evidence,
                        "schema": {
                            "sufficient": "boolean",
                            "covered_aspects": ["string"],
                            "missing_aspects": ["string"],
                            "retry_queries": ["at most 3 strings"],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0,
        max_tokens=600,
        extra_body={"thinking": {"type": "disabled"}},
    )
    payload = json.loads((response.choices[0].message.content or "{}").strip())
    if not isinstance(payload, dict):
        return None
    return RetrievalAssessment(
        sufficient=bool(payload.get("sufficient", False)),
        confidence=0.75 if payload.get("sufficient") else 0.35,
        covered_aspects=list(payload.get("covered_aspects") or []),
        missing_aspects=list(payload.get("missing_aspects") or []),
        retry_queries=list(payload.get("retry_queries") or [])[:3],
        should_retry=not bool(payload.get("sufficient", False)),
    )
