"""Optional DeepSeek planning for only the retrieval questions that need it."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from byzantine.models.retrieval import QueryPlan

PLANNER_VERSION = "query-planner-v1"


def compact_context(messages: Sequence[dict[str, str]], *, limit: int = 4) -> list[dict[str, str]]:
    """Keep conversational intent available without treating it as evidence."""
    return [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))[:700]}
        for item in messages[-limit:]
        if item.get("content")
    ]


def plan_with_deepseek(
    base_plan: QueryPlan,
    *,
    conversation_context: Sequence[dict[str, str]] = (),
    client: Any | None = None,
    model: str = "deepseek-chat",
) -> QueryPlan:
    """Return strict JSON planning output; no historical answer is requested."""
    if client is None or not base_plan.needs_agent_planning:
        return base_plan
    context = compact_context(conversation_context)
    prompt = {
        "task": "Create a retrieval plan, not an answer. Return JSON only. Never provide reasoning.",
        "question": base_plan.original_query,
        "local_plan": base_plan.model_dump(mode="json"),
        "conversation_context_for_coreference_only": context,
        "schema": {
            "intent": "fact_lookup|causal_analysis|comparison|process_analysis|person_lookup",
            "rewritten_query": "standalone retrieval query",
            "people": ["canonical reviewed names only when known"],
            "places": ["canonical reviewed places only when known"],
            "topics": ["short topics"],
            "date_start": "integer or null",
            "date_end": "integer or null",
            "subqueries": ["at most 3 retrieval queries"],
            "needs_multi_query": "boolean",
        },
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON. Do not reveal reasoning."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0,
        max_tokens=700,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = (response.choices[0].message.content or "").strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("Query planner did not return a JSON object.")
    return QueryPlan.model_validate(
        {
            **base_plan.model_dump(mode="json"),
            **parsed,
            "original_query": base_plan.original_query,
            "needs_agent_planning": True,
            "used_conversation_context": bool(context),
            "subqueries": list(parsed.get("subqueries") or [])[:3],
        }
    )
