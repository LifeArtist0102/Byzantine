"""The single adaptive retrieval core used by chat and research workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from byzantine.models.evidence import Evidence
from byzantine.models.retrieval import QueryPlan, RetrievalResult
from byzantine.retrieval.filters import matches_metadata, metadata_values
from byzantine.retrieval.hybrid import retrieve_evidence
from byzantine.retrieval.planner import plan_with_deepseek
from byzantine.retrieval.quality import assess_evidence, grade_with_deepseek
from byzantine.retrieval.query_analysis import analyse_query
from byzantine.retrieval.rerank import rerank_candidates
from byzantine.storage.database import LibraryDatabase

Progress = Callable[[str], None]


def _without_metadata(plan: QueryPlan, *, partial: bool = False) -> QueryPlan:
    updates: dict[str, object] = {"topics": [], "date_start": None, "date_end": None}
    if not partial:
        updates.update({"people": [], "places": []})
    return plan.model_copy(update=updates)


def _retrieve_once(
    plan: QueryPlan,
    *,
    database: LibraryDatabase,
    vector_search: Callable[..., Sequence[Evidence]] | None,
    sparse_search: Callable[..., Sequence[Evidence]] | None,
    document_ids: Sequence[str],
    collection_ids: Sequence[str],
) -> list[Evidence]:
    return retrieve_evidence(
        plan.original_query,
        database=database,
        vector_search=vector_search,
        sparse_search=sparse_search,
        document_ids=document_ids,
        collection_ids=collection_ids,
        top_k=20,
        query_plan=plan,
    )


def _retrieve_plans(
    plans: Sequence[QueryPlan],
    *,
    database: LibraryDatabase,
    vector_search: Callable[..., Sequence[Evidence]] | None,
    sparse_search: Callable[..., Sequence[Evidence]] | None,
    document_ids: Sequence[str],
    collection_ids: Sequence[str],
) -> list[Evidence]:
    hits: list[Evidence] = []
    for plan in plans:
        hits.extend(
            _retrieve_once(
                plan,
                database=database,
                vector_search=vector_search,
                sparse_search=sparse_search,
                document_ids=document_ids,
                collection_ids=collection_ids,
            )
        )
    return _deduplicate(hits)


def _first_round_plans(plan: QueryPlan) -> list[QueryPlan]:
    plans = [plan]
    if plan.needs_multi_query and plan.subqueries:
        plans.extend(
            plan.model_copy(
                update={"rewritten_query": query, "needs_multi_query": False, "subqueries": []}
            )
            for query in plan.subqueries[:3]
            if query.strip() and query != plan.rewritten_query
        )
    return plans


def _metadata_fallback(
    plans: Sequence[QueryPlan],
    strict: list[Evidence],
    *,
    database: LibraryDatabase,
    vector_search: Callable[..., Sequence[Evidence]] | None,
    sparse_search: Callable[..., Sequence[Evidence]] | None,
    document_ids: Sequence[str],
    collection_ids: Sequence[str],
) -> list[Evidence]:
    """Relax incomplete metadata in deterministic stages to protect recall."""
    plan = plans[0]
    if not any(metadata_values(plan).values()) or len(strict) >= 3:
        return strict
    partial_plans = [_without_metadata(item, partial=True) for item in plans]
    partial = _retrieve_plans(
        partial_plans,
        database=database,
        vector_search=vector_search,
        sparse_search=sparse_search,
        document_ids=document_ids,
        collection_ids=collection_ids,
    )
    partial_matches = [item for item in partial if matches_metadata(item, partial_plans[0])]
    if len(partial_matches) >= 3:
        return partial_matches
    return _retrieve_plans(
        [_without_metadata(item) for item in plans],
        database=database,
        vector_search=vector_search,
        sparse_search=sparse_search,
        document_ids=document_ids,
        collection_ids=collection_ids,
    )


def _deduplicate(items: Sequence[Evidence]) -> list[Evidence]:
    unique: dict[str, Evidence] = {}
    for item in items:
        unique.setdefault(item.chunk_id, item)
    return list(unique.values())


def _retry_queries(plan: QueryPlan, assessment_retry_queries: Sequence[str], missing: Sequence[str]) -> list[str]:
    if assessment_retry_queries:
        return list(assessment_retry_queries)[:3]
    if plan.subqueries:
        return plan.subqueries[:3]
    suffix = " ".join(missing)
    return [f"{plan.rewritten_query} {suffix}".strip()] if suffix else [plan.rewritten_query]


def run_adaptive_retrieval(
    question: str,
    *,
    database: LibraryDatabase,
    vector_search: Callable[..., Sequence[Evidence]] | None = None,
    sparse_search: Callable[..., Sequence[Evidence]] | None = None,
    reranker: Callable[[str, Sequence[Evidence]], Sequence[Evidence]] | None = None,
    document_ids: Sequence[str] = (),
    collection_ids: Sequence[str] = (),
    seed_path: Path | None = None,
    conversation_context: Sequence[dict[str, str]] = (),
    planner_client: Any | None = None,
    planner_model: str = "deepseek-chat",
    retrieval_intent: str | None = None,
    allow_planner: bool = True,
    allow_retry: bool = True,
    top_k: int = 8,
    progress: Progress | None = None,
) -> RetrievalResult:
    """Run local-first planning, retrieval, assessment and at most one retry."""
    if progress:
        progress("正在理解问题")
    plan = analyse_query(question, seed_path=seed_path, conversation_context=conversation_context)
    plan = plan.model_copy(
        update={
            "intent": retrieval_intent or plan.intent,
            "document_ids": list(document_ids),
            "collection_ids": list(collection_ids),
        }
    )
    planner_used = False
    if allow_planner and plan.needs_agent_planning and planner_client is not None:
        try:
            if progress:
                progress("正在结合对话上下文规划检索")
            plan = plan_with_deepseek(
                plan,
                conversation_context=conversation_context,
                client=planner_client,
                model=planner_model,
            )
            planner_used = True
        except (ValueError, TypeError, KeyError):
            # Planner is an optional decision aid, never a retrieval dependency.
            pass

    if progress:
        progress("正在按实体、地点、时间和主题检索史料")
    first_round_plans = _first_round_plans(plan)
    strict_candidates = _retrieve_plans(
        first_round_plans,
        database=database,
        vector_search=vector_search,
        sparse_search=sparse_search,
        document_ids=document_ids,
        collection_ids=collection_ids,
    )
    strict_matches = [item for item in strict_candidates if matches_metadata(item, plan)]
    candidates = _metadata_fallback(
        first_round_plans,
        strict_matches,
        database=database,
        vector_search=vector_search,
        sparse_search=sparse_search,
        document_ids=document_ids,
        collection_ids=collection_ids,
    )
    if progress:
        progress(f"正在重排 {len(candidates)} 条候选证据")
    evidence = rerank_candidates(plan.rewritten_query, candidates, reranker=reranker, limit=top_k)
    assessment = assess_evidence(plan, evidence)

    # A compact optional grader is reserved for borderline complex retrieval;
    # simple questions never incur this LLM call.
    if assessment.should_retry and plan.needs_agent_planning and planner_client is not None:
        try:
            graded = grade_with_deepseek(plan, evidence, client=planner_client, model=planner_model)
            if graded is not None:
                assessment = graded
        except (ValueError, TypeError, KeyError):
            pass

    retried = False
    retry_queries: list[str] = []
    if allow_retry and assessment.should_retry:
        retry_queries = _retry_queries(
            plan, assessment.retry_queries, assessment.missing_aspects
        )
        if progress:
            progress("当前证据覆盖不足，正在补充检索一次")
        retry_hits: list[Evidence] = []
        retry_plans = [
            _without_metadata(
                plan.model_copy(
                    update={
                        "rewritten_query": retry_query,
                        "subqueries": [],
                        "needs_multi_query": False,
                    }
                ),
                partial=True,
            )
            for retry_query in retry_queries
        ]
        retry_hits.extend(
            _retrieve_plans(
                retry_plans,
                database=database,
                vector_search=vector_search,
                sparse_search=sparse_search,
                document_ids=document_ids,
                collection_ids=collection_ids,
            )
        )
        candidates = _deduplicate([*candidates, *retry_hits])
        evidence = rerank_candidates(plan.rewritten_query, candidates, reranker=reranker, limit=top_k)
        assessment = assess_evidence(plan, evidence)
        retried = True

    return RetrievalResult(
        query_plan=plan,
        evidence=evidence,
        assessment=assessment,
        retried=retried,
        retry_queries=retry_queries,
        planner_used=planner_used,
        metadata_filters=metadata_values(plan),
        candidate_count=len(candidates),
        reranked_count=len(evidence),
    )
