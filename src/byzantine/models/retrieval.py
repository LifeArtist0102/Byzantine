"""Structured contracts for adaptive, evidence-grounded retrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field

from byzantine.models.evidence import Evidence


class QueryPlan(BaseModel):
    """A retrieval decision, never a historical answer or chain of thought."""

    original_query: str
    rewritten_query: str
    intent: str = "fact_lookup"
    people: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    date_start: int | None = None
    date_end: int | None = None
    collection_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    needs_multi_query: bool = False
    needs_agent_planning: bool = False
    used_conversation_context: bool = False


class RetrievalAssessment(BaseModel):
    """Local/optional-agent assessment distinct from citation validation."""

    sufficient: bool
    confidence: float = Field(ge=0, le=1)
    missing_aspects: list[str] = Field(default_factory=list)
    covered_aspects: list[str] = Field(default_factory=list)
    retry_queries: list[str] = Field(default_factory=list)
    should_retry: bool = False


class RetrievalResult(BaseModel):
    """Traceable output of the one-shot adaptive retrieval workflow."""

    query_plan: QueryPlan
    evidence: list[Evidence] = Field(default_factory=list)
    assessment: RetrievalAssessment
    retried: bool = False
    retry_queries: list[str] = Field(default_factory=list)
    planner_used: bool = False
    metadata_filters: dict[str, object] = Field(default_factory=dict)
    candidate_count: int = 0
    reranked_count: int = 0
