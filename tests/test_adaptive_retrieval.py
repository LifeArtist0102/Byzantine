from __future__ import annotations

from pathlib import Path

from byzantine.models.document import BibliographicMetadata
from byzantine.models.retrieval import QueryPlan
from byzantine.retrieval.pipeline import run_adaptive_retrieval
from byzantine.retrieval.planner import plan_with_deepseek
from byzantine.retrieval.quality import assess_evidence
from byzantine.retrieval.query_analysis import analyse_query
from byzantine.storage.database import LibraryDatabase


def _database_with_evidence(tmp_path: Path):
    database = LibraryDatabase(tmp_path / "library.db")
    database.initialize()
    document = database.create_document(
        collection_id="personal",
        metadata=BibliographicMetadata(title="Chronicle", author="A"),
        file_path="source.txt",
        file_hash="b" * 64,
        mime_type="text/plain",
    )
    database.save_chunks(
        document.document_id,
        [
            {
                "chunk_id": f"{document.document_id}_chunk_00000",
                "chunk_index": 0,
                "section_path": ["Chapter 1"],
                "text": "In 1204 Constantinople was captured during the Fourth Crusade.",
                "original_text": "In 1204 Constantinople was captured during the Fourth Crusade.",
                "retrieval_text": "Constantinople Fourth Crusade 1204",
                "page_start": 1,
                "page_end": 1,
                "source_regions": [
                    {"region_id": "r", "coordinate_space": "text_characters", "page": 1}
                ],
                "metadata": {
                    "people": [],
                    "places": ["Constantinople"],
                    "topics": ["warfare"],
                    "date_start": 1204,
                    "date_end": 1204,
                },
            }
        ],
    )
    return database, document.document_id, database.document_evidence(document.document_id)[0]


def test_local_query_analysis_recognizes_alias_dates_topics_and_skips_simple_planner():
    plan = analyse_query(
        "1204年君士坦丁堡发生了什么？",
        seed_path=Path("config/entity_seed.yaml"),
    )
    assert plan.date_start == plan.date_end == 1204
    assert plan.places == ["Constantinople"]
    assert not plan.needs_agent_planning

    class MustNotCall:
        def create(self, **_: object):
            raise AssertionError("simple queries must not call the planner")

    client = type("Client", (), {"chat": type("Chat", (), {"completions": MustNotCall()})()})()
    assert plan_with_deepseek(plan, client=client) == plan


def test_multiturn_planner_rewrites_a_pronoun_question_without_exposing_reasoning():
    base = analyse_query(
        "那后来有什么效果？",
        conversation_context=[{"role": "user", "content": "Why did Alexios I reform the army?"}],
    )

    class Completions:
        def create(self, **_: object):
            message = type(
                "Message",
                (),
                {
                    "content": (
                        '{"intent":"process_analysis","rewritten_query":"What were the effects '
                        'of Alexios I military reforms?","people":["Alexios I Komnenos"],'
                        '"subqueries":["Alexios I military reforms effects"],"needs_multi_query":false}'
                    )
                },
            )()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    planned = plan_with_deepseek(base, conversation_context=[{"role": "user", "content": "Alexios"}], client=client)
    assert planned.rewritten_query.startswith("What were the effects")
    assert planned.used_conversation_context


def test_pipeline_passes_metadata_to_vector_search_and_returns_trace(tmp_path):
    database, document_id, evidence = _database_with_evidence(tmp_path)
    received: list[dict[str, object]] = []

    def vector_search(_: str, **kwargs: object):
        received.append(kwargs)
        return [evidence]

    result = run_adaptive_retrieval(
        "1204 Constantinople war",
        database=database,
        vector_search=vector_search,
        sparse_search=vector_search,
        document_ids=[document_id],
        seed_path=Path("config/entity_seed.yaml"),
        allow_retry=False,
    )
    assert result.evidence
    assert result.metadata_filters["date_start"] == 1204
    assert any(call["places"] == ["Constantinople"] for call in received)
    assert any(call["date_start"] == 1204 for call in received)


def test_metadata_filter_relaxes_before_returning_no_results(tmp_path):
    database, document_id, evidence = _database_with_evidence(tmp_path)
    unmatched = evidence.model_copy(
        update={"metadata": {"people": [], "places": [], "topics": []}}
    )
    received: list[dict[str, object]] = []

    def vector_search(_: str, **kwargs: object):
        received.append(kwargs)
        return [unmatched]

    result = run_adaptive_retrieval(
        "1204 Constantinople",
        database=database,
        vector_search=vector_search,
        sparse_search=vector_search,
        document_ids=[document_id],
        seed_path=Path("config/entity_seed.yaml"),
        allow_retry=False,
    )
    assert result.evidence
    assert any(call["places"] == [] for call in received)


def test_quality_distinguishes_empty_from_entity_covered_evidence(tmp_path):
    _, _, evidence = _database_with_evidence(tmp_path)
    plan = QueryPlan(original_query="q", rewritten_query="q", places=["Constantinople"])
    assert not assess_evidence(plan, []).sufficient
    assert assess_evidence(plan, [evidence]).sufficient


def test_pipeline_retries_once_for_a_complex_insufficient_question(tmp_path):
    database, document_id, evidence = _database_with_evidence(tmp_path)
    calls = 0

    def vector_search(_: str, **__: object):
        nonlocal calls
        calls += 1
        return [evidence]

    result = run_adaptive_retrieval(
        "为什么这些变化分别影响了贵族和中央政府？",
        database=database,
        vector_search=vector_search,
        sparse_search=vector_search,
        document_ids=[document_id],
        allow_planner=False,
    )
    assert result.retried
    assert len(result.retry_queries) == 1
    assert calls >= 2
