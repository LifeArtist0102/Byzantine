from __future__ import annotations

from pathlib import Path

from byzantine.chunking.semantic import build_hierarchical_chunks, estimate_tokens
from byzantine.generation.deepseek import build_user_prompt, make_sources
from byzantine.indexing.library_index import colbert_rerank
from byzantine.metadata.enrichment import build_retrieval_text, enrich_chunk
from byzantine.metadata.inference import infer_metadata_batches, needs_llm_inference
from byzantine.models.document import BibliographicMetadata
from byzantine.research.services import classify_difference, parallel_reading
from byzantine.retrieval.hybrid import expand_context, parse_local_query, retrieve_evidence
from byzantine.storage.database import LibraryDatabase

SEED = {
    "people": [{"canonical": "Basil II", "aliases": ["Basil II", "巴西尔二世"]}],
    "places": [{"canonical": "Constantinople", "aliases": ["Constantinople", "君士坦丁堡"]}],
    "events": [{"canonical": "Fourth Crusade", "aliases": ["Fourth Crusade", "第四次十字军东征"]}],
    "topics": {"warfare": ["army", "war"]},
}


def _units() -> list[dict]:
    return [
        {
            "text": "Basil II led the army at Constantinople. " * 18,
            "section_path": ["Part I", "Chapter 1"],
            "page_start": 2,
            "page_end": 2,
            "source_regions": [
                {
                    "region_id": "a",
                    "coordinate_space": "pdf_points",
                    "page": 2,
                    "bbox": [1, 2, 3, 4],
                }
            ],
        },
        {
            "text": "The army remained in the city. " * 18,
            "section_path": ["Part I", "Chapter 1"],
            "page_start": 3,
            "page_end": 3,
            "source_regions": [
                {
                    "region_id": "b",
                    "coordinate_space": "pdf_points",
                    "page": 3,
                    "bbox": [1, 2, 3, 4],
                }
            ],
        },
    ]


def _database(tmp_path: Path) -> tuple[LibraryDatabase, str]:
    database = LibraryDatabase(tmp_path / "library.db")
    database.initialize()
    document = database.create_document(
        collection_id="personal",
        metadata=BibliographicMetadata(title="Chronicle", author="A"),
        file_path="source.txt",
        file_hash="a" * 64,
        mime_type="text/plain",
    )
    return database, document.document_id


def test_section_parent_child_preserves_tokens_pages_and_regions():
    chunks = build_hierarchical_chunks(
        _units(),
        document_id="doc",
        parent_target_tokens=80,
        parent_max_tokens=120,
        child_target_tokens=45,
        child_max_tokens=70,
    )
    assert chunks and all(item["parent_id"] and item["section_id"] for item in chunks)
    assert all(estimate_tokens(item["original_text"]) <= 70 for item in chunks)
    assert chunks[0]["source_regions"][0]["bbox"] == [1, 2, 3, 4]
    assert chunks[0]["next_chunk_id"] is not None


def test_token_limits_hold_for_an_unspaced_cjk_paragraph():
    chunks = build_hierarchical_chunks(
        [
            {
                "text": "史" * 230,
                "section_path": ["章节"],
                "source_regions": [{"region_id": "cjk", "coordinate_space": "text_characters"}],
            }
        ],
        document_id="doc",
        parent_max_tokens=100,
        child_max_tokens=50,
    )
    assert chunks
    assert all(estimate_tokens(item["original_text"]) <= 50 for item in chunks)
    assert all(item["parent"]["token_count"] <= 100 for item in chunks)


def test_original_text_is_separate_from_retrieval_text_and_database_migrates(tmp_path):
    database, document_id = _database(tmp_path)
    chunk = enrich_chunk(
        {
            **build_hierarchical_chunks(
                _units()[:1], document_id=document_id, child_max_tokens=800
            )[0]
        },
        SEED,
    )
    chunk["retrieval_text"] = build_retrieval_text(
        chunk["original_text"],
        bibliographic={"title": "Chronicle", "author": "A"},
        section_path=chunk["section_path"],
        trusted=chunk["metadata"]["trusted"],
        candidates=chunk["metadata"]["candidate"],
    )
    database.save_chunks(document_id, [chunk])
    evidence = database.document_evidence(document_id)[0]
    assert evidence.original_text == evidence.text
    assert "Chronicle" in (evidence.retrieval_text or "")
    assert "Chronicle" not in evidence.original_text


def test_llm_metadata_is_gated_and_cached(tmp_path):
    database, _ = _database(tmp_path)
    chunk = {
        "text": "He changed it.",
        "original_text": "He changed it.",
        "metadata": {"trusted": {}, "candidate": {"person_candidates": ["He"]}},
    }
    assert needs_llm_inference(chunk)

    class Completions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            message = type(
                "Message",
                (),
                {
                    "content": '[{"index":0,"contextual_prefix":"The ruler is Basil II","confidence":0.4}]'
                },
            )()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    first = infer_metadata_batches(
        [chunk], database=database, document_hash="hash", model_name="fake", client=client
    )
    second = infer_metadata_batches(
        [chunk], database=database, document_hash="hash", model_name="fake", client=client
    )
    assert first[0]["metadata"]["llm_inference"]["contextual_prefix"]
    assert second[0]["metadata"]["llm_inference"]["confidence"] == 0.4
    assert completions.calls == 1


def test_alias_query_rrf_dedupe_context_and_colbert_off(tmp_path):
    database, document_id = _database(tmp_path)
    chunks = build_hierarchical_chunks(
        _units(),
        document_id=document_id,
        parent_target_tokens=100,
        parent_max_tokens=150,
        child_target_tokens=45,
        child_max_tokens=70,
    )
    for chunk in chunks:
        enriched = enrich_chunk(chunk, SEED)
        enriched["retrieval_text"] = (
            "Basil II 巴西尔二世 Constantinople 君士坦丁堡 " + enriched["original_text"]
        )
        enriched["search_text"] = enriched["retrieval_text"]
        chunk.update(enriched)
    database.save_chunks(document_id, chunks)
    evidence = database.document_evidence(document_id)
    parsed = parse_local_query("巴西尔二世在君士坦丁堡做了什么？", Path("config/entity_seed.yaml"))
    assert parsed.people or parsed.aliases
    retrieved = retrieve_evidence(
        "Basil II Constantinople",
        database=database,
        vector_search=lambda *_args, **_kwargs: evidence,
        sparse_search=lambda *_args, **_kwargs: evidence,
        top_k=6,
    )
    assert retrieved and len({item.chunk_id for item in retrieved}) == len(retrieved)
    contextual = expand_context(
        retrieved, database=database, question="What process did he follow?", token_budget=250
    )
    assert contextual


def test_context_expansion_updates_citation_pages_and_source_regions(tmp_path):
    database, document_id = _database(tmp_path)
    parent_id = "parent_1"
    chunks = []
    for index, page in enumerate((10, 11, 12)):
        chunks.append(
            {
                "chunk_id": f"{document_id}_chunk_{index:05d}",
                "chunk_index": index,
                "section_path": ["Chapter"],
                "text": f"Child {index} explains the process.",
                "page_start": page,
                "page_end": page,
                "parent_id": parent_id,
                "section_id": "section_1",
                "source_regions": [
                    {"region_id": f"r-{page}", "coordinate_space": "pdf_points", "page": page}
                ],
                "metadata": {},
            }
        )
    database.save_chunks(document_id, chunks)
    middle = database.document_evidence(document_id)[1]
    expanded = expand_context(
        [middle], database=database, question="What was the process?", token_budget=200
    )[0]
    assert expanded.pdf_page_start == 10
    assert expanded.pdf_page_end == 12
    assert {region.region_id for region in expanded.source_regions} == {"r-10", "r-11", "r-12"}


def test_colbert_is_a_safe_no_model_fallback_when_disabled(monkeypatch):
    monkeypatch.delenv("BYZANTINE_ENABLE_COLBERT", raising=False)
    marker = object()
    assert colbert_rerank("Basil II", [marker]) == [marker]


def test_parallel_reading_is_document_separate_and_years_alone_are_not_contradiction(tmp_path):
    database, document_id = _database(tmp_path)
    chunk = enrich_chunk(build_hierarchical_chunks(_units()[:1], document_id=document_id)[0], SEED)
    database.save_chunks(document_id, [chunk])
    left = database.document_evidence(document_id)[0]
    right = left.model_copy(
        update={
            "document_id": "other",
            "metadata": {"people": [], "places": [], "candidate": {}},
            "text": "A different year 1204.",
        }
    )
    comparison = parallel_reading("question", [left, right], ["事件描述"])
    assert {cell["document_id"] for cell in comparison["comparison_cells"]} == {
        document_id,
        "other",
    }
    assert classify_difference(left, right) == "not_comparable"


def test_llm_prompt_never_places_retrieval_metadata_inside_source():
    sources = make_sources(
        {
            "hits": [
                {
                    "section_path": ["Chapter"],
                    "page_start": 1,
                    "page_end": 1,
                    "text": "Original source text.",
                    "original_text": "Original source text.",
                    "metadata": {"llm_inference": {"contextual_prefix": "not evidence"}},
                }
            ]
        },
        max_characters=200,
    )
    prompt = build_user_prompt("question", sources)
    source_block = prompt.split("<SOURCE", 1)[1]
    assert "<RETRIEVAL_METADATA>" in prompt
    assert "not evidence" not in source_block
