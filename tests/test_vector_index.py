from byzantine.indexing.vector_index import build_payload, qdrant_id


def test_qdrant_ids_are_stable_uuids() -> None:
    assert qdrant_id("chunk_abc") == qdrant_id("chunk_abc")
    assert qdrant_id("chunk_abc") != qdrant_id("chunk_def")


def test_payload_preserves_text_and_metadata() -> None:
    chunk = {
        "chunk_id": "chunk_abc",
        "book_id": "book",
        "section_path": ["PART I", "Chapter"],
        "chunk_index": 1,
        "text": "Evidence text",
        "page_start": 10,
        "page_end": 11,
        "prev_chunk_id": None,
        "next_chunk_id": "chunk_def",
        "metadata": {"people": ["Basil II"]},
    }
    payload = build_payload(chunk)
    assert payload["text"] == "Evidence text"
    assert payload["metadata"]["people"] == ["Basil II"]
