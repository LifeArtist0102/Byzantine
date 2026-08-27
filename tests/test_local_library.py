from __future__ import annotations

import pytest

from byzantine.citations import format_chicago_note, format_gbt7714
from byzantine.models.document import BibliographicMetadata
from byzantine.retrieval.hybrid import reciprocal_rank_fusion
from byzantine.storage.database import LibraryDatabase
from byzantine.workflows.process_document import _extract_pdf, process_document


def _document(database: LibraryDatabase, file_hash: str = "a" * 64):
    return database.create_document(
        collection_id="personal",
        metadata=BibliographicMetadata(title="Anna Komnene", author="Anna Komnene", edition="2nd ed.", language="English", source_type="translation"),
        file_path="C:/library/alexiad.txt",
        file_hash=file_hash,
        mime_type="text/plain",
    )


def _chunk(document_id: str) -> dict[str, object]:
    return {
        "chunk_id": f"{document_id}_chunk_00000",
        "chunk_index": 0,
        "section_path": ["Book I", "Alexios"],
        "text": "Alexios prepared the army at Constantinople.",
        "search_text": "Alexios prepared the army at Constantinople.",
        "page_start": 12,
        "page_end": 12,
        "source_regions": [{"page": 12, "region_id": "region_12_03", "bbox": [82, 315, 506, 468], "coordinate_space": "pdf_points", "page_width": 595, "page_height": 842}],
        "metadata": {"people": ["Alexios I"], "places": ["Constantinople"], "topics": ["warfare"]},
    }


def test_idempotent_library_and_fts_evidence(tmp_path):
    database = LibraryDatabase(tmp_path / "library.db")
    database.initialize()
    database.initialize()
    assert {item["collection_id"] for item in database.collections()} == {"starter", "personal"}

    document = _document(database)
    database.save_chunks(document.document_id, [_chunk(document.document_id)])
    rows = database.fts_search("Alexios")
    assert len(rows) == 1
    evidence = database.evidence_from_row(rows[0])
    assert evidence.document_id == document.document_id
    assert evidence.source_regions[0].bbox == [82.0, 315.0, 506.0, 468.0]
    assert "Anna Komnene" in format_gbt7714(evidence)
    assert "Anna Komnene" in format_chicago_note(evidence)


def test_documents_do_not_overwrite_and_duplicate_is_rejected(tmp_path):
    database = LibraryDatabase(tmp_path / "library.db")
    database.initialize()
    first = _document(database, "1" * 64)
    second = _document(database, "2" * 64)
    database.save_chunks(first.document_id, [_chunk(first.document_id)])
    database.save_chunks(second.document_id, [_chunk(second.document_id)])
    assert len(database.document_evidence(first.document_id)) == 1
    assert len(database.document_evidence(second.document_id)) == 1
    with pytest.raises(ValueError, match="已经导入"):
        _document(database, "1" * 64)


def test_rrf_deduplicates_and_favors_consistent_results():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[:3] == ["b", "a", "c"]


def test_txt_import_preserves_document_scope_and_text_regions(tmp_path, monkeypatch):
    from byzantine.indexing import library_index

    monkeypatch.setenv("BYZANTINE_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setattr(library_index, "upsert_evidence", lambda *args, **kwargs: None)
    source = tmp_path / "chronicle.txt"
    source.write_text("Alexios arrived at Constantinople.\n\nThe army was prepared.", encoding="utf-8")
    database = LibraryDatabase(tmp_path / "app-data" / "library.db")
    document = process_document(source, collection_id="personal", metadata=BibliographicMetadata(title="Chronicle", language="English"), database=database)
    evidence = database.document_evidence(document.document_id)
    assert document.status == "ready"
    assert evidence[0].document_id == document.document_id
    assert evidence[0].source_regions[0].coordinate_space == "text_characters"


def test_pdf_import_persists_page_and_bbox(tmp_path, monkeypatch):
    import fitz

    from byzantine.indexing import library_index

    monkeypatch.setenv("BYZANTINE_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setattr(library_index, "upsert_evidence", lambda *args, **kwargs: None)
    source = tmp_path / "source.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Alexios governed Constantinople.")
    pdf.save(source)
    pdf.close()
    database = LibraryDatabase(tmp_path / "app-data" / "library.db")
    document = process_document(source, collection_id="starter", metadata=BibliographicMetadata(title="Test PDF", language="English"), database=database)
    region = database.document_evidence(document.document_id)[0].source_regions[0]
    assert region.page == 1
    assert region.coordinate_space == "pdf_points"
    assert region.bbox is not None


def test_pdf_layout_blocks_are_aggregated_into_one_evidence_chunk(tmp_path):
    import fitz

    source = tmp_path / "blocks.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    for y, text in ((72, "First paragraph."), (120, "Second paragraph."), (168, "Third paragraph.")):
        page.insert_text((72, y), text)
    pdf.save(source)
    pdf.close()
    chunks, page_count = _extract_pdf(source, "doc_test")
    assert page_count == 1
    assert len(chunks) == 1
    assert len(chunks[0]["source_regions"]) == 3
