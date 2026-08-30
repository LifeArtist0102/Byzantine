"""The canonical evidence object shared by every research feature."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceRegion(BaseModel):
    page: int | None = None
    region_id: str
    bbox: list[float] | None = None
    coordinate_space: str
    page_width: float | None = None
    page_height: float | None = None
    paragraph_index: int | None = None
    character_start: int | None = None
    character_end: int | None = None


class Evidence(BaseModel):
    evidence_id: str
    chunk_id: str
    document_id: str
    collection_id: str
    collection_type: str
    title: str
    author: str | None = None
    translator: str | None = None
    edition: str | None = None
    publisher: str | None = None
    publication_year: int | None = None
    language: str = "unknown"
    source_type: str = "secondary_study"
    section_path: list[str] = Field(default_factory=list)
    pdf_page_start: int | None = None
    pdf_page_end: int | None = None
    printed_page_start: int | None = None
    printed_page_end: int | None = None
    source_regions: list[SourceRegion] = Field(default_factory=list)
    source_file: str
    text: str
    # ``text`` remains the canonical, citable original for legacy snapshots.
    # Retrieval-only expansions live separately and are never citation text.
    original_text: str | None = None
    retrieval_text: str | None = None
    parent_id: str | None = None
    section_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    epistemic_type: str = "direct_record"
    created_at: str
