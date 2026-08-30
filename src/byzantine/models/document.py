"""Bibliographic and document records."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BibliographicMetadata(BaseModel):
    title: str
    author: str | None = None
    translator: str | None = None
    edition: str | None = None
    publisher: str | None = None
    publication_year: int | None = None
    language: str = "unknown"
    source_type: str = "secondary_study"


class DocumentRecord(BibliographicMetadata):
    document_id: str
    collection_id: str
    file_path: str
    file_hash: str
    mime_type: str
    page_count: int | None = None
    status: str = "uploaded"
    error_message: str | None = None
    created_at: str
    updated_at: str
    extra: dict[str, object] = Field(default_factory=dict)
