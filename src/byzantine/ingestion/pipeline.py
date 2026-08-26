"""Phase 1: turn a source PDF into auditable, RAG-ready source artifacts.

Docling produces a hierarchy-aware Markdown representation.  PyPDF keeps a
page-level text map so every later answer can be traced back to PDF pages.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _safe_book_id(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "source_book"


@dataclass(frozen=True)
class PageRecord:
    pdf_page: int
    extracted_characters: int
    extracted_words: int
    text: str


@dataclass(frozen=True)
class IngestionReport:
    book_id: str
    source_pdf: str
    source_size_bytes: int
    pdf_page_count: int
    title: str
    extracted_characters: int
    extracted_words: int
    blank_pages: list[int]
    low_text_pages: list[int]
    blank_page_ratio: float
    docling_status: str
    created_at_utc: str


def load_book_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def extract_page_records(source_pdf: Path) -> tuple[list[PageRecord], dict[str, str]]:
    reader = PdfReader(str(source_pdf))
    metadata = reader.metadata or {}
    pages: list[PageRecord] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(
            PageRecord(
                pdf_page=number,
                extracted_characters=len(text),
                extracted_words=_word_count(text),
                text=text,
            )
        )
    return pages, {str(key): str(value or "") for key, value in metadata.items()}


def export_docling_markdown(source_pdf: Path, output_path: Path) -> str:
    """Convert the PDF with Docling and return a status string.

    This deliberately fails loudly: an unreadable hierarchy must not silently
    turn into a lower-quality RAG corpus.
    """
    try:
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - user-facing dependency error
        raise RuntimeError(
            "Docling is not installed. Create the project virtual environment and run "
            '`pip install -e ".[dev]"`.'
        ) from exc

    # The default docling-parse backend cannot resolve bundled glyph resources
    # from this Windows workspace path. PyPdfium keeps Docling's structured
    # conversion pipeline while avoiding that backend-specific failure.
    pipeline_options = PdfPipelineOptions()
    # This book has an embedded text layer. OCR would make it far slower and
    # could introduce a second, conflicting transcription of the same page.
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pipeline_options,
            ),
        }
    )
    document = converter.convert(str(source_pdf)).document
    output_path.write_text(document.export_to_markdown(), encoding="utf-8")
    return "ok"


def run_ingestion(
    source_pdf: Path,
    output_root: Path,
    config: dict[str, Any],
    *,
    use_docling: bool = True,
) -> tuple[Path, IngestionReport]:
    if not source_pdf.is_file():
        raise FileNotFoundError(f"Source PDF not found: {source_pdf}")

    book_id = str(config.get("book_id") or _safe_book_id(source_pdf.stem))
    destination = output_root / book_id
    destination.mkdir(parents=True, exist_ok=True)

    pages, metadata = extract_page_records(source_pdf)
    threshold = int(
        config.get("quality_gate", {}).get("min_extracted_characters_per_text_page", 120)
    )
    blank_pages = [page.pdf_page for page in pages if page.extracted_characters == 0]
    low_text_pages = [
        page.pdf_page
        for page in pages
        if 0 < page.extracted_characters < threshold
    ]

    with (destination / "pages.jsonl").open("w", encoding="utf-8") as handle:
        for page in pages:
            handle.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")

    docling_status = "skipped"
    if use_docling:
        docling_status = export_docling_markdown(source_pdf, destination / "document.md")

    report = IngestionReport(
        book_id=book_id,
        source_pdf=str(source_pdf.resolve()),
        source_size_bytes=source_pdf.stat().st_size,
        pdf_page_count=len(pages),
        title=metadata.get("/Title") or str(config.get("title") or source_pdf.stem),
        extracted_characters=sum(page.extracted_characters for page in pages),
        extracted_words=sum(page.extracted_words for page in pages),
        blank_pages=blank_pages,
        low_text_pages=low_text_pages,
        blank_page_ratio=round(len(blank_pages) / len(pages), 5) if pages else 1.0,
        docling_status=docling_status,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    (destination / "quality_report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination, report
