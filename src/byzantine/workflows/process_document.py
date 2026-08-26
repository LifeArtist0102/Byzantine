"""End-to-end local import: preserve source, extract provenance, index, and persist."""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
from pathlib import Path
from typing import Any

from byzantine.metadata.enrichment import enrich_chunk, load_seed
from byzantine.models.document import BibliographicMetadata, DocumentRecord
from byzantine.paths import ensure_app_data_dir
from byzantine.storage.database import LibraryDatabase

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".jpg", ".jpeg", ".png"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_chunks(text: str, *, document_id: str, source_regions: list[dict[str, Any]], page: int | None = None) -> list[dict[str, Any]]:
    paragraphs = [piece.strip() for piece in text.replace("\r\n", "\n").split("\n\n") if piece.strip()]
    if not paragraphs:
        return []
    output = []
    for index, paragraph in enumerate(paragraphs):
        for part_index, part in enumerate([paragraph[offset:offset + 2600] for offset in range(0, len(paragraph), 2600)]):
            output.append({"chunk_id": f"{document_id}_chunk_{len(output):05d}", "chunk_index": len(output), "section_path": ["Imported document"], "text": part, "search_text": part, "page_start": page, "page_end": page, "source_regions": source_regions, "metadata": {}})
    for index, chunk in enumerate(output):
        chunk["prev_chunk_id"] = output[index - 1]["chunk_id"] if index else None
        chunk["next_chunk_id"] = output[index + 1]["chunk_id"] if index + 1 < len(output) else None
    return output


def _extract_pdf(path: Path, document_id: str) -> tuple[list[dict[str, Any]], int]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("请安装 PyMuPDF：pip install -e \".[app]\"") from exc
    chunks: list[dict[str, Any]] = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            blocks = page.get_text("blocks")
            for block_index, block in enumerate(blocks):
                x0, y0, x1, y1, text, *_ = block
                text = str(text).strip()
                if not text:
                    continue
                region = {"page": page_number, "region_id": f"region_{page_number}_{block_index}", "bbox": [x0, y0, x1, y1], "coordinate_space": "pdf_points", "page_width": page.rect.width, "page_height": page.rect.height}
                chunks.extend(_text_chunks(text, document_id=document_id, source_regions=[region], page=page_number))
        for index, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{document_id}_chunk_{index:05d}"
            chunk["chunk_index"] = index
            chunk["prev_chunk_id"] = chunks[index - 1]["chunk_id"] if index else None
            chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else None
        return chunks, len(pdf)


def _extract_text(path: Path, document_id: str) -> tuple[list[dict[str, Any]], int | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    regions = [{"region_id": "paragraph_source", "coordinate_space": "text_characters", "paragraph_index": 0, "character_start": 0, "character_end": len(text)}]
    return _text_chunks(text, document_id=document_id, source_regions=regions), None


def _extract_image(path: Path, document_id: str) -> tuple[list[dict[str, Any]], int | None]:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("图片或扫描件需要 OCR。请安装：pip install -e \".[ocr]\"") from exc
    result = PaddleOCR(use_angle_cls=True, lang="en").ocr(str(path), cls=True)
    chunks: list[dict[str, Any]] = []
    for item in result[0] if result else []:
        polygon, (text, confidence) = item
        xs, ys = [point[0] for point in polygon], [point[1] for point in polygon]
        region = {"region_id": f"ocr_{len(chunks):04d}", "coordinate_space": "image_pixels", "bbox": [min(xs), min(ys), max(xs), max(ys)]}
        chunks.extend(_text_chunks(text, document_id=document_id, source_regions=[region]))
        chunks[-1]["metadata"] = {"ocr_raw": text, "corrected_text": text, "ocr_confidence": confidence}
    return chunks, 1


def process_document(source: Path, *, collection_id: str, metadata: BibliographicMetadata, database: LibraryDatabase | None = None, seed_path: Path | None = None) -> DocumentRecord:
    """Synchronously process one allowed upload; failures remain visible in SQLite."""
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("仅支持 PDF、TXT、Markdown、JPG、JPEG、PNG。");
    root = ensure_app_data_dir()
    db = database or LibraryDatabase(root / "library.db")
    db.initialize()
    digest = file_hash(source)
    document = db.create_document(collection_id=collection_id, metadata=metadata, file_path="", file_hash=digest, mime_type=mimetypes.guess_type(source.name)[0] or "application/octet-stream")
    destination = root / "documents" / document.document_id
    destination.mkdir(parents=True, exist_ok=True)
    stored_source = destination / f"source{source.suffix.lower()}"
    shutil.copy2(source, stored_source)
    db.update_document(document.document_id, file_path=str(stored_source), status="extracting")
    try:
        if source.suffix.lower() == ".pdf":
            chunks, page_count = _extract_pdf(stored_source, document.document_id)
        elif source.suffix.lower() in {".txt", ".md", ".markdown"}:
            chunks, page_count = _extract_text(stored_source, document.document_id)
        else:
            db.update_document(document.document_id, status="ocr_processing")
            chunks, page_count = _extract_image(stored_source, document.document_id)
        if not chunks:
            raise ValueError("未从文件中提取到可检索文本。")
        db.update_document(document.document_id, status="enriching", page_count=page_count)
        seed = load_seed(seed_path) if seed_path else {}
        chunks = [enrich_chunk(chunk, seed) for chunk in chunks]
        db.save_chunks(document.document_id, chunks)
        # FTS5 is always available. When the optional RAG dependency is installed,
        # BGE-M3 and Qdrant receive the identical Evidence payload.
        try:
            from byzantine.indexing.library_index import upsert_evidence

            db.update_document(document.document_id, status="indexing")
            upsert_evidence(db.document_evidence(document.document_id), qdrant_path=str(root / "qdrant"))
        except RuntimeError as exc:
            # A text-only library remains usable; the reason is recorded rather
            # than pretending that dense indexing completed.
            db.update_document(document.document_id, error_message=f"向量索引未建立：{exc}")
        db.update_document(document.document_id, status="ready", page_count=page_count)
    except Exception as exc:
        db.update_document(document.document_id, status="failed", error_message=str(exc))
        raise
    return db.get_document(document.document_id)
