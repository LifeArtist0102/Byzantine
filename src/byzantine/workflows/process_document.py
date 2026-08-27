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
            # A PDF page often has dozens of layout-level blocks (lines,
            # footnotes, headers). Vectorising every block produces tens of
            # thousands of tiny vectors for one book. Keep page provenance but
            # combine neighbouring prose blocks into retrieval-sized passages.
            blocks = sorted(page.get_text("blocks"), key=lambda block: (block[1], block[0]))
            buffer: list[str] = []
            regions: list[dict[str, Any]] = []

            def emit(
                buffer: list[str] = buffer,
                regions: list[dict[str, Any]] = regions,
                page_number: int = page_number,
            ) -> None:
                if not buffer:
                    return
                text = "\n\n".join(buffer).strip()
                if text:
                    chunks.append(
                        {
                            "chunk_id": "",
                            "chunk_index": 0,
                            "section_path": [f"PDF page {page_number}"],
                            "text": text,
                            "search_text": text,
                            "page_start": page_number,
                            "page_end": page_number,
                            "source_regions": regions.copy(),
                            "metadata": {},
                        }
                    )
                buffer.clear()
                regions.clear()

            for block_index, block in enumerate(blocks):
                x0, y0, x1, y1, text, *_ = block
                text = str(text).strip()
                if not text:
                    continue
                region = {"page": page_number, "region_id": f"region_{page_number}_{block_index}", "bbox": [x0, y0, x1, y1], "coordinate_space": "pdf_points", "page_width": page.rect.width, "page_height": page.rect.height}
                if buffer and len("\n\n".join(buffer)) + len(text) + 2 > 2200:
                    emit()
                # Very long blocks are exceptional; split only those while
                # retaining the same source rectangle for every resulting part.
                for part in [text[offset:offset + 2600] for offset in range(0, len(text), 2600)]:
                    if buffer and len("\n\n".join(buffer)) + len(part) + 2 > 2200:
                        emit()
                    buffer.append(part)
                    regions.append(region)
                    if len(part) >= 2600:
                        emit()
            emit()
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


def _extract_enrich_save_index(
    *,
    source: Path,
    document: DocumentRecord,
    database: LibraryDatabase,
    root: Path,
    seed_path: Path | None,
) -> DocumentRecord:
    """Run the reusable extraction/indexing tail for new or retried documents."""
    database.update_document(document.document_id, status="extracting", error_message=None)
    if source.suffix.lower() == ".pdf":
        chunks, page_count = _extract_pdf(source, document.document_id)
    elif source.suffix.lower() in {".txt", ".md", ".markdown"}:
        chunks, page_count = _extract_text(source, document.document_id)
    else:
        database.update_document(document.document_id, status="ocr_processing")
        chunks, page_count = _extract_image(source, document.document_id)
    if not chunks:
        raise ValueError("未从文件中提取到可检索文本。")
    database.update_document(document.document_id, status="enriching", page_count=page_count)
    seed = load_seed(seed_path) if seed_path else {}
    chunks = [enrich_chunk(chunk, seed) for chunk in chunks]
    # Remove stale vectors first. This matters when a previous interrupted
    # attempt created more chunks than the current, improved chunker.
    try:
        from byzantine.indexing.library_index import delete_evidence

        delete_evidence([item.chunk_id for item in database.document_evidence(document.document_id)], qdrant_path=str(root / "qdrant"))
    except RuntimeError:
        pass
    database.save_chunks(document.document_id, chunks)
    try:
        from byzantine.indexing.library_index import upsert_evidence

        database.update_document(document.document_id, status="indexing")
        upsert_evidence(database.document_evidence(document.document_id), qdrant_path=str(root / "qdrant"))
    except RuntimeError as exc:
        database.update_document(document.document_id, error_message=f"向量索引未建立：{exc}")
    database.update_document(document.document_id, status="ready", page_count=page_count)
    return database.get_document(document.document_id)


def process_document(source: Path, *, collection_id: str, metadata: BibliographicMetadata, database: LibraryDatabase | None = None, seed_path: Path | None = None) -> DocumentRecord:
    """Synchronously process one allowed upload; failures remain visible in SQLite."""
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("仅支持 PDF、TXT、Markdown、JPG、JPEG、PNG。");
    root = ensure_app_data_dir()
    db = database or LibraryDatabase(root / "library.db")
    db.initialize()
    digest = file_hash(source)
    existing = db.find_duplicate(digest)
    if existing:
        document = db.get_document(str(existing["document_id"]))
        stored_directory = root / "documents" / document.document_id
        stored_directory.mkdir(parents=True, exist_ok=True)
        stored_source = stored_directory / f"source{source.suffix.lower()}"
        if document.status == "ready" and Path(document.file_path).is_file():
            raise ValueError(f"该文件已经导入：{document.title}（{document.document_id}）")
        # A previous interrupted import, or an early-version record missing its
        # file_path, can be repaired safely using the same file hash.
        shutil.copy2(source, stored_source)
        db.update_document(document.document_id, file_path=str(stored_source))
        return reprocess_document(document.document_id, database=db, seed_path=seed_path)
    document = db.create_document(collection_id=collection_id, metadata=metadata, file_path="", file_hash=digest, mime_type=mimetypes.guess_type(source.name)[0] or "application/octet-stream")
    destination = root / "documents" / document.document_id
    destination.mkdir(parents=True, exist_ok=True)
    stored_source = destination / f"source{source.suffix.lower()}"
    shutil.copy2(source, stored_source)
    db.update_document(document.document_id, file_path=str(stored_source))
    try:
        return _extract_enrich_save_index(
            source=stored_source,
            document=document,
            database=db,
            root=root,
            seed_path=seed_path,
        )
    except Exception as exc:
        db.update_document(document.document_id, status="failed", error_message=str(exc))
        raise


def reprocess_document(document_id: str, *, database: LibraryDatabase, seed_path: Path | None = None) -> DocumentRecord:
    """Retry a local document without a new browser upload or a new document ID."""
    document = database.get_document(document_id)
    source = Path(document.file_path)
    if not source.is_file():
        root = ensure_app_data_dir()
        candidates = sorted((root / "documents" / document_id).glob("source.*"))
        if not candidates:
            raise FileNotFoundError("找不到本地原文件，无法重新处理。请重新上传该文献。")
        source = candidates[0]
        database.update_document(document_id, file_path=str(source))
    root = ensure_app_data_dir()
    try:
        return _extract_enrich_save_index(
            source=source,
            document=document,
            database=database,
            root=root,
            seed_path=seed_path,
        )
    except Exception as exc:
        database.update_document(document_id, status="failed", error_message=str(exc))
        raise
