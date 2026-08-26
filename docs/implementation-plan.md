# Byzantine local research Agent implementation status

## Completed MVP foundation

- [x] Local data directory via `platformdirs`, with `BYZANTINE_DATA_DIR` for development and tests.
- [x] Idempotent SQLite schema with automatic `starter` and `personal` collections.
- [x] Canonical `Evidence` and `SourceRegion` model shared by retrieval, generation, citations, research records and the reader.
- [x] PDF/TXT/Markdown/image import workflow, source-file hash detection, metadata capture, status/error recording and document-specific storage.
- [x] PyMuPDF PDF text blocks and coordinate capture; optional PaddleOCR adapter for images and scans.
- [x] SQLite FTS5 keyword retrieval, BGE-M3/Qdrant vector adapter, RRF fusion and collection/document scope support.
- [x] Streamlit local interface and DeepSeek grounded answer adapter.
- [x] Evidence reader with PDF bbox overlay, deterministic GB/T 7714 and Chicago Notes references.
- [x] Initial parallel reading, claim ledger, draft audit, counter-evidence and source-profile workflows persisted in SQLite.

## Verification performed

- `pytest -q`: 21 tests, including idempotent SQLite initialization, duplicate detection, document isolation, TXT region persistence, PDF page/bbox persistence, FTS5 and RRF.
- `python -m compileall -q src tests` and `ruff check src tests`.
- Headless Streamlit HTTP startup check (no model download and no DeepSeek request).

## Next quality gates

1. Import two legally held sources and manually verify page-coordinate highlights against the originals.
2. Build a Byzantine question set with supporting and counter-evidence labels; measure recall and citation validity.
3. Add a reviewer-facing UI for changing every claim-evidence relationship and for browsing persisted comparison/audit/contradiction records.
4. Add controlled two-hop retrieval only after the single-hop evidence quality gate passes.
