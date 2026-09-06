# Historia implementation status

## Implemented local research workflow

- [x] Local data directory via `platformdirs`, with `BYZANTINE_DATA_DIR` for user-controlled storage.
- [x] Idempotent SQLite schema, automatic `starter` / `personal` collections, durable conversations, topics and evidence snapshots.
- [x] PDF, DOCX, TXT, Markdown and image import with source hash detection, bibliographic metadata and document lifecycle tracking.
- [x] PDF page and bbox provenance; DOCX heading hierarchy; Section → Parent → Child chunking with token budgets, sibling links and original/retrieval-text separation.
- [x] Three-tier metadata: deterministic trusted metadata, local candidate metadata, and gated/cached optional DeepSeek inference metadata.
- [x] BGE-M3 dense and sparse vectors in local Qdrant, SQLite FTS5, RRF fusion, optional ColBERT reranking and context expansion under a token budget.
- [x] Adaptive retrieval planner with query rewrite, aliases, multi-query retrieval, one bounded retry and grounded-answer citation validation.
- [x] Streamlit research workspace: scoped Agent chat, topics, parallel reading, contradiction/counter-evidence, bibliographic citations and document management.
- [x] Durable import jobs: staged local source copy, persisted progress, pause/resume/retry and recovery after application restart.

## Verification performed

- Unit and integration coverage for ingestion, hierarchical chunking, metadata, retrieval fusion, citation safety, adaptive retrieval, local-library persistence and resilient import jobs.
- `python -m pytest -q`, `ruff check src tests` and `python -m compileall -q src tests` run before release commits.

## Remaining evaluation work

1. Import legally held real monographs and manually verify page/bbox provenance against originals.
2. Build a Byzantine-history benchmark containing direct, causal, multi-hop and counter-evidence questions; measure retrieval recall, citation validity and abstention quality.
3. Compare BGE-M3-only retrieval, dense+sparse+FTS5 fusion, and optional ColBERT reranking on that benchmark.
4. Assess runtime and memory on CPU/GPU hardware for very large or OCR-heavy documents.
