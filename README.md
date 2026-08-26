# Byzantine

An evidence-grounded RAG agent for answering questions from *The Oxford Handbook
of Byzantine Studies*. The first deliverable is not a chatbot: it is a verified,
page-traceable source corpus. The model must refuse claims the selected book does
not support.

## Current status

Phase 1 is scaffolded: `byzantine-ingest` converts the supplied PDF to structured
Markdown using Docling, creates a page-level text map with `pypdf`, and writes a
quality report. Retrieval, generation, and the multi-hop agent are deliberately
postponed until that corpus has passed manual review.

## Open in PyCharm

1. Open `D:\下载软件\Byzantine` as the project folder.
2. In PyCharm, create a Python 3.11-3.13 virtual environment in `.venv`.
3. In PyCharm's terminal, install the first-stage dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. Run ingestion from the project root. The source book remains outside the
   project and is never copied into Git:

   ```powershell
   byzantine-ingest "D:\C盘迁移\Desktop\The Oxford Handbook of Byzantine Studies.pdf"
   ```

5. Inspect these generated artifacts before beginning retrieval:

   ```text
   data/processed/oxford_handbook_byzantine_studies/document.md
   data/processed/oxford_handbook_byzantine_studies/pages.jsonl
   data/processed/oxford_handbook_byzantine_studies/quality_report.json
   ```

If Docling setup is temporarily unavailable, run with `--skip-docling` to create
only the page map and quality report. This is a diagnostic mode, not a substitute
for the required structure-preserving conversion.

## Phase 2: semantic chunks

After `document.md` exists, create retrieval-ready chunks. This keeps each chunk
inside a handbook part and heading, groups consecutive prose paragraphs, records
the matching PDF-page span, and preserves adjacent-chunk IDs for later context
expansion:

```powershell
byzantine-chunk
```

Review `data/processed/oxford_handbook_byzantine_studies/chunking_report.json`.
Before indexing, manually inspect a sample of `chunks.jsonl`: page mappings must
be correct and references/index material must remain excluded.

## Phase 2b: historical metadata

Create reviewable metadata without sending book content to an external model:

```powershell
byzantine-enrich
```

The command writes `enriched_chunks.jsonl` with trusted dictionary matches for
people and places, year/range extraction, topic tags, and `person_candidates`
that must be reviewed before being promoted to trusted metadata. Extend the
curated aliases and topic vocabulary in `config/entity_seed.yaml`.

## Phase 3: BGE-M3 vectors and Qdrant

Build the local retrieval index after reviewing metadata:

```powershell
pip install -e ".[rag]"
byzantine-index
```

Each chunk becomes one BGE-M3 dense vector. Qdrant stores that vector beside the
source text, page references, heading path, and complete `metadata` object. The
local database is stored in `data/qdrant/`; use `--recreate` only when you intend
to replace the collection after changing chunks or embedding settings.

## Phase 4: inspect retrieval evidence

Search the indexed handbook with a Chinese or English question. This command
returns source evidence only: score, chapter path, PDF pages, metadata, and an
original-text excerpt. It deliberately does not generate an answer yet.

```powershell
byzantine-search "Why did Basil II strengthen imperial authority?" --top-k 5
byzantine-search "巴西尔二世为何能巩固皇权？" --person "Basil II" --top-k 5
```

Optional filters are repeatable and can be combined with semantic similarity:

```powershell
byzantine-search "imperial warfare" --place Constantinople --topic warfare --date-start 976 --date-end 1025
```

Use `--json` for a machine-readable result. A later answer-generation layer will
receive only these retrieved passages and must cite their chapter and PDF pages.

## Phase 5: DeepSeek grounded answers

Install the optional generation dependency, then copy `.env.example` to `.env`
and set your private `DEEPSEEK_API_KEY`. Do not paste the key into source files
or commit `.env`.

```powershell
pip install -e ".[generation]"
Copy-Item .env.example .env
# Edit .env and set DEEPSEEK_API_KEY=...
byzantine-ask "巴西尔二世为何能巩固皇权？" --person "Basil II"
```

`byzantine-ask` performs retrieval first, labels each supplied passage as
`[S1]`, `[S2]`, and so on, then asks DeepSeek to answer only from those
passages. It rejects empty answers, answers without citations, and answers that
refer to a source label that was never retrieved. The output always includes an
evidence catalogue containing the chapter path and PDF pages behind each label.

## Project layout

```text
Byzantine/
├── config/book.yaml                 # Source and quality-gate settings
├── data/source/                     # Ignored source documents
├── data/processed/                  # Ignored derived corpus artifacts
├── docs/implementation-plan.md      # Stage gates and tasks
├── src/byzantine/
│   ├── ingestion/                   # Phase 1, implemented
│   ├── retrieval/                   # Phase 3
│   ├── generation/                  # Phase 4
│   └── workflows/                   # Phase 6
└── tests/
```

## Non-negotiable answer rules

- Use only passages retrieved from this book.
- Cite each factual historical claim with section and PDF page range.
- State `Insufficient evidence in this book` when evidence is absent.
- Never manufacture dates, quotations, relationships, or citations.

The complete staged task list is in [docs/implementation-plan.md](docs/implementation-plan.md).
