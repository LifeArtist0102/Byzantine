# Byzantine implementation plan

The project is deliberately staged. A later phase may begin only when the
acceptance checks for the preceding phase pass.

## Phase 0 - Source governance and project baseline

- [x] Create the Python project and preserve the source PDF outside Git.
- [x] Verify page count, searchable-text availability, and representative page layouts.
- [ ] Record the exact edition and the right to process it in `config/book.yaml`.
- [ ] Create a Git repository and make the initial commit.

**Acceptance:** the source is readable, its edition is known, and no source PDF
or API secret can be committed accidentally.

## Phase 1 - Structured ingestion and quality gate

- [x] Create the ingestion command.
- [ ] Create a project virtual environment and install project dependencies.
- [ ] Run Docling conversion to `document.md`.
- [ ] Review the table of contents, three body sections, and the index against the PDF.
- [ ] Review `quality_report.json`; investigate every unexpectedly blank or low-text page.
- [ ] Decide whether front matter, bibliography, contributor notes, and index belong in retrieval.

**Acceptance:** Markdown headings reflect the book hierarchy and every retained
passage can be associated with at least one PDF page.

## Phase 2 - Historical chunks and metadata

- [x] Implement Markdown heading parsing and section-aware chunking.
- [x] Implement 700-3,600-character leaf chunks from consecutive paragraphs.
- [x] Implement parent heading text, preceding/following chunk IDs, PDF page span, and section path.
- [x] Create an initial, reviewable canonical registry for people, places, and topics.
- [x] Implement candidate entities and date spans; retain provenance with every metadata field.
- [ ] Manually expand aliases for emperors, authors, places, dynasties, and dates found in this handbook.

**Acceptance:** 30 sampled chunks preserve their local argument and show accurate
chapter path/page references.

## Phase 3 - Retrieval MVP

- [ ] Start Qdrant and create payload indexes for `section_path`, `date_start`, `date_end`, `persons`, and `places`.
- [ ] Embed chunks with BGE-M3.
- [ ] Add dense plus lexical retrieval, fuse candidates, then rerank the top candidates.
- [ ] Expand selected chunks with adjacent chunks from the same section.
- [ ] Expose a retrieval-only CLI that prints score, section path, page range, and text.

**Acceptance:** a gold set of factual and chronological questions retrieves the
documented supporting passage in the top results.

## Phase 4 - Evidence-constrained answers

- [ ] Implement the historian system prompt and a strict "insufficient evidence" response.
- [ ] Give the generator only retrieved passages, never the complete book.
- [ ] Require machine-readable citations containing source chunk ID, section path, and page range.
- [ ] Validate that every cited ID was retrieved in the current request.
- [ ] Add FastAPI endpoints and a minimal Streamlit chat page.

**Acceptance:** answers have valid, inspectable citations; unsupported questions
are refused rather than answered from model memory.

## Phase 5 - Evaluation before agentic behavior

- [ ] Write 80-120 questions with gold source passages: facts, chronology, causal chains, comparisons, and unanswerable cases.
- [ ] Measure retrieval recall, citation validity, answer faithfulness, and refusal accuracy.
- [ ] Review errors manually and improve parsing/chunking/retrieval before changing prompts.

**Acceptance:** the MVP meets the team's quality threshold on the held-out set.

## Phase 6 - Restricted multi-hop history agent

- [ ] Add a planner that identifies when two evidence searches are necessary.
- [ ] Restrict the workflow to `search_history` and `get_adjacent_context` tools, with a two- or three-hop limit.
- [ ] Add coverage checks between hops and a final citation validator.
- [ ] Log each plan, retrieval hop, evidence set, and final answer for audit.

**Acceptance:** causal questions improve over the MVP without increasing unsupported claims.
