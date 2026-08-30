"""Build retrieval chunks without severing headings, paragraphs, or page provenance."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
PART = re.compile(r"^PART\s+[IVXLC]+\b", re.IGNORECASE)
EXCLUDED = re.compile(
    r"\b(references?|bibliography|further reading|suggested reading|index|list of |contributors|"
    r"abbreviations?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Paragraph:
    text: str
    section_path: list[str]


@dataclass(frozen=True)
class SemanticChunk:
    chunk_id: str
    book_id: str
    section_path: list[str]
    chunk_index: int
    text: str
    page_start: int | None
    page_end: int | None
    prev_chunk_id: str | None
    next_chunk_id: str | None


def estimate_tokens(text: str) -> int:
    """Fast, deterministic token estimate for chunk and prompt budgets.

    It intentionally overestimates mixed CJK/Latin text, preventing an
    oversized child even when the optional model tokenizer is unavailable.
    """
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))
    punctuation = len(re.findall(r"[^\sA-Za-z0-9\u3400-\u9fff]", text))
    return max(1, cjk + words + punctuation // 3)


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?。！？])\s+", clean_text(text))
    return [piece for piece in pieces if piece]


def _bounded_parts(text: str, maximum_tokens: int) -> list[str]:
    if estimate_tokens(text) <= maximum_tokens:
        return [text]
    output, current = [], []
    for sentence in split_sentences(text):
        if current and estimate_tokens(" ".join([*current, sentence])) > maximum_tokens:
            output.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        output.append(" ".join(current))
    def force_token_parts(value: str) -> list[str]:
        # Keep whitespace and CJK characters intact.  `.split()` would leave
        # one enormous CJK sentence unsplit and would alter quoted source text.
        pieces: list[str] = []
        current = ""
        for token in re.findall(r"\s+|[\u3400-\u9fff]|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[^\s]", value):
            candidate = current + token
            if current and estimate_tokens(candidate) > maximum_tokens:
                pieces.append(current.strip())
                current = token.lstrip()
            else:
                current = candidate
        if current.strip():
            pieces.append(current.strip())
        return pieces
    return [
        piece
        for value in output
        for piece in (
            force_token_parts(value) if estimate_tokens(value) > maximum_tokens else [value]
        )
    ]


def build_hierarchical_chunks(
    units: list[dict[str, Any]],
    *,
    document_id: str,
    parent_target_tokens: int = 1200,
    parent_max_tokens: int = 1500,
    child_target_tokens: int = 480,
    child_max_tokens: int = 800,
    overlap_tokens: int = 0,
    semantic_embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[dict[str, Any]]:
    """Build Section -> Parent -> Child records without losing source regions.

    Units are already in reading order and may come from PDF blocks, DOCX
    paragraphs, OCR regions or Markdown. Optional BGE embeddings only decide
    whether adjacent paragraphs merit an earlier boundary; they never rewrite
    source text.
    """
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for unit in units:
        text = clean_text(str(unit.get("original_text", unit.get("text", ""))))
        if text:
            path = tuple(unit.get("section_path") or ["Untitled section"])
            # A single OCR/PDF/DOCX paragraph may exceed a Parent by itself.
            # Split it at sentence/token boundaries while retaining the same
            # source region so a parent can never cross the configured cap.
            for text_part in _bounded_parts(text, parent_max_tokens):
                grouped.setdefault(path, []).append({**unit, "original_text": text_part})
    result: list[dict[str, Any]] = []
    global_index = 0
    effective_child_target = min(child_target_tokens, child_max_tokens)
    for section_number, (path, section_units) in enumerate(grouped.items()):
        section_id = f"{document_id}_section_{section_number:04d}"
        semantic_breaks: set[int] = set()
        if semantic_embedder and len(section_units) > 2:
            try:
                vectors = semantic_embedder([item["original_text"] for item in section_units])
                for index in range(1, len(vectors)):
                    left, right = vectors[index - 1], vectors[index]
                    similarity = sum(a * b for a, b in zip(left, right, strict=False))
                    if similarity < 0.42:
                        semantic_breaks.add(index)
            except (ArithmeticError, TypeError, ValueError):
                # Structure and token boundaries stay correct if BGE is absent.
                semantic_breaks = set()
        parents: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for index, unit in enumerate(section_units):
            current_tokens = estimate_tokens("\n\n".join(item["original_text"] for item in current))
            unit_tokens = estimate_tokens(unit["original_text"])
            if current and (
                current_tokens + unit_tokens > parent_max_tokens
                or (index in semantic_breaks and current_tokens >= 800)
                or current_tokens >= parent_target_tokens
            ):
                parents.append(current)
                current = []
            current.append(unit)
        if current:
            parents.append(current)
        for parent_index, parent_units in enumerate(parents):
            parent_id = f"{section_id}_parent_{parent_index:04d}"
            parent_text = "\n\n".join(item["original_text"] for item in parent_units)
            parent_regions = [
                region for item in parent_units for region in item.get("source_regions", [])
            ]
            parent = {
                "parent_id": parent_id,
                "section_id": section_id,
                "section_path": list(path),
                "parent_index": parent_index,
                "original_text": parent_text,
                "token_count": estimate_tokens(parent_text),
                "page_start": min(
                    (
                        item.get("page_start")
                        for item in parent_units
                        if item.get("page_start") is not None
                    ),
                    default=None,
                ),
                "page_end": max(
                    (
                        item.get("page_end")
                        for item in parent_units
                        if item.get("page_end") is not None
                    ),
                    default=None,
                ),
                "source_regions": parent_regions,
            }
            child_parts: list[dict[str, Any]] = []
            child_buffer: list[dict[str, Any]] = []
            for unit in parent_units:
                for text_part in _bounded_parts(unit["original_text"], child_max_tokens):
                    piece = {**unit, "original_text": text_part}
                    candidate = "\n\n".join(
                        item["original_text"] for item in [*child_buffer, piece]
                    )
                    if child_buffer and estimate_tokens(candidate) > effective_child_target:
                        child_parts.append({"units": child_buffer})
                        child_buffer = []
                    child_buffer.append(piece)
            if child_buffer:
                child_parts.append({"units": child_buffer})
            for child in child_parts:
                child_units = child["units"]
                original_text = "\n\n".join(item["original_text"] for item in child_units)
                if overlap_tokens and result:
                    previous = result[-1]["original_text"].split()[-overlap_tokens:]
                    original_text = " ".join([*previous, original_text])
                result.append(
                    {
                        "chunk_id": f"{document_id}_chunk_{global_index:05d}",
                        "chunk_index": global_index,
                        "section_id": section_id,
                        "parent_id": parent_id,
                        "section_path": list(path),
                        "original_text": original_text,
                        "text": original_text,
                        "retrieval_text": original_text,
                        "page_start": min(
                            (
                                item.get("page_start")
                                for item in child_units
                                if item.get("page_start") is not None
                            ),
                            default=None,
                        ),
                        "page_end": max(
                            (
                                item.get("page_end")
                                for item in child_units
                                if item.get("page_end") is not None
                            ),
                            default=None,
                        ),
                        "source_regions": [
                            region
                            for item in child_units
                            for region in item.get("source_regions", [])
                        ],
                        "metadata": {},
                        "parent": parent,
                    }
                )
                global_index += 1
    for index, chunk in enumerate(result):
        chunk["prev_chunk_id"] = result[index - 1]["chunk_id"] if index else None
        chunk["next_chunk_id"] = result[index + 1]["chunk_id"] if index + 1 < len(result) else None
    return result


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def is_excluded_path(path: list[str]) -> bool:
    return any(EXCLUDED.search(item) for item in path)


def is_probable_author_heading(title: str) -> bool:
    """Identify OCR-split contributor names without discarding title headings."""
    tokens = re.findall(r"[A-Za-z]+", title)
    single_letters = sum(len(token) == 1 for token in tokens)
    return len(tokens) >= 3 and single_letters >= 2


def parse_markdown_paragraphs(markdown: str) -> list[Paragraph]:
    """Collect prose under generic Markdown heading paths.

    `PART` headings are retained as a useful PDF/ebook convention, but are
    not required.  A normal Heading 1--6 hierarchy is enough to produce a
    complete ``section_path``.  This keeps the old handbook import behaviour
    while also supporting DOCX/Docling exports with ordinary headings.
    """
    paragraphs: list[Paragraph] = []
    heading_stack: list[str] = []
    in_excluded_section = False
    pending: list[str] = []

    def emit_pending() -> None:
        text = clean_text(" ".join(pending))
        pending.clear()
        if not text or text.startswith(("<!--", "|")) or in_excluded_section:
            return
        path = heading_stack or ["Untitled section"]
        if not is_excluded_path(path):
            paragraphs.append(Paragraph(text=text, section_path=path.copy()))

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        heading = HEADING.match(line)
        if heading:
            emit_pending()
            level = len(heading.group(1))
            title = clean_text(heading.group(2))
            compact_title = re.sub(r"\s+", "", title).casefold()
            if compact_title == "index" or (title and EXCLUDED.search(title)):
                in_excluded_section = True
                heading_stack = []
            elif title and not (heading_stack and is_probable_author_heading(title)):
                in_excluded_section = False
                if PART.match(title):
                    # Ebooks commonly give PART and Chapter the same Markdown
                    # level; retain the Part rather than replacing it.
                    heading_stack = [title]
                else:
                    heading_stack = [*heading_stack[: max(level - 1, 0)], title]
            continue
        if not line:
            emit_pending()
            continue
        if line.startswith(("<!--", "|", "-")):
            emit_pending()
            continue
        pending.append(line)
    emit_pending()
    return paragraphs


def split_long_paragraph(text: str, max_characters: int) -> list[str]:
    if len(text) <= max_characters:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    output: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_characters:
            output.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        output.append(current)
    # A single unpunctuated string still needs a bounded retrieval unit.
    return [piece for value in output for piece in _hard_split(value, max_characters)]


def _hard_split(text: str, max_characters: int) -> list[str]:
    if len(text) <= max_characters:
        return [text]
    return [
        text[offset : offset + max_characters] for offset in range(0, len(text), max_characters)
    ]


class PageLocator:
    """Monotonic, exact-normalized matching from Markdown prose back to PDF pages."""

    def __init__(self, page_records: Iterable[dict[str, object]]) -> None:
        corpus_parts: list[str] = []
        self.offsets: list[int] = []
        self.page_numbers: list[int] = []
        offset = 0
        for record in page_records:
            normalized = normalize_for_match(str(record.get("text", "")))
            if not normalized:
                continue
            self.offsets.append(offset)
            self.page_numbers.append(int(record["pdf_page"]))
            corpus_parts.append(normalized)
            offset += len(normalized)
        self.corpus = "".join(corpus_parts)
        self.cursor = 0

    def locate(self, text: str) -> tuple[int | None, int | None]:
        normalized = normalize_for_match(text)
        if len(normalized) < 24:
            return None, None
        # Longer anchors avoid matching repeated scholarly phrases; smaller
        # anchors tolerate line-break and footnote differences in Docling.
        position = -1
        for anchor_size in (160, 100, 60):
            anchor = normalized[: min(anchor_size, len(normalized))]
            position = self.corpus.find(anchor, self.cursor)
            if position == -1:
                position = self.corpus.find(anchor)
            if position != -1:
                break
        if position == -1:
            return None, None
        end_position = position + len(normalized)
        self.cursor = position
        start_index = bisect.bisect_right(self.offsets, position) - 1
        end_index = bisect.bisect_right(self.offsets, end_position) - 1
        start_index = max(0, start_index)
        end_index = min(len(self.page_numbers) - 1, max(0, end_index))
        return self.page_numbers[start_index], self.page_numbers[end_index]


def read_page_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_semantic_chunks(
    paragraphs: list[Paragraph],
    *,
    book_id: str,
    page_locator: PageLocator,
    target_characters: int,
    min_characters: int,
    max_characters: int,
    overlap_characters: int,
) -> list[SemanticChunk]:
    drafts: list[dict[str, object]] = []
    current_path: list[str] | None = None
    buffer: list[str] = []

    def emit() -> None:
        nonlocal buffer
        text = clean_text("\n\n".join(buffer))
        if len(text) >= min_characters and current_path:
            drafts.append({"text": text, "section_path": current_path.copy()})
        buffer = []

    for paragraph in paragraphs:
        pieces = split_long_paragraph(paragraph.text, max_characters)
        for piece in pieces:
            if current_path != paragraph.section_path:
                emit()
                current_path = paragraph.section_path.copy()
            buffered = clean_text("\n\n".join(buffer))
            if (
                buffer
                and len(buffered) >= target_characters
                and len(buffered) + len(piece) > target_characters
            ):
                tail = buffered[-overlap_characters:] if overlap_characters else ""
                emit()
                buffer = [tail] if tail else []
            buffer.append(piece)
    emit()

    chunks: list[SemanticChunk] = []
    for index, draft in enumerate(drafts):
        text = str(draft["text"])
        section_path = list(draft["section_path"])
        chunk_id = hashlib.sha256(
            f"{book_id}|{' > '.join(section_path)}|{index}|{text}".encode()
        ).hexdigest()[:20]
        page_start, page_end = page_locator.locate(text)
        chunks.append(
            SemanticChunk(
                chunk_id=f"chunk_{chunk_id}",
                book_id=book_id,
                section_path=section_path,
                chunk_index=index,
                text=text,
                page_start=page_start,
                page_end=page_end,
                prev_chunk_id=None,
                next_chunk_id=None,
            )
        )
    return [
        SemanticChunk(
            **{
                **asdict(chunk),
                "prev_chunk_id": chunks[index - 1].chunk_id if index else None,
                "next_chunk_id": chunks[index + 1].chunk_id if index + 1 < len(chunks) else None,
            }
        )
        for index, chunk in enumerate(chunks)
    ]


def run_chunking(processed_book_dir: Path, config: dict[str, object]) -> dict[str, object]:
    markdown_path = processed_book_dir / "document.md"
    pages_path = processed_book_dir / "pages.jsonl"
    if not markdown_path.is_file() or not pages_path.is_file():
        raise FileNotFoundError(
            "Run byzantine-ingest successfully before running semantic chunking."
        )
    chunking = dict(config.get("chunking", {}))
    paragraphs = parse_markdown_paragraphs(markdown_path.read_text(encoding="utf-8"))
    chunks = build_semantic_chunks(
        paragraphs,
        book_id=str(config["book_id"]),
        page_locator=PageLocator(read_page_records(pages_path)),
        target_characters=int(chunking.get("target_characters", 2400)),
        min_characters=int(chunking.get("min_characters", 700)),
        max_characters=int(chunking.get("max_characters", 3600)),
        overlap_characters=int(chunking.get("overlap_characters", 250)),
    )
    chunks_path = processed_book_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    report = {
        "paragraph_count": len(paragraphs),
        "chunk_count": len(chunks),
        "mapped_chunk_count": sum(chunk.page_start is not None for chunk in chunks),
        "unmapped_chunk_count": sum(chunk.page_start is None for chunk in chunks),
        "chunks_path": str(chunks_path),
    }
    (processed_book_dir / "chunking_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
