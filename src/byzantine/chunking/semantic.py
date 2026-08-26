"""Build retrieval chunks without severing headings, paragraphs, or page provenance."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

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
    """Collect prose paragraphs under a stable Part -> heading path.

    The supplied handbook uses mostly level-two headings, including chapter
    titles.  Treating every Markdown level as a semantic hierarchy would
    therefore be misleading; this parser explicitly promotes PART headings and
    stores other headings as the current chapter/section label.
    """
    paragraphs: list[Paragraph] = []
    current_part: str | None = None
    current_heading: str | None = None
    in_index = False
    pending: list[str] = []

    def emit_pending() -> None:
        text = clean_text(" ".join(pending))
        pending.clear()
        if not text or text.startswith(("<!--", "|")):
            return
        if in_index or current_part is None or current_heading is None:
            return
        path = [current_part, current_heading]
        if not is_excluded_path(path):
            paragraphs.append(Paragraph(text=text, section_path=path))

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        heading = HEADING.match(line)
        if heading:
            emit_pending()
            title = clean_text(heading.group(2))
            compact_title = re.sub(r"\s+", "", title).casefold()
            if compact_title == "index":
                in_index = True
                current_heading = None
            elif PART.match(title):
                in_index = False
                current_part = title
                current_heading = None
            elif title and not (current_heading and is_probable_author_heading(title)):
                current_heading = title
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
    return [text[offset : offset + max_characters] for offset in range(0, len(text), max_characters)]


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
            if buffer and len(buffered) >= target_characters and len(buffered) + len(piece) > target_characters:
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
        raise FileNotFoundError("Run byzantine-ingest successfully before running semantic chunking.")
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
