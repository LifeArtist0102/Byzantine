"""Deterministic bibliographic footnotes; never delegate citations to an LLM."""

from __future__ import annotations

from byzantine.models.evidence import Evidence


def _pages(evidence: Evidence) -> str:
    if evidence.printed_page_start:
        end = evidence.printed_page_end or evidence.printed_page_start
        return str(evidence.printed_page_start) if end == evidence.printed_page_start else f"{evidence.printed_page_start}-{end}"
    if evidence.pdf_page_start:
        end = evidence.pdf_page_end or evidence.pdf_page_start
        suffix = str(evidence.pdf_page_start) if end == evidence.pdf_page_start else f"{evidence.pdf_page_start}-{end}"
        return f"PDF {suffix}"
    return ""


def format_gbt7714(evidence: Evidence) -> str:
    """Generate a conservative GB/T 7714 reference with graceful missing fields."""
    authors = evidence.author or "佚名"
    parts = [f"{authors}. {evidence.title}"]
    if evidence.translator:
        parts.append(f"{evidence.translator}译")
    publication = ": ".join(part for part in (evidence.publisher, str(evidence.publication_year) if evidence.publication_year else "") if part)
    if publication:
        parts.append(publication)
    if evidence.edition:
        parts.append(evidence.edition)
    if page := _pages(evidence):
        parts.append(f"第{page}页")
    return "[M]. " + ". ".join(parts) + "."


def format_chicago_note(evidence: Evidence) -> str:
    """Generate a Chicago Notes and Bibliography note from structured fields."""
    author = evidence.author or "Anonymous"
    title = f"{evidence.title}"
    publication = ", ".join(part for part in (evidence.publisher, str(evidence.publication_year) if evidence.publication_year else "") if part)
    note = f"{author}, {title}"
    if evidence.translator:
        note += f", trans. {evidence.translator}"
    if evidence.edition:
        note += f", {evidence.edition}"
    if publication:
        note += f" ({publication})"
    if page := _pages(evidence):
        note += f", {page}"
    return note + "."
