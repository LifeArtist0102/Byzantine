"""Command-line interface for the source-book ingestion step."""

from pathlib import Path

import typer

from byzantine.ingestion.pipeline import load_book_config, run_ingestion

app = typer.Typer(add_completion=False, help="Parse and quality-check a Byzantine history source PDF.")


@app.command()
def run(
    source_pdf: Path = typer.Argument(..., exists=True, readable=True, help="Path to the source PDF."),
    config: Path = typer.Option(Path("config/book.yaml"), help="Project book configuration."),
    output_root: Path = typer.Option(Path("data/processed"), help="Directory for derived artifacts."),
    skip_docling: bool = typer.Option(False, help="Only create the page map and quality report."),
) -> None:
    """Create document.md, pages.jsonl and quality_report.json."""
    settings = load_book_config(config)
    destination, report = run_ingestion(
        source_pdf=source_pdf,
        output_root=output_root,
        config=settings,
        use_docling=not skip_docling,
    )
    typer.echo(f"Created source artifacts in: {destination}")
    typer.echo(f"Pages: {report.pdf_page_count}; extracted words: {report.extracted_words:,}")
    typer.echo(f"Blank pages: {len(report.blank_pages)}; Docling: {report.docling_status}")


if __name__ == "__main__":
    app()
