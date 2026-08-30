"""CLI for structure-aware chunking."""

from pathlib import Path

import typer

from byzantine.chunking.semantic import run_chunking
from byzantine.ingestion.pipeline import load_book_config

app = typer.Typer(
    add_completion=False, help="Create page-traceable semantic chunks from Docling Markdown."
)


@app.command()
def run(
    processed_book_dir: Path = typer.Argument(
        Path("data/processed/oxford_handbook_byzantine_studies"),
        exists=True,
        file_okay=False,
        help="Directory created by byzantine-ingest.",
    ),
    config: Path = typer.Option(Path("config/book.yaml"), help="Project book configuration."),
) -> None:
    """Write chunks.jsonl and chunking_report.json."""
    report = run_chunking(processed_book_dir, load_book_config(config))
    typer.echo(
        f"Chunks: {report['chunk_count']}; mapped to PDF pages: {report['mapped_chunk_count']}; "
        f"unmapped: {report['unmapped_chunk_count']}"
    )


if __name__ == "__main__":
    app()
