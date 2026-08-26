"""Command-line semantic evidence search for the Byzantine handbook."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from byzantine.ingestion.pipeline import load_book_config
from byzantine.retrieval.search import render_evidence, search_local_index

app = typer.Typer(add_completion=False, help="Retrieve cited evidence from the Byzantine Qdrant index.")


@app.command()
def run(
    question: str = typer.Argument(..., help="Question in Chinese or English."),
    config: Path = typer.Option(Path("config/book.yaml"), help="Project book configuration."),
    qdrant_path: Path = typer.Option(Path("data/qdrant"), help="Local Qdrant storage directory."),
    top_k: int = typer.Option(5, min=1, max=20, help="Maximum evidence passages to return."),
    person: list[str] = typer.Option([], "--person", help="Filter by a known person; repeatable."),
    place: list[str] = typer.Option([], "--place", help="Filter by a known place; repeatable."),
    topic: list[str] = typer.Option([], "--topic", help="Filter by topic; repeatable."),
    date_start: int | None = typer.Option(None, help="Earliest requested historical year."),
    date_end: int | None = typer.Option(None, help="Latest requested historical year."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable evidence JSON."),
) -> None:
    """Find source passages matching a question; this command does not generate an LLM answer."""
    settings = load_book_config(config)
    embedding = settings["embedding"]
    configured_path = Path(str(embedding.get("local_model_path", "")))
    model_name = str(configured_path) if configured_path.is_dir() else str(embedding["model_name"])
    result = search_local_index(
        question,
        qdrant_path=str(qdrant_path),
        collection_name=str(embedding["collection_name"]),
        model_name=model_name,
        max_length=int(embedding["max_length"]),
        limit=top_k,
        people=person,
        places=place,
        topics=topic,
        date_start=date_start,
        date_end=date_end,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2) if json_output else render_evidence(result))


if __name__ == "__main__":
    app()
