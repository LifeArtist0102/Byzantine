"""CLI that retrieves evidence and asks DeepSeek for a cited answer."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from byzantine.generation.deepseek import (
    generate_grounded_answer,
    load_deepseek_api_key,
    render_answer,
)
from byzantine.ingestion.pipeline import load_book_config
from byzantine.retrieval.search import search_local_index

app = typer.Typer(
    add_completion=False, help="Answer Byzantine-history questions from retrieved book evidence."
)


@app.command()
def run(
    question: str = typer.Argument(..., help="Question in Chinese or English."),
    config: Path = typer.Option(Path("config/book.yaml"), help="Project book configuration."),
    qdrant_path: Path = typer.Option(Path("data/qdrant"), help="Local Qdrant storage directory."),
    top_k: int = typer.Option(5, min=1, max=12, help="Evidence passages supplied to DeepSeek."),
    person: list[str] = typer.Option([], "--person", help="Filter by a known person; repeatable."),
    place: list[str] = typer.Option([], "--place", help="Filter by a known place; repeatable."),
    topic: list[str] = typer.Option([], "--topic", help="Filter by topic; repeatable."),
    date_start: int | None = typer.Option(None, help="Earliest requested historical year."),
    date_end: int | None = typer.Option(None, help="Latest requested historical year."),
    json_output: bool = typer.Option(False, "--json", help="Print answer and sources as JSON."),
) -> None:
    """Retrieve book evidence, then ask DeepSeek to write a cited answer in Chinese."""
    settings = load_book_config(config)
    embedding = settings["embedding"]
    generation = settings["generation"]
    configured_path = Path(str(embedding.get("local_model_path", "")))
    model_name = str(configured_path) if configured_path.is_dir() else str(embedding["model_name"])
    evidence = search_local_index(
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
    answer = generate_grounded_answer(
        question,
        evidence,
        api_key=load_deepseek_api_key(),
        base_url=str(generation["base_url"]),
        model=str(generation["model"]),
        max_output_tokens=int(generation["max_output_tokens"]),
        temperature=float(generation["temperature"]),
        max_evidence_characters=int(generation["max_evidence_characters"]),
    )
    typer.echo(
        json.dumps(answer, ensure_ascii=False, indent=2) if json_output else render_answer(answer)
    )


if __name__ == "__main__":
    app()
