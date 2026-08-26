"""Download BGE-M3 from ModelScope into a project-local, ignored directory."""

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Download the BGE-M3 embedding model from ModelScope.")


@app.command()
def run(
    output_dir: Path = typer.Option(Path("models/bge-m3"), help="Local directory for model weights."),
) -> None:
    """Fetch the public BAAI/bge-m3 weights without downloading book content."""
    try:
        from modelscope import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency error for users
        raise RuntimeError("Install ModelScope with `pip install modelscope`.") from exc
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    destination = snapshot_download(
        "BAAI/bge-m3",
        local_dir=str(output_dir),
        ignore_patterns=["onnx/*"],
        max_workers=4,
    )
    typer.echo(f"BGE-M3 downloaded to: {destination}")


if __name__ == "__main__":
    app()
