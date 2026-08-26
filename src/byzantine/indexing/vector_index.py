"""Create a durable, metadata-filterable dense-vector index in local Qdrant."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


def load_enriched_chunks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def qdrant_id(chunk_id: str) -> str:
    """Return a stable UUID accepted by Qdrant without exposing source text."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"byzantine:{chunk_id}"))


def build_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    """Keep original text and all evidence metadata together with the vector."""
    return {
        "chunk_id": chunk["chunk_id"],
        "book_id": chunk["book_id"],
        "section_path": chunk["section_path"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "prev_chunk_id": chunk["prev_chunk_id"],
        "next_chunk_id": chunk["next_chunk_id"],
        "metadata": chunk["metadata"],
    }


def create_local_index(
    chunks_path: Path,
    qdrant_path: Path,
    *,
    collection_name: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    recreate: bool = False,
) -> dict[str, Any]:
    """Embed source chunks with BGE-M3 and upsert them with payload metadata.

    Qdrant local mode persists the collection below ``qdrant_path``. It uses the
    same query API as a server deployment, so a later Docker/cloud move does
    not require a corpus migration.
    """
    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover - dependency error for users
        raise RuntimeError('Install retrieval dependencies with `pip install -e ".[rag]"`.') from exc

    chunks = load_enriched_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No enriched chunks found in {chunks_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(model_name, use_fp16=device == "cuda", device=device)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    try:
        if recreate and client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        indexed = 0
        vector_size: int | None = None
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            output = model.encode(
                [str(chunk["text"]) for chunk in batch],
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = output["dense_vecs"]
            if vector_size is None:
                vector_size = len(vectors[0])
                if not client.collection_exists(collection_name):
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        ),
                    )
                    for field, schema in (
                        ("book_id", models.PayloadSchemaType.KEYWORD),
                        ("section_path", models.PayloadSchemaType.KEYWORD),
                        ("metadata.people", models.PayloadSchemaType.KEYWORD),
                        ("metadata.places", models.PayloadSchemaType.KEYWORD),
                        ("metadata.topics", models.PayloadSchemaType.KEYWORD),
                        ("metadata.date_start", models.PayloadSchemaType.INTEGER),
                        ("metadata.date_end", models.PayloadSchemaType.INTEGER),
                        ("page_start", models.PayloadSchemaType.INTEGER),
                    ):
                        client.create_payload_index(collection_name, field_name=field, field_schema=schema)
            client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=qdrant_id(str(chunk["chunk_id"])),
                        vector=vector.tolist(),
                        payload=build_payload(chunk),
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ],
                wait=True,
            )
            indexed += len(batch)
            print(f"Indexed {indexed}/{len(chunks)} chunks", flush=True)

        return {
            "collection_name": collection_name,
            "qdrant_path": str(qdrant_path.resolve()),
            "model_name": model_name,
            "device": device,
            "vector_size": vector_size,
            "indexed_chunks": indexed,
        }
    finally:
        client.close()
