"""BGE-M3/Qdrant indexing and searching for canonical Evidence records."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from byzantine.indexing.vector_index import qdrant_id
from byzantine.models.evidence import Evidence

DEFAULT_COLLECTION = "byzantine_library_v1"


def _model_name() -> str:
    configured = os.getenv("BYZANTINE_EMBEDDING_MODEL")
    if configured:
        return configured
    bundled = Path(__file__).resolve().parents[3] / "models" / "bge-m3"
    # Reuse the project-local model before attempting a download into the
    # user's C-drive Hugging Face cache.
    return str(bundled) if bundled.is_dir() else "BAAI/bge-m3"


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('向量索引依赖未安装。请运行：pip install -e ".[rag]"') from exc
    return torch, BGEM3FlagModel, QdrantClient, models


def _load_model() -> Any:
    torch, model_class, _, _ = _dependencies()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model_class(_model_name(), use_fp16=device == "cuda", device=device)


def upsert_evidence(evidence: Sequence[Evidence], *, qdrant_path: str, collection_name: str = DEFAULT_COLLECTION) -> None:
    """Embed one document's evidence and persist it beside full provenance payloads."""
    if not evidence:
        return
    _, _, client_class, models = _dependencies()
    model = _load_model()
    batch_size = max(1, int(os.getenv("BYZANTINE_EMBEDDING_BATCH_SIZE", "8")))
    client = client_class(path=qdrant_path)
    try:
        for start in range(0, len(evidence), batch_size):
            batch = evidence[start:start + batch_size]
            output = model.encode([item.text for item in batch], batch_size=batch_size, max_length=4096, return_dense=True, return_sparse=False, return_colbert_vecs=False)
            if not client.collection_exists(collection_name):
                client.create_collection(collection_name, vectors_config=models.VectorParams(size=len(output["dense_vecs"][0]), distance=models.Distance.COSINE))
                for field in ("document_id", "collection_id", "collection_type", "source_type", "language", "metadata.people", "metadata.places", "metadata.topics"):
                    client.create_payload_index(collection_name, field, models.PayloadSchemaType.KEYWORD)
            client.upsert(collection_name, points=[models.PointStruct(id=qdrant_id(item.chunk_id), vector=vector.tolist(), payload={"evidence": item.model_dump(mode="json"), "chunk_id": item.chunk_id, "document_id": item.document_id, "collection_id": item.collection_id, "collection_type": item.collection_type, "source_type": item.source_type, "language": item.language, "metadata": item.metadata}) for item, vector in zip(batch, output["dense_vecs"], strict=True)], wait=True)
            print(f"已向量化 {min(start + len(batch), len(evidence))}/{len(evidence)} 条证据", flush=True)
    finally:
        client.close()


def search_evidence(query: str, *, qdrant_path: str, document_ids: Sequence[str] = (), collection_ids: Sequence[str] = (), limit: int = 20, collection_name: str = DEFAULT_COLLECTION) -> list[Evidence]:
    """Return vector-ranked canonical Evidence, respecting selected library/document scopes."""
    _, _, client_class, models = _dependencies()
    model = _load_model()
    vector = model.encode([query], batch_size=1, max_length=8192, return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"][0].tolist()
    conditions = []
    if document_ids:
        conditions.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=list(document_ids))))
    if collection_ids:
        conditions.append(models.FieldCondition(key="collection_id", match=models.MatchAny(any=list(collection_ids))))
    client = client_class(path=qdrant_path)
    try:
        if not client.collection_exists(collection_name):
            return []
        points = client.query_points(collection_name, query=vector, query_filter=models.Filter(must=conditions) if conditions else None, limit=limit, with_payload=True, with_vectors=False).points
        return [Evidence.model_validate(point.payload["evidence"]) for point in points]
    finally:
        client.close()


def delete_evidence(chunk_ids: Sequence[str], *, qdrant_path: str, collection_name: str = DEFAULT_COLLECTION) -> None:
    if not chunk_ids:
        return
    _, _, client_class, models = _dependencies()
    client = client_class(path=qdrant_path)
    try:
        if client.collection_exists(collection_name):
            client.delete(collection_name, points_selector=models.PointIdsList(points=[qdrant_id(item) for item in chunk_ids]), wait=True)
    finally:
        client.close()
