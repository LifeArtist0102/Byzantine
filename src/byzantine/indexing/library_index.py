"""BGE-M3/Qdrant indexing and searching for canonical Evidence records."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from byzantine.indexing.vector_index import qdrant_id
from byzantine.models.evidence import Evidence

DEFAULT_COLLECTION = "byzantine_library_v1"
_QDRANT_LOCK_TIMEOUT_SECONDS = 30.0
_QDRANT_LOCK_POLL_SECONDS = 0.15


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def _local_qdrant_access(qdrant_path: str):
    """Serialize embedded-Qdrant access across local Historia processes."""
    directory = Path(qdrant_path)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".historia-qdrant-access.lock"
    deadline = time.monotonic() + _QDRANT_LOCK_TIMEOUT_SECONDS
    file_descriptor: int | None = None
    while file_descriptor is None:
        try:
            file_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(file_descriptor, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            try:
                owner_pid = int(lock_path.read_text(encoding="ascii"))
            except (OSError, ValueError):
                owner_pid = -1
            if owner_pid > 0 and not _process_is_running(owner_pid):
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "向量索引正在被另一项 Historia 操作使用。请等待当前导入或检索完成后重试。"
                ) from None
            time.sleep(_QDRANT_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _qdrant_client(client_class: Any, qdrant_path: str):
    """Open exactly one local Qdrant client while holding the process lock."""
    with _local_qdrant_access(qdrant_path):
        client = client_class(path=qdrant_path)
        try:
            yield client
        finally:
            client.close()


def _model_name() -> str:
    configured = os.getenv("BYZANTINE_EMBEDDING_MODEL")
    if configured:
        return configured
    bundled = Path(__file__).resolve().parents[3] / "models" / "bge-m3"
    return str(bundled) if bundled.is_dir() else "BAAI/bge-m3"


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('向量索引依赖未安装。请运行：pip install -e ".[rag]"') from exc
    QdrantClient, models = _qdrant_dependencies()
    return torch, BGEM3FlagModel, QdrantClient, models


def _qdrant_dependencies() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('Qdrant 依赖未安装。请运行：pip install -e ".[rag]"') from exc
    return QdrantClient, models


def _load_model() -> Any:
    torch, model_class, _, _ = _dependencies()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model_class(_model_name(), use_fp16=device == "cuda", device=device)


def _payload(item: Evidence) -> dict[str, Any]:
    return {
        "evidence": item.model_dump(mode="json"),
        "chunk_id": item.chunk_id,
        "document_id": item.document_id,
        "collection_id": item.collection_id,
        "collection_type": item.collection_type,
        "source_type": item.source_type,
        "language": item.language,
        "metadata": item.metadata,
    }


def upsert_evidence(
    evidence: Sequence[Evidence],
    *,
    qdrant_path: str,
    collection_name: str = DEFAULT_COLLECTION,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Embed one document's evidence in bounded batches and persist provenance."""
    if not evidence:
        return
    _, _, client_class, models = _dependencies()
    model = _load_model()
    batch_size = max(1, int(os.getenv("BYZANTINE_EMBEDDING_BATCH_SIZE", "8")))
    with _qdrant_client(client_class, qdrant_path) as client:
        for start in range(0, len(evidence), batch_size):
            batch = evidence[start : start + batch_size]
            output = model.encode(
                [item.text for item in batch],
                batch_size=batch_size,
                max_length=4096,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            if not client.collection_exists(collection_name):
                client.create_collection(
                    collection_name,
                    vectors_config=models.VectorParams(
                        size=len(output["dense_vecs"][0]),
                        distance=models.Distance.COSINE,
                    ),
                )
                for field in (
                    "document_id",
                    "collection_id",
                    "collection_type",
                    "source_type",
                    "language",
                    "metadata.people",
                    "metadata.places",
                    "metadata.topics",
                ):
                    client.create_payload_index(
                        collection_name,
                        field,
                        models.PayloadSchemaType.KEYWORD,
                    )
            client.upsert(
                collection_name,
                points=[
                    models.PointStruct(
                        id=qdrant_id(item.chunk_id),
                        vector=vector.tolist(),
                        payload=_payload(item),
                    )
                    for item, vector in zip(batch, output["dense_vecs"], strict=True)
                ],
                wait=True,
            )
            completed = min(start + len(batch), len(evidence))
            print(f"已向量化 {completed}/{len(evidence)} 条证据", flush=True)
            if progress:
                progress(completed, len(evidence))


def search_evidence(
    query: str,
    *,
    qdrant_path: str,
    document_ids: Sequence[str] = (),
    collection_ids: Sequence[str] = (),
    limit: int = 20,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[Evidence]:
    """Return vector-ranked canonical Evidence in the requested scope."""
    _, _, client_class, models = _dependencies()
    model = _load_model()
    vector = model.encode(
        [query],
        batch_size=1,
        max_length=4096,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"][0].tolist()
    conditions = []
    if document_ids:
        conditions.append(
            models.FieldCondition(key="document_id", match=models.MatchAny(any=list(document_ids)))
        )
    if collection_ids:
        conditions.append(
            models.FieldCondition(
                key="collection_id", match=models.MatchAny(any=list(collection_ids))
            )
        )
    with _qdrant_client(client_class, qdrant_path) as client:
        if not client.collection_exists(collection_name):
            return []
        points = client.query_points(
            collection_name,
            query=vector,
            query_filter=models.Filter(must=conditions) if conditions else None,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ).points
        return [Evidence.model_validate(point.payload["evidence"]) for point in points]


def delete_evidence(
    chunk_ids: Sequence[str],
    *,
    qdrant_path: str,
    collection_name: str = DEFAULT_COLLECTION,
) -> None:
    if not chunk_ids:
        return
    _, _, client_class, models = _dependencies()
    with _qdrant_client(client_class, qdrant_path) as client:
        if client.collection_exists(collection_name):
            client.delete(
                collection_name,
                points_selector=models.PointIdsList(points=[qdrant_id(item) for item in chunk_ids]),
                wait=True,
            )


def index_status(*, qdrant_path: str, collection_name: str = DEFAULT_COLLECTION) -> dict[str, Any]:
    """Return a lightweight health report without loading the embedding model."""
    client_class, _ = _qdrant_dependencies()
    with _qdrant_client(client_class, qdrant_path) as client:
        if not client.collection_exists(collection_name):
            return {"healthy": False, "points": 0, "message": "向量集合尚未建立"}
        points = int(client.count(collection_name, exact=True).count)
        return {"healthy": True, "points": points, "message": "向量索引运行正常"}


def rebuild_index(
    evidence: Sequence[Evidence],
    *,
    qdrant_path: str,
    collection_name: str = DEFAULT_COLLECTION,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Replace the collection and rebuild it from canonical SQLite evidence."""
    client_class, _ = _qdrant_dependencies()
    with _qdrant_client(client_class, qdrant_path) as client:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
    upsert_evidence(
        evidence,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        progress=progress,
    )
