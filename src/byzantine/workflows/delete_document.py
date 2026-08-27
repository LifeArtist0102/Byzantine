"""Safe deletion of one local-library document and all of its derived records."""

from __future__ import annotations

import shutil

from byzantine.paths import ensure_app_data_dir
from byzantine.storage.database import LibraryDatabase


def delete_document_from_library(document_id: str, *, database: LibraryDatabase) -> None:
    """Remove a document's vectors, database evidence, research links and local copy.

    The original user-selected source outside the app-data directory is never
    targeted. Only ``APP_DATA/documents/<document_id>`` may be removed.
    """
    root = ensure_app_data_dir().resolve()
    document = database.get_document(document_id)
    evidence = database.document_evidence(document_id)
    try:
        from byzantine.indexing.library_index import delete_evidence

        delete_evidence([item.chunk_id for item in evidence], qdrant_path=str(root / "qdrant"))
    except RuntimeError as exc:
        raise RuntimeError(f"无法删除 Qdrant 向量；文献尚未删除：{exc}") from exc

    database.delete_document(document_id)
    documents_root = (root / "documents").resolve()
    target = (documents_root / document.document_id).resolve()
    if target.parent != documents_root:
        raise RuntimeError("拒绝删除异常的文献目录。")
    if target.exists():
        shutil.rmtree(target)
