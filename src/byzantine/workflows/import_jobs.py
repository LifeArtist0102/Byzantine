"""Durable, resumable local import jobs for the Streamlit library screen."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from byzantine.models.document import BibliographicMetadata
from byzantine.storage.database import LibraryDatabase
from byzantine.workflows.process_document import process_document

_LOCK = threading.RLock()
_WORKERS: dict[str, threading.Thread] = {}
_PAUSE_REQUESTS: set[str] = set()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _state_path(root: Path) -> Path:
    return root / "import_jobs.json"


def _read(root: Path) -> dict[str, list[dict[str, Any]]]:
    path = _state_path(root)
    if not path.is_file():
        return {"draft": [], "jobs": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"draft": [], "jobs": []}
    return {"draft": list(payload.get("draft", [])), "jobs": list(payload.get("jobs", []))}


def _write(root: Path, payload: dict[str, list[dict[str, Any]]]) -> None:
    path = _state_path(root)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _mutate(root: Path, action: Any) -> Any:
    with _LOCK:
        state = _read(root)
        result = action(state)
        _write(root, state)
        return result


def draft_items(root: Path) -> list[dict[str, Any]]:
    return _mutate(root, lambda state: list(state["draft"]))


def add_draft_item(root: Path, item: dict[str, Any], content: bytes) -> dict[str, Any]:
    """Persist the browser-uploaded bytes before displaying them in the queue."""
    item = dict(item)
    item["queue_id"] = item.get("queue_id") or f"queued_{uuid.uuid4().hex}"
    staging = root / ".uploads" / "drafts"
    staging.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(item["name"])).suffix.lower()
    staged = staging / f"{item['queue_id']}{suffix}"
    staged.write_bytes(content)
    item["staged_path"] = str(staged)
    item.pop("content", None)

    def append(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        if any(existing.get("file_hash") == item.get("file_hash") for existing in state["draft"]):
            raise ValueError("这份文件已经在待处理队列中。")
        state["draft"].append(item)
        return item

    try:
        return _mutate(root, append)
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def remove_draft_item(root: Path, queue_id: str) -> None:
    def remove(state: dict[str, list[dict[str, Any]]]) -> str | None:
        retained = [item for item in state["draft"] if item.get("queue_id") != queue_id]
        removed = next((item for item in state["draft"] if item.get("queue_id") == queue_id), None)
        state["draft"] = retained
        return str(removed.get("staged_path")) if removed else None

    staged_path = _mutate(root, remove)
    if staged_path:
        Path(staged_path).unlink(missing_ok=True)


def jobs(root: Path) -> list[dict[str, Any]]:
    return _mutate(root, lambda state: list(state["jobs"]))


def job_is_running(job_id: str) -> bool:
    """Return whether this app process still owns a live worker for the job."""
    with _LOCK:
        worker = _WORKERS.get(job_id)
        return bool(worker and worker.is_alive())


def create_job_from_draft(root: Path) -> dict[str, Any]:
    def create(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        if not state["draft"]:
            raise ValueError("待处理队列为空。")
        now = _now()
        job = {
            "job_id": f"import_{uuid.uuid4().hex}",
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "current_stage": "等待开始",
            "items": [{**item, "status": "queued", "progress": 0.0} for item in state["draft"]],
        }
        state["draft"] = []
        state["jobs"].insert(0, job)
        return job

    return _mutate(root, create)


def _find_job(state: dict[str, list[dict[str, Any]]], job_id: str) -> dict[str, Any]:
    job = next((item for item in state["jobs"] if item.get("job_id") == job_id), None)
    if job is None:
        raise KeyError(f"导入任务不存在：{job_id}")
    return job


def _set_item_progress(root: Path, job_id: str, queue_id: str, stage: str, fraction: float) -> None:
    def update(state: dict[str, list[dict[str, Any]]]) -> None:
        job = _find_job(state, job_id)
        item = next(item for item in job["items"] if item["queue_id"] == queue_id)
        item["progress"] = max(0.0, min(1.0, fraction))
        item["stage"] = stage
        job["current_stage"] = f"《{item['title']}》：{stage}"
        job["updated_at"] = _now()

    _mutate(root, update)


def _run_job(root: Path, job_id: str) -> None:
    database = LibraryDatabase(root / "library.db")
    database.initialize()
    try:
        while True:
            def next_item(state: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
                job = _find_job(state, job_id)
                if job["status"] == "pausing" or job_id in _PAUSE_REQUESTS:
                    job["status"] = "paused"
                    job["current_stage"] = "已暂停；已完成文献已保存"
                    job["updated_at"] = _now()
                    return None
                item = next((item for item in job["items"] if item["status"] == "queued"), None)
                if item is None:
                    job["status"] = "completed"
                    job["current_stage"] = "处理完成"
                    job["updated_at"] = _now()
                    return None
                job["status"] = "running"
                item["status"] = "processing"
                item["stage"] = "准备处理"
                job["current_stage"] = f"《{item['title']}》：准备处理"
                job["updated_at"] = _now()
                return dict(item)

            item = _mutate(root, next_item)
            if item is None:
                return
            queue_id = str(item["queue_id"])
            try:
                document = process_document(
                    Path(item["staged_path"]),
                    collection_id=item["collection_id"],
                    metadata=BibliographicMetadata(
                        title=item["title"],
                        author=item.get("author"),
                        edition=item.get("edition"),
                        publisher=item.get("publisher"),
                        publication_year=item.get("publication_year"),
                        language=item["language"],
                        source_type=item["source_type"],
                    ),
                    database=database,
                    seed_path=Path("config/entity_seed.yaml"),
                    progress=lambda stage, fraction, queue_id=queue_id: _set_item_progress(
                        root, job_id, queue_id, stage, fraction
                    ),
                )

                def complete(
                    state: dict[str, list[dict[str, Any]]],
                    *,
                    queue_id: str = queue_id,
                    document_id: str = document.document_id,
                ) -> None:
                    job = _find_job(state, job_id)
                    current = next(value for value in job["items"] if value["queue_id"] == queue_id)
                    current.update(
                        {"status": "completed", "progress": 1.0, "stage": "已完成", "document_id": document_id}
                    )
                    job["updated_at"] = _now()

                _mutate(root, complete)
                Path(item["staged_path"]).unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                def fail(
                    state: dict[str, list[dict[str, Any]]],
                    *,
                    queue_id: str = queue_id,
                    error: str = str(exc),
                ) -> None:
                    job = _find_job(state, job_id)
                    current = next(value for value in job["items"] if value["queue_id"] == queue_id)
                    current.update({"status": "failed", "stage": "处理失败", "error": error})
                    job["updated_at"] = _now()

                _mutate(root, fail)
    finally:
        with _LOCK:
            _WORKERS.pop(job_id, None)
            _PAUSE_REQUESTS.discard(job_id)


def start_or_resume_job(root: Path, job_id: str) -> None:
    """Start once per local application process; completed items are never repeated."""
    with _LOCK:
        active = _WORKERS.get(job_id)
        if active and active.is_alive():
            return

        def resume(state: dict[str, list[dict[str, Any]]]) -> None:
            job = _find_job(state, job_id)
            if job["status"] == "completed":
                return
            # A browser/app restart can leave one item marked as processing even
            # though no worker exists any more. It is safe to redo just that item.
            for item in job["items"]:
                if item.get("status") == "processing":
                    item.update({"status": "queued", "progress": 0.0, "stage": "等待恢复"})
            job["status"] = "queued"
            job["current_stage"] = "等待恢复"
            job["updated_at"] = _now()

        _mutate(root, resume)
        worker = threading.Thread(target=_run_job, args=(root, job_id), daemon=True)
        _WORKERS[job_id] = worker
        worker.start()


def pause_job(root: Path, job_id: str) -> None:
    """Pause after the current document reaches a safe processing boundary."""
    with _LOCK:
        _PAUSE_REQUESTS.add(job_id)

    def pause(state: dict[str, list[dict[str, Any]]]) -> None:
        job = _find_job(state, job_id)
        if job["status"] != "completed":
            job["status"] = "pausing"
            job["current_stage"] = "将在当前文献完成后暂停"
            job["updated_at"] = _now()

    _mutate(root, pause)


def retry_failed_items(root: Path, job_id: str) -> None:
    def retry(state: dict[str, list[dict[str, Any]]]) -> None:
        job = _find_job(state, job_id)
        for item in job["items"]:
            if item["status"] == "failed":
                item.update({"status": "queued", "progress": 0.0, "stage": "等待重试"})
                item.pop("error", None)
        job["status"] = "queued"
        job["current_stage"] = "等待重试"
        job["updated_at"] = _now()

    _mutate(root, retry)
