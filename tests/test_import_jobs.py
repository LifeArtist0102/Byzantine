from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

from byzantine.workflows import import_jobs


def _item(title: str = "A History") -> dict[str, object]:
    return {
        "name": "history.txt",
        "file_hash": f"hash-{title}",
        "size": 12,
        "title": title,
        "author": "Historian",
        "publisher": None,
        "edition": None,
        "publication_year": None,
        "collection_id": "personal",
        "language": "English",
        "source_type": "secondary_study",
    }


def test_draft_upload_is_persisted_outside_streamlit_session(tmp_path: Path) -> None:
    saved = import_jobs.add_draft_item(tmp_path, _item(), b"durable text")

    draft = import_jobs.draft_items(tmp_path)

    assert draft == [saved]
    assert Path(saved["staged_path"]).read_bytes() == b"durable text"
    assert "content" not in saved


def test_resume_requeues_interrupted_item_without_repeating_completed_work(
    tmp_path: Path, monkeypatch
) -> None:
    import_jobs.add_draft_item(tmp_path, _item("Completed"), b"one")
    import_jobs.add_draft_item(tmp_path, _item("Interrupted"), b"two")
    job = import_jobs.create_job_from_draft(tmp_path)

    def mark_interrupted(state: dict[str, list[dict[str, Any]]]) -> None:
        stored = next(item for item in state["jobs"] if item["job_id"] == job["job_id"])
        stored["status"] = "running"
        stored["items"][0]["status"] = "completed"
        stored["items"][0]["progress"] = 1.0
        stored["items"][1]["status"] = "processing"
        stored["items"][1]["progress"] = 0.45

    import_jobs._mutate(tmp_path, mark_interrupted)
    processed: list[str] = []
    done = Event()

    class FakeDatabase:
        def __init__(self, _: Path) -> None:
            pass

        def initialize(self) -> None:
            pass

    def fake_process(_: Path, **kwargs):
        processed.append(kwargs["metadata"].title)
        done.set()
        return type("Document", (), {"document_id": "doc_resumed"})()

    monkeypatch.setattr(import_jobs, "LibraryDatabase", FakeDatabase)
    monkeypatch.setattr(import_jobs, "process_document", fake_process)
    import_jobs.start_or_resume_job(tmp_path, job["job_id"])

    assert done.wait(timeout=2)
    worker = import_jobs._WORKERS.get(job["job_id"])
    if worker:
        worker.join(timeout=2)
    recovered = import_jobs.jobs(tmp_path)[0]
    assert processed == ["Interrupted"]
    assert recovered["items"][0]["status"] == "completed"
    assert recovered["items"][1]["status"] == "completed"


def test_worker_persists_progress_and_completed_document(tmp_path: Path, monkeypatch) -> None:
    completed = Event()

    class FakeDatabase:
        def __init__(self, _: Path) -> None:
            pass

        def initialize(self) -> None:
            pass

    def fake_process(path: Path, **kwargs):
        assert path.read_bytes() == b"source text"
        kwargs["progress"]("建立索引", 0.6)
        completed.set()
        return type("Document", (), {"document_id": "doc_saved"})()

    monkeypatch.setattr(import_jobs, "LibraryDatabase", FakeDatabase)
    monkeypatch.setattr(import_jobs, "process_document", fake_process)
    import_jobs.add_draft_item(tmp_path, _item(), b"source text")
    job = import_jobs.create_job_from_draft(tmp_path)

    import_jobs.start_or_resume_job(tmp_path, job["job_id"])

    assert completed.wait(timeout=2)
    worker = import_jobs._WORKERS.get(job["job_id"])
    if worker:
        worker.join(timeout=2)
    stored = import_jobs.jobs(tmp_path)[0]
    item = stored["items"][0]
    assert stored["status"] == "completed"
    assert item["status"] == "completed"
    assert item["document_id"] == "doc_saved"
    assert not Path(item["staged_path"]).exists()
    assert not import_jobs.job_is_running(job["job_id"])
