"""Idempotent SQLite schema and repository for a local research library."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from byzantine.models.document import BibliographicMetadata, DocumentRecord
from byzantine.models.evidence import Evidence


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class LibraryDatabase:
    """SQLite-backed state. All create operations are safe on repeated startup."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        statements = [
            """CREATE TABLE IF NOT EXISTS collections (
                collection_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                collection_type TEXT NOT NULL CHECK(collection_type IN ('starter','personal')),
                description TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(collection_id),
                title TEXT NOT NULL, author TEXT, translator TEXT, publisher TEXT, publication_year INTEGER,
                edition TEXT, language TEXT NOT NULL, source_type TEXT NOT NULL, file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL, mime_type TEXT NOT NULL, page_count INTEGER, status TEXT NOT NULL,
                error_message TEXT, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)",
            """CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                collection_id TEXT NOT NULL REFERENCES collections(collection_id), section_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL, text TEXT NOT NULL, search_text TEXT NOT NULL,
                page_start INTEGER, page_end INTEGER, printed_page_start INTEGER, printed_page_end INTEGER,
                source_regions TEXT NOT NULL, prev_chunk_id TEXT, next_chunk_id TEXT, metadata_json TEXT NOT NULL)""",
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, chunk_index)",
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, document_id UNINDEXED, collection_id UNINDEXED, search_text)",
            """CREATE TABLE IF NOT EXISTS research_topics (
                topic_id TEXT PRIMARY KEY, title TEXT NOT NULL, research_question TEXT, description TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS topic_items (
                item_id TEXT PRIMARY KEY, topic_id TEXT NOT NULL REFERENCES research_topics(topic_id) ON DELETE CASCADE,
                item_type TEXT NOT NULL, question TEXT, answer TEXT, chunk_id TEXT, document_id TEXT,
                evidence_snapshot TEXT NOT NULL DEFAULT '[]', note TEXT, created_at TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY, claim_text TEXT NOT NULL, claim_status TEXT NOT NULL, user_note TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS claim_evidence (
                claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE, evidence_id TEXT NOT NULL,
                relation_type TEXT NOT NULL CHECK(relation_type IN ('support','oppose','qualify','context')),
                strength TEXT, user_note TEXT, evidence_snapshot TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(claim_id, evidence_id, relation_type))""",
            """CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id TEXT PRIMARY KEY, question TEXT NOT NULL, selected_document_ids TEXT NOT NULL,
                dimensions TEXT NOT NULL, comparison_cells TEXT NOT NULL, summary TEXT, created_at TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS audits (
                audit_id TEXT PRIMARY KEY, title TEXT NOT NULL, original_text TEXT NOT NULL,
                sentence_results TEXT NOT NULL, selected_document_ids TEXT NOT NULL, selected_collection_ids TEXT NOT NULL,
                created_at TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS source_profiles (
                document_id TEXT PRIMARY KEY REFERENCES documents(document_id) ON DELETE CASCADE,
                profile_json TEXT NOT NULL, review_status TEXT NOT NULL, updated_at TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS contradictions (
                contradiction_id TEXT PRIMARY KEY, subject TEXT NOT NULL, description TEXT NOT NULL,
                classification TEXT NOT NULL, evidence_side_a TEXT NOT NULL, evidence_side_b TEXT NOT NULL,
                agent_explanation TEXT, review_status TEXT NOT NULL, created_at TEXT NOT NULL)""",
        ]
        with self.connect() as connection:
            for statement in statements:
                connection.execute(statement)
            now = utc_now()
            for collection_id, name, kind, description in (
                ("starter", "基础知识库", "starter", "项目所有者导入的合法资料"),
                ("personal", "个人知识库", "personal", "当前用户上传的研究资料"),
            ):
                connection.execute(
                    """INSERT INTO collections(collection_id,name,collection_type,description,created_at,updated_at)
                       VALUES(?,?,?,?,?,?) ON CONFLICT(collection_id) DO NOTHING""",
                    (collection_id, name, kind, description, now, now),
                )

    def collections(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM collections ORDER BY collection_type")]

    def find_duplicate(self, file_hash: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE file_hash=?", (file_hash,)).fetchone()
            return dict(row) if row else None

    def create_document(self, *, collection_id: str, metadata: BibliographicMetadata, file_path: str, file_hash: str, mime_type: str) -> DocumentRecord:
        duplicate = self.find_duplicate(file_hash)
        if duplicate:
            raise ValueError(f"该文件已经导入：{duplicate['title']}（{duplicate['document_id']}）")
        now = utc_now()
        document_id = f"doc_{uuid.uuid4().hex}"
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO documents(document_id,collection_id,title,author,translator,publisher,publication_year,edition,language,source_type,file_path,file_hash,mime_type,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (document_id, collection_id, metadata.title, metadata.author, metadata.translator, metadata.publisher,
                 metadata.publication_year, metadata.edition, metadata.language, metadata.source_type, file_path,
                 file_hash, mime_type, "uploaded", now, now),
            )
        return self.get_document(document_id)

    def get_document(self, document_id: str) -> DocumentRecord:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE document_id=?", (document_id,)).fetchone()
        if not row:
            raise KeyError(f"文献不存在：{document_id}")
        return DocumentRecord(**{**dict(row), "extra": json.loads(row["metadata_json"])})

    def list_documents(self, collection_ids: Sequence[str] = ()) -> list[DocumentRecord]:
        query = "SELECT * FROM documents"
        params: list[Any] = []
        if collection_ids:
            query += f" WHERE collection_id IN ({','.join('?' for _ in collection_ids)})"
            params.extend(collection_ids)
        query += " ORDER BY updated_at DESC"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [DocumentRecord(**{**dict(row), "extra": json.loads(row["metadata_json"])}) for row in rows]

    def update_document(self, document_id: str, **updates: Any) -> None:
        allowed = {"title", "author", "translator", "publisher", "publication_year", "edition", "language", "source_type", "file_path", "page_count", "status", "error_message"}
        changes = {key: value for key, value in updates.items() if key in allowed}
        if not changes:
            return
        changes["updated_at"] = utc_now()
        columns = ", ".join(f"{key}=?" for key in changes)
        with self.connect() as connection:
            connection.execute(f"UPDATE documents SET {columns} WHERE document_id=?", [*changes.values(), document_id])

    def save_chunks(self, document_id: str, chunks: Sequence[dict[str, Any]]) -> None:
        document = self.get_document(document_id)
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks_fts WHERE document_id=?", (document_id,))
            connection.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            for chunk in chunks:
                connection.execute(
                    """INSERT INTO chunks(chunk_id,document_id,collection_id,section_path,chunk_index,text,search_text,page_start,page_end,printed_page_start,printed_page_end,source_regions,prev_chunk_id,next_chunk_id,metadata_json)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (chunk["chunk_id"], document_id, document.collection_id, json.dumps(chunk.get("section_path", []), ensure_ascii=False),
                     chunk["chunk_index"], chunk["text"], chunk.get("search_text", chunk["text"]), chunk.get("page_start"), chunk.get("page_end"),
                     chunk.get("printed_page_start"), chunk.get("printed_page_end"), json.dumps(chunk.get("source_regions", []), ensure_ascii=False),
                     chunk.get("prev_chunk_id"), chunk.get("next_chunk_id"), json.dumps(chunk.get("metadata", {}), ensure_ascii=False)),
                )
                connection.execute("INSERT INTO chunks_fts(chunk_id,document_id,collection_id,search_text) VALUES(?,?,?,?)", (chunk["chunk_id"], document_id, document.collection_id, chunk.get("search_text", chunk["text"])))

    def fts_search(self, query: str, *, document_ids: Sequence[str] = (), collection_ids: Sequence[str] = (), limit: int = 20) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        where, params = ["chunks_fts MATCH ?"], [query]
        if document_ids:
            where.append(f"c.document_id IN ({','.join('?' for _ in document_ids)})")
            params.extend(document_ids)
        if collection_ids:
            where.append(f"c.collection_id IN ({','.join('?' for _ in collection_ids)})")
            params.extend(collection_ids)
        params.append(limit)
        sql = f"""SELECT c.*, d.title,d.author,d.translator,d.edition,d.publisher,d.publication_year,d.language,d.source_type,d.file_path,co.collection_type,
                         bm25(chunks_fts) AS rank
                  FROM chunks_fts JOIN chunks c ON c.chunk_id=chunks_fts.chunk_id
                  JOIN documents d ON d.document_id=c.document_id JOIN collections co ON co.collection_id=c.collection_id
                  WHERE {' AND '.join(where)} ORDER BY rank LIMIT ?"""
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, params)]

    def evidence_from_row(self, row: dict[str, Any]) -> Evidence:
        return Evidence(
            evidence_id=f"ev_{row['chunk_id']}", chunk_id=row["chunk_id"], document_id=row["document_id"], collection_id=row["collection_id"],
            collection_type=row["collection_type"], title=row["title"], author=row["author"], translator=row["translator"], edition=row["edition"],
            publisher=row["publisher"], publication_year=row["publication_year"], language=row["language"], source_type=row["source_type"],
            section_path=json.loads(row["section_path"]), pdf_page_start=row["page_start"], pdf_page_end=row["page_end"],
            printed_page_start=row["printed_page_start"], printed_page_end=row["printed_page_end"], source_regions=json.loads(row["source_regions"]),
            source_file=row["file_path"], text=row["text"], metadata=json.loads(row["metadata_json"]), created_at=utc_now(),
        )

    def document_evidence(self, document_id: str) -> list[Evidence]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT c.*, d.title,d.author,d.translator,d.edition,d.publisher,d.publication_year,d.language,d.source_type,d.file_path,co.collection_type
                   FROM chunks c JOIN documents d ON d.document_id=c.document_id
                   JOIN collections co ON co.collection_id=c.collection_id
                   WHERE c.document_id=? ORDER BY c.chunk_index""",
                (document_id,),
            ).fetchall()
        return [self.evidence_from_row(dict(row)) for row in rows]

    def create_topic(self, title: str, research_question: str = "", description: str = "") -> str:
        topic_id, now = f"topic_{uuid.uuid4().hex}", utc_now()
        with self.connect() as connection:
            connection.execute("INSERT INTO research_topics VALUES(?,?,?,?,?,?)", (topic_id, title, research_question, description, now, now))
        return topic_id

    def list_topics(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM research_topics ORDER BY updated_at DESC")]

    def add_topic_item(self, topic_id: str, item_type: str, *, question: str | None = None, answer: str | None = None, evidence: Sequence[Evidence] = (), note: str | None = None) -> str:
        item_id = f"item_{uuid.uuid4().hex}"
        snapshot = [item.model_dump(mode="json") for item in evidence]
        with self.connect() as connection:
            connection.execute("INSERT INTO topic_items VALUES(?,?,?,?,?,?,?,?,?,?)", (item_id, topic_id, item_type, question, answer, None, None, json.dumps(snapshot, ensure_ascii=False), note, utc_now()))
        return item_id

    def create_claim(self, claim_text: str, claim_status: str = "draft", user_note: str | None = None) -> str:
        claim_id, now = f"claim_{uuid.uuid4().hex}", utc_now()
        with self.connect() as connection:
            connection.execute("INSERT INTO claims VALUES(?,?,?,?,?,?)", (claim_id, claim_text, claim_status, user_note, now, now))
        return claim_id

    def link_claim_evidence(self, claim_id: str, evidence: Evidence, relation_type: str, *, strength: str | None = None, user_note: str | None = None) -> None:
        if relation_type not in {"support", "oppose", "qualify", "context"}:
            raise ValueError("证据关系必须是 support、oppose、qualify 或 context")
        with self.connect() as connection:
            connection.execute("INSERT OR REPLACE INTO claim_evidence VALUES(?,?,?,?,?,?,?)", (claim_id, evidence.evidence_id, relation_type, strength, user_note, evidence.model_dump_json(), utc_now()))

    def save_source_profile(self, document_id: str, profile: dict[str, Any], review_status: str) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO source_profiles VALUES(?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET profile_json=excluded.profile_json, review_status=excluded.review_status, updated_at=excluded.updated_at", (document_id, json.dumps(profile, ensure_ascii=False), review_status, utc_now()))

    def source_profile(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM source_profiles WHERE document_id=?", (document_id,)).fetchone()
        return {"profile": json.loads(row["profile_json"]), "review_status": row["review_status"], "updated_at": row["updated_at"]} if row else None

    def save_comparison(self, comparison: dict[str, Any]) -> str:
        comparison_id = str(comparison.get("comparison_id") or f"comparison_{uuid.uuid4().hex}")
        with self.connect() as connection:
            connection.execute("INSERT OR REPLACE INTO comparisons VALUES(?,?,?,?,?,?,?)", (comparison_id, comparison["question"], json.dumps(comparison["selected_document_ids"]), json.dumps(comparison["dimensions"], ensure_ascii=False), json.dumps(comparison["comparison_cells"], ensure_ascii=False), comparison.get("summary"), comparison.get("created_at") or utc_now()))
        return comparison_id

    def save_audit(self, *, title: str, original_text: str, sentence_results: Sequence[dict[str, Any]], document_ids: Sequence[str] = (), collection_ids: Sequence[str] = ()) -> str:
        audit_id = f"audit_{uuid.uuid4().hex}"
        with self.connect() as connection:
            connection.execute("INSERT INTO audits VALUES(?,?,?,?,?,?,?)", (audit_id, title, original_text, json.dumps(sentence_results, ensure_ascii=False), json.dumps(list(document_ids)), json.dumps(list(collection_ids)), utc_now()))
        return audit_id

    def save_contradiction(self, *, subject: str, description: str, classification: str, evidence_side_a: Evidence, evidence_side_b: Evidence, explanation: str = "") -> str:
        contradiction_id = f"contradiction_{uuid.uuid4().hex}"
        with self.connect() as connection:
            connection.execute("INSERT INTO contradictions VALUES(?,?,?,?,?,?,?,?,?)", (contradiction_id, subject, description, classification, evidence_side_a.model_dump_json(), evidence_side_b.model_dump_json(), explanation, "unreviewed", utc_now()))
        return contradiction_id

    def delete_document(self, document_id: str) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute("SELECT chunk_id FROM chunks WHERE document_id=?", (document_id,)).fetchall()
            # Research records keep evidence snapshots rather than fragile live
            # foreign keys. Delete only entries that explicitly mention this
            # document, leaving unrelated topics and claims intact.
            marker = f"%{document_id}%"
            connection.execute(
                "DELETE FROM topic_items WHERE document_id=? OR evidence_snapshot LIKE ?",
                (document_id, marker),
            )
            connection.execute("DELETE FROM claim_evidence WHERE evidence_snapshot LIKE ?", (marker,))
            connection.execute("DELETE FROM comparisons WHERE selected_document_ids LIKE ?", (marker,))
            connection.execute("DELETE FROM audits WHERE selected_document_ids LIKE ?", (marker,))
            connection.execute(
                "DELETE FROM contradictions WHERE evidence_side_a LIKE ? OR evidence_side_b LIKE ?",
                (marker, marker),
            )
            connection.execute("DELETE FROM chunks_fts WHERE document_id=?", (document_id,))
            connection.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
        return [row["chunk_id"] for row in rows]
