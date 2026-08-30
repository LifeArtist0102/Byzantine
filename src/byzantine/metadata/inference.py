"""Optional, cached LLM metadata for genuinely difficult local chunks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from byzantine.storage.database import LibraryDatabase

PROMPT_VERSION = "metadata-v1"


def needs_llm_inference(chunk: dict[str, Any]) -> bool:
    metadata = dict(chunk.get("metadata", {}))
    candidate = dict(metadata.get("candidate", {}))
    trusted = dict(metadata.get("trusted", {}))
    text = str(chunk.get("original_text", chunk.get("text", "")))
    unresolved = candidate.get("person_candidates", [])
    pronouns = len(re.findall(r"\b(he|she|they|this policy|this event|it)\b", text, re.IGNORECASE))
    return bool(
        len(trusted.get("people", []))
        + len(trusted.get("places", []))
        + len(trusted.get("themes", []))
        < 2
        or unresolved
        or pronouns >= 3
        or not chunk.get("section_path")
        or float(metadata.get("ocr_confidence", 1.0)) < 0.72
    )


def cache_key(document_hash: str, text: str, model_name: str) -> tuple[str, str]:
    chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw = f"{document_hash}|{chunk_hash}|{PROMPT_VERSION}|{model_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), chunk_hash


def infer_metadata_batches(
    chunks: Sequence[dict[str, Any]],
    *,
    database: LibraryDatabase,
    document_hash: str,
    model_name: str,
    client: Any | None = None,
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """Infer metadata only for gated chunks and cache JSON by source identity.

    No request is made when no injected/API client is supplied.  The workflow
    therefore remains entirely local by default.
    """
    output = [dict(chunk) for chunk in chunks]
    pending: list[tuple[int, str, str]] = []
    for index, chunk in enumerate(output):
        if not needs_llm_inference(chunk):
            continue
        key, chunk_hash = cache_key(
            document_hash, str(chunk.get("original_text", chunk.get("text", ""))), model_name
        )
        cached = database.llm_metadata_get(key)
        if cached is not None:
            chunk.setdefault("metadata", {}).setdefault("llm_inference", {}).update(cached)
        elif client is not None:
            pending.append((index, key, chunk_hash))
    for offset in range(0, len(pending), max(1, min(15, batch_size))):
        batch = pending[offset : offset + max(1, min(15, batch_size))]
        payload = [
            {
                "index": index,
                "section_path": output[index].get("section_path", []),
                "text": output[index].get("original_text", output[index].get("text", "")),
            }
            for index, _, _ in batch
        ]
        prompt = (
            "Return only JSON: an array of objects with index, contextual_prefix, resolved_people, "
            "resolved_places, events, themes, source_genre, epistemic_type, stance, confidence. "
            "Do not alter or quote source text.\n" + json.dumps(payload, ensure_ascii=False)
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1800,
        )
        parsed = json.loads((response.choices[0].message.content or "[]").strip())
        by_index = {
            int(item["index"]): item
            for item in parsed
            if isinstance(item, dict) and "index" in item
        }
        for index, key, chunk_hash in batch:
            value = {
                key_: value for key_, value in by_index.get(index, {}).items() if key_ != "index"
            }
            database.llm_metadata_put(
                cache_key=key,
                document_hash=document_hash,
                chunk_hash=chunk_hash,
                prompt_version=PROMPT_VERSION,
                model_name=model_name,
                payload=value,
            )
            output[index].setdefault("metadata", {}).setdefault("llm_inference", {}).update(value)
    return output
