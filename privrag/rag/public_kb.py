"""
Public knowledge base (the "PublicData" of paper Eq. 1).

Stored as plain JSONL (chunk_id, text) and embedded once; the embedding matrix
is cached as .npy next to it. Public data is not secret, so it is not encrypted.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path("data/public_kb.jsonl")


def load_public_kb(
    embedder, path: Path = DEFAULT_PATH, max_docs: int | None = None
) -> tuple[np.ndarray, list[dict]]:
    path = Path(path)
    if not path.exists():
        return np.empty((0, 384), dtype=np.float32), []
    meta: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta.append({"chunk_id": r["chunk_id"], "text": r["text"]})
            if max_docs and len(meta) >= max_docs:
                break
    # The key covers the content, so editing a document invalidates the cache.
    digest = hashlib.sha256("\n".join(m["text"] for m in meta).encode("utf-8")).hexdigest()[:10]
    cache = path.with_suffix(f".{getattr(embedder, 'name', 'emb')}.{len(meta)}.{digest}.npy")
    if cache.exists():
        embs = np.load(cache)
    else:
        embs = embedder.embed([m["text"] for m in meta]).astype(np.float32)
        tmp = cache.with_name(cache.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, embs)
        os.replace(tmp, cache)
    return embs, meta
