"""Load the synthetic corpus (data/make_dataset.py) into any store."""

import json
from collections import defaultdict
from pathlib import Path

DEFAULT_CORPUS = Path("data/private_corpus.jsonl")
DEFAULT_QUERIES = Path("data/private_queries.jsonl")
DEMO_PASSWORD = "demo-password"


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def seed_store(
    service,
    corpus: list[dict],
    password_for=lambda user_id: DEMO_PASSWORD,
    skip_existing: bool = True,
) -> dict[str, str]:
    """
    Register every corpus user in `service.store` and ingest their chunks
    (embedded + encrypted by the store).

    Returns {store_chunk_id: dataset_chunk_id} so evaluation can compare the
    retrieved ids with the gold labels.
    """
    from privrag.store import AuthenticationError, IntegrityError

    by_user = defaultdict(list)
    for c in corpus:
        by_user[c["user_id"]].append(c)

    id_map: dict[str, str] = {}
    for user_id, chunks in by_user.items():
        pw = password_for(user_id)
        try:
            service.store.register_user(user_id, pw)
        except IntegrityError:
            if not skip_existing:
                raise
        try:
            session = service.store.unlock(user_id, pw)
        except AuthenticationError:
            continue
        existing = service.store.load_user_data(session)
        if existing:  # already seeded: map by canary-bearing text
            by_text = {c["text"]: c["chunk_id"] for c in chunks}
            for e in existing:
                if e.text in by_text:
                    id_map[e.chunk_id] = by_text[e.text]
            continue
        ids = service.add_texts(
            session, [c["text"] for c in chunks], [dict(c.get("metadata", {})) for c in chunks]
        )
        for store_id, c in zip(ids, chunks, strict=True):
            id_map[store_id] = c["chunk_id"]
    return id_map
