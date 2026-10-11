import os
import tempfile

# Must be set before privrag.api is imported (it builds a module-level app).
_tmp = tempfile.mkdtemp(prefix="privrag_test_")
os.environ.setdefault("PRIVRAG_DB_PATH", os.path.join(_tmp, "api.db"))
os.environ.setdefault("PRIVRAG_EMBEDDER", "hash")
os.environ.setdefault("PRIVRAG_LLM", "echo")  # unit tests never need a running Ollama
os.environ.setdefault("PRIVRAG_PUBLIC_KB", os.path.join(_tmp, "no_public_kb.jsonl"))

import numpy as np
import pytest

DIM = 384


def unit_vec(seed: int, dim: int = DIM) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def make_record():
    from privrag.store import ChunkRecord

    counter = {"n": 0}

    def _make(text: str, metadata: dict | None = None):
        counter["n"] += 1
        return ChunkRecord(text=text, metadata=metadata or {}, embedding=unit_vec(counter["n"]))

    return _make
