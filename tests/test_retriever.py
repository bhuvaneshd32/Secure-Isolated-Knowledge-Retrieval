import numpy as np
import pytest

from privrag.rag.retriever import Retriever
from privrag.store import DecryptedChunk, IsolationViolation


def test_retriever_isolation_violation():
    retriever = Retriever()

    # Dummy query embedding (384-d float32)
    query_emb = np.zeros(384, dtype=np.float32)

    valid_chunk = DecryptedChunk(
        chunk_id="chunk_1",
        text="This is my own private data.",
        embedding=np.zeros(384, dtype=np.float32),
        metadata={"owner": "Alice"},
    )

    # A chunk belonging to another user somehow got passed to Alice's retrieval request
    foreign_chunk = DecryptedChunk(
        chunk_id="chunk_2",
        text="This is Bob's highly confidential data.",
        embedding=np.zeros(384, dtype=np.float32),
        metadata={"owner": "Bob"},
    )

    # The retriever must catch the mismatched owner and raise the violation
    with pytest.raises(IsolationViolation):
        retriever.retrieve(
            query_emb=query_emb, user_id="Alice", private_chunks=[valid_chunk, foreign_chunk]
        )


def test_retriever_success_and_scoring():
    retriever = Retriever()

    # A mock scenario where query closely matches chunk_2
    query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)

    chunk_1 = DecryptedChunk(
        chunk_id="chunk_1",
        text="Unrelated data.",
        embedding=np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32),
        metadata={"owner": "Alice"},
    )

    chunk_2 = DecryptedChunk(
        chunk_id="chunk_2",
        text="Highly relevant data.",
        embedding=np.array([0.9, 0.1, 0.0] + [0.0] * 381, dtype=np.float32),
        metadata={"owner": "Alice"},
    )

    results = retriever.retrieve(
        query_emb=query_emb, user_id="Alice", private_chunks=[chunk_1, chunk_2]
    )

    assert len(results) == 2
    # chunk_2 should score higher and be first
    assert results[0].chunk_id == "chunk_2"
    assert results[0].source == "private:Alice"
