"""
End-to-end RAG flow shared by the API, the demo UI and the evaluation scripts.

    ingest : text -> chunk -> embed -> store.add_chunks  (encrypted by the store)
    answer : authenticate -> decrypt user's data -> (private + public) pool
             -> cosine top-k -> prompt -> LLM            (paper Fig. 2, Eq. 1, 25, 26)
"""

from collections.abc import Sequence

import numpy as np

from privrag.rag.generator import LLM, build_prompt
from privrag.rag.ingest import TextChunker
from privrag.rag.retriever import RetrievedChunk, Retriever
from privrag.store import ChunkRecord, DecryptedChunk, UserSession


class RagService:
    def __init__(self, store, embedder, retriever: Retriever, llm: LLM):
        self.store = store
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm

    # ---------------------------------------------------------------- ingest
    def add_texts(
        self, session: UserSession, texts: Sequence[str], metadatas: Sequence[dict] | None = None
    ) -> list[str]:
        """Each text is stored as one chunk (already chunk-sized)."""
        if not texts:
            return []
        metadatas = metadatas or [{} for _ in texts]
        embs = self.embedder.embed(list(texts))
        records = [
            ChunkRecord(text=t, metadata=dict(m), embedding=e)
            for t, m, e in zip(texts, metadatas, embs, strict=True)
        ]
        return self.store.add_chunks(session, records)

    def ingest_document(
        self,
        session: UserSession,
        text: str,
        metadata: dict | None = None,
        chunker: TextChunker | None = None,
    ) -> list[str]:
        chunks = (chunker or TextChunker()).chunk(text)
        return self.add_texts(session, chunks, [dict(metadata or {}) for _ in chunks])

    # ---------------------------------------------------------------- query
    def retrieve(self, session: UserSession, question: str, k: int = 5) -> list[RetrievedChunk]:
        private = self.store.load_user_data(session)
        q = self.embedder.embed_one(question)
        return self.retriever.retrieve(q, session.user_id, private, top_k=k)

    def prompt_for(
        self, session: UserSession, question: str, k: int = 5
    ) -> tuple[list[RetrievedChunk], str]:
        """Everything up to (not including) the LLM call: what the model would be shown."""
        retrieved = self.retrieve(session, question, k)
        return retrieved, build_prompt(question, retrieved)

    def answer(
        self, session: UserSession, question: str, k: int = 5
    ) -> tuple[str, list[RetrievedChunk], str]:
        retrieved, prompt = self.prompt_for(session, question, k)
        return self.llm.generate(prompt), retrieved, prompt


class NaiveSharedRAG:
    """
    The vulnerable baseline from the paper's Figure 1: one shared knowledge
    base, no per-user isolation. Any user's query can surface any user's chunk.
    """

    def __init__(
        self,
        all_chunks: list[DecryptedChunk],
        embedder,
        llm: LLM,
        public_embs: np.ndarray | None = None,
        public_meta: list[dict] | None = None,
    ):
        self.embedder = embedder
        self.llm = llm
        embs = [c.embedding for c in all_chunks]
        self.texts = [c.text for c in all_chunks]
        self.ids = [c.chunk_id for c in all_chunks]
        self.sources = ["private:" + str(c.metadata.get("owner", "?")) for c in all_chunks]
        if public_embs is not None and len(public_embs):
            embs += list(public_embs)
            for m in public_meta or []:
                self.texts.append(m["text"])
                self.ids.append(m["chunk_id"])
                self.sources.append("public")
        self.embs = np.vstack(embs) if embs else np.empty((0, 384), dtype=np.float32)

    def prompt_for(self, question: str, k: int = 5) -> tuple[list[RetrievedChunk], str]:
        """Everything up to (not including) the LLM call."""
        q = self.embedder.embed_one(question)
        scores = self.embs @ q if len(self.embs) else np.array([])
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.ids[i]))[:k]
        retrieved = [
            RetrievedChunk(self.ids[i], self.texts[i], float(scores[i]), self.sources[i])
            for i in order
        ]
        return retrieved, build_prompt(question, retrieved)

    def answer(self, question: str, k: int = 5) -> tuple[str, list[RetrievedChunk], str]:
        retrieved, prompt = self.prompt_for(question, k)
        return self.llm.generate(prompt), retrieved, prompt
