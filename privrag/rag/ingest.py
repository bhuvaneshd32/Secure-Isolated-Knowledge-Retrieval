import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger("privrag.ingest")

EMBED_DIM = 384


def get_text_hash(text: str) -> str:
    """SHA-256 hash for deduplication and caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentLoader:
    @staticmethod
    def load(filepath: str) -> str:
        path = Path(filepath)
        return DocumentLoader.load_bytes(path.read_bytes(), path.suffix)

    @staticmethod
    def load_bytes(data: bytes, ext: str) -> str:
        """Extract text from an in-memory file (nothing touches the disk)."""
        if ext.lower() == ".pdf":
            import io

            from pypdf import PdfReader  # lazy: only needed for PDFs

            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return data.decode("utf-8")


class TextChunker:
    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_long(self, p: str) -> list[str]:
        """Break an oversized paragraph on whitespace into pieces <= chunk_size,
        each starting `overlap` characters before the previous one ended."""
        pieces, start, n = [], 0, len(p)
        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:  # prefer a whitespace boundary in the second half of the window
                cut = p.rfind(" ", start + self.chunk_size // 2, end)
                if cut > start:
                    end = cut
            pieces.append(p[start:end].strip())
            if end >= n:
                break
            start = max(end - self.overlap, start + 1)  # always advances
        return [x for x in pieces if x]

    def chunk(self, text: str) -> list[str]:
        # Paragraphs first; oversized paragraphs (e.g. PDF text without blank lines)
        # are split on whitespace with overlap.
        units: list[str] = []
        for p in text.split("\n\n"):
            units.extend(self._split_long(p) if len(p) > self.chunk_size else [p])

        chunks: list[str] = []
        current = ""
        for u in units:
            if current and len(current) + len(u) > self.chunk_size:
                chunks.append(current.strip())
                current = ""
            current += u + "\n\n"
        if current.strip():
            chunks.append(current.strip())

        # SHA-256 deduplication; empty chunks are dropped
        unique: dict[str, str] = {}
        for c in chunks:
            if c:
                unique.setdefault(get_text_hash(c), c)
        return list(unique.values())


class Embedder:
    """
    Deliberately has NO on-disk cache: embeddings of private chunks and user
    queries must only ever reach disk encrypted (paper §3.2). The public KB has
    its own cache in privrag/rag/public_kb.py.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        embs = self.model.encode(
            list(texts),
            batch_size=64,
            normalize_embeddings=True,  # Enforces L2-normalization for cosine dot products
            convert_to_numpy=True,
        )
        return np.asarray(embs, dtype=np.float32)  # float32 as defined in the contract

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashEmbedder:
    """
    Dependency-free fallback embedder: hashed bag of words / bigrams projected
    to 384-d and L2-normalised. Same interface and output contract
    (float32, unit norm) as `Embedder`. Lexical only, but deterministic and
    fast: used for tests and when sentence-transformers is not installed.
    """

    name = "hash"

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = re.findall(r"[a-z0-9]+", text.lower())
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:], strict=False)]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self._vec(text)


class _LazyMiniLM:
    """Defers the model download/load until the first embed() call."""

    name = "minilm"

    def __init__(self):
        self._inner = None

    def embed(self, texts):
        if self._inner is None:
            self._inner = Embedder()
        return self._inner.embed(texts)

    def embed_one(self, text):
        return self.embed([text])[0]


def get_embedder(kind: str | None = None):
    """kind: 'minilm' | 'hash' | 'auto' (default; env PRIVRAG_EMBEDDER)."""
    kind = (kind or os.getenv("PRIVRAG_EMBEDDER", "auto")).lower()
    if kind == "hash":
        return HashEmbedder()
    if kind == "minilm":
        return _LazyMiniLM()
    try:
        import sentence_transformers

        return _LazyMiniLM()
    except ImportError:
        logger.warning("sentence-transformers not installed; using HashEmbedder")
        return HashEmbedder()
