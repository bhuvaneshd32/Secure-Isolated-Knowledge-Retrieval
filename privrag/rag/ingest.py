import hashlib
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from privrag.store import ChunkRecord

def get_text_hash(text: str) -> str:
    """SHA-256 hash for deduplication and caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class DocumentLoader:
    @staticmethod
    def load(filepath: str) -> str:
        ext = Path(filepath).suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(filepath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

class TextChunker:
    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        # A lightweight recursive approach: split by paragraphs first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for p in paragraphs:
            if len(current_chunk) + len(p) <= self.chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Fallback: if a single paragraph is huge, we just take it as-is 
                # (A production version would sub-split on sentences/spaces here)
                current_chunk = p + "\n\n"
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # SHA-256 Deduplication
        unique_chunks = {}
        for c in chunks:
            h = get_text_hash(c)
            if h not in unique_chunks:
                unique_chunks[h] = c
                
        return list(unique_chunks.values())

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = ".cache/embeddings"):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = [None] * len(texts)
        to_compute = []
        to_compute_idx = []

        # 1. Check disk cache first via SHA-256
        for i, text in enumerate(texts):
            thash = get_text_hash(text)
            cpath = self.cache_dir / f"{thash}.npy"
            if cpath.exists():
                embeddings[i] = np.load(cpath)
            else:
                to_compute.append(text)
                to_compute_idx.append(i)

        # 2. Compute missing embeddings in batches of 64
        if to_compute:
            new_embs = self.model.encode(
                to_compute, 
                batch_size=64, 
                normalize_embeddings=True, # Enforces L2-normalization for cosine dot products
                convert_to_numpy=True
            )
            for idx, text, emb in zip(to_compute_idx, to_compute, new_embs):
                # Ensure float32 as defined in the contract
                emb = emb.astype(np.float32)
                embeddings[idx] = emb
                thash = get_text_hash(text)
                np.save(self.cache_dir / f"{thash}.npy", emb)

        return np.stack(embeddings)