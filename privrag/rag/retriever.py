import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from privrag.store import DecryptedChunk, IsolationViolation

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str  # "public" or "private:<uid>"

class Retriever:
    def __init__(self, public_embs: Optional[np.ndarray] = None, public_meta: Optional[List[Dict]] = None):
        # We will load the BEIR SciFact dataset into this on Day 4
        self.public_embs = public_embs if public_embs is not None else np.empty((0, 384), dtype=np.float32)
        self.public_meta = public_meta if public_meta is not None else []

    def retrieve(
        self, 
        query_emb: np.ndarray, 
        user_id: str, 
        private_chunks: List[DecryptedChunk], 
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        
        # 1. Isolation Assertion: The final firewall
        for chunk in private_chunks:
            # We assume the Store or Ingestion process embeds the owner in the metadata
            chunk_owner = chunk.metadata.get("owner", user_id)
            if chunk_owner != user_id:
                raise IsolationViolation(f"Security breach: Chunk {chunk.chunk_id} belongs to {chunk_owner}, not {user_id}")

        # 2. Build the temporary request pool
        pool_embs = list(self.public_embs)
        pool_data = [{"id": m["chunk_id"], "text": m["text"], "source": "public"} for m in self.public_meta]

        for pc in private_chunks:
            pool_embs.append(pc.embedding)
            pool_data.append({
                "id": pc.chunk_id, 
                "text": pc.text, 
                "source": f"private:{user_id}"
            })

        if not pool_embs:
            return []

        pool_embs_array = np.vstack(pool_embs)
        
        # 3. Cosine Similarity via Dot Product (Embeddings are already L2-normalized)
        scores = np.dot(pool_embs_array, query_emb)

        # 4. Top-K with deterministic tie-breaking
        k = min(top_k, len(scores))
        
        # argpartition is faster than sorting the whole array
        top_indices = np.argpartition(scores, -k)[-k:]
        
        top_results = []
        for idx in top_indices:
            top_results.append((scores[idx], pool_data[idx]))
            
        # Sort by score descending, tie-break by chunk_id alphabetically
        top_results.sort(key=lambda x: (-x[0], x[1]["id"]))

        # Drop the pool immediately after extraction
        results = [
            RetrievedChunk(
                chunk_id=res[1]["id"],
                text=res[1]["text"],
                score=float(res[0]),
                source=res[1]["source"]
            )
            for res in top_results
        ]
        
        del pool_embs, pool_embs_array, pool_data
        
        return results