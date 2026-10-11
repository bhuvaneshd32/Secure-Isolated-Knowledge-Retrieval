"""
Quality parity (paper claim: encryption preserves RAG functionality).

Runs the same corpus and gold-labelled queries through every store with the
same real embedder, computes Recall/MRR/nDCG@10, and ASSERTS that every store
returns exactly the same ranked chunks as the plaintext baseline.

    python -m evaluation.quality
"""

import csv
import math
from pathlib import Path

from privrag.rag.generator import get_llm
from privrag.rag.ingest import get_embedder
from privrag.rag.pipeline import RagService
from privrag.rag.public_kb import load_public_kb
from privrag.rag.retriever import Retriever
from privrag.rag.seed import DEMO_PASSWORD, load_jsonl, seed_store
from privrag.store.method_a import MethodAStore
from privrag.store.method_b import MethodBStore
from privrag.store.plaintext import PlaintextStore


def calc_metrics(retrieved_ids, gold_ids, k=10):
    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(gold_ids))
    recall = hits / len(gold_ids) if gold_ids else 0.0
    mrr = 0.0
    for i, rid in enumerate(retrieved_k):
        if rid in gold_ids:
            mrr = 1.0 / (i + 1)
            break
    dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(retrieved_k) if rid in gold_ids)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return recall, mrr, (dcg / idcg if idcg > 0 else 0.0)


def run_quality_parity(
    use_public_kb: bool = True, out_path: Path = Path("evaluation/results/parity_results.csv")
):
    corpus = load_jsonl("data/private_corpus.jsonl")
    queries = load_jsonl("data/private_queries.jsonl")

    emb = get_embedder()
    pub_embs, pub_meta = load_public_kb(emb) if use_public_kb else load_public_kb(emb, "none.jsonl")

    stores = {
        "plaintext": PlaintextStore(":memory:"),
        "method_a": MethodAStore(":memory:", kdf_n=2**12),
        "method_b_paper": MethodBStore(":memory:", mode="paper", kdf_n=2**12),
        "method_b_hardened": MethodBStore(":memory:", mode="hardened", kdf_n=2**12),
    }

    results = []
    reference = {}  # query_id -> ranked dataset ids (plaintext)
    for name, store in stores.items():
        svc = RagService(store, emb, Retriever(pub_embs, pub_meta), get_llm())
        id_map = seed_store(svc, corpus)
        id_map.update(
            {m["chunk_id"]: m["chunk_id"] for m in pub_meta}
        )  # public ids map to themselves

        metrics = []
        for q in queries:
            session = store.unlock(q["user_id"], DEMO_PASSWORD)
            retrieved = svc.retrieve(session, q["question"], k=10)
            ranked = [id_map[r.chunk_id] for r in retrieved]

            if name == "plaintext":
                reference[q["query_id"]] = ranked
            else:  # core parity assertion
                assert ranked == reference[q["query_id"]], (
                    f"Parity failed for {name} on {q['query_id']}: "
                    f"{ranked} != {reference[q['query_id']]}"
                )
            metrics.append(calc_metrics(ranked, q["gold_chunk_ids"]))

        n = len(metrics)
        results.append(
            {
                "Store": name,
                "Recall@10": f"{sum(m[0] for m in metrics) / n:.4f}",
                "MRR@10": f"{sum(m[1] for m in metrics) / n:.4f}",
                "nDCG@10": f"{sum(m[2] for m in metrics) / n:.4f}",
                "Queries": n,
                "Embedder": getattr(emb, "name", "?"),
                "PublicDocs": len(pub_meta),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    for r in results:
        print(r)
    print(f"Parity evaluation complete (identical rankings across stores). Saved to {out_path}")
    return results


if __name__ == "__main__":
    run_quality_parity()
