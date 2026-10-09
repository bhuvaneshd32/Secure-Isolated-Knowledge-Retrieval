import json
import csv
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from privrag.store import ChunkRecord
from privrag.store.plaintext import PlaintextStore
from privrag.rag.retriever import Retriever

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def calc_metrics(retrieved_ids, gold_ids, k=10):
    retrieved_k = retrieved_ids[:k]
    
    # Recall@k
    hits = len(set(retrieved_k) & set(gold_ids))
    recall = hits / len(gold_ids) if gold_ids else 0.0
    
    # MRR@k
    mrr = 0.0
    for i, rid in enumerate(retrieved_k):
        if rid in gold_ids:
            mrr = 1.0 / (i + 1)
            break
            
    # nDCG@k
    dcg = sum([1.0 / math.log2(i + 2) for i, rid in enumerate(retrieved_k) if rid in gold_ids])
    idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k))])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return recall, mrr, ndcg

def run_quality_parity():
    data_dir = Path("data")
    corpus = load_jsonl(data_dir / "private_corpus.jsonl")
    queries = load_jsonl(data_dir / "private_queries.jsonl")

    # Person 1 will inject MethodAStore() and MethodBStore() here later
    stores = {
        "plaintext": PlaintextStore(":memory:")
    }

    retriever = Retriever()
    # Dummy embedding since we bypassed the real LLM API for local testing
    dummy_emb = np.zeros(384, dtype=np.float32)
    
    # Group corpus by user
    user_chunks_map = defaultdict(list)
    for c in corpus:
        user_chunks_map[c['user_id']].append(c)

    results = []
    top_k_history = defaultdict(dict) # To track and assert identical results across stores

    for store_name, store in stores.items():
        # 1. Ingestion Phase
        for user_id, chunks in user_chunks_map.items():
            store.register_user(user_id, "testpass")
            session = store.unlock(user_id, "testpass")
            records = [ChunkRecord(text=c['text'], metadata=c.get('metadata', {})) for c in chunks]
            store.add_chunks(session, records)

        # 2. Query Phase
        store_metrics = []
        for q in queries:
            session = store.unlock(q['user_id'], "testpass")
            private_data = store.load_user_data(session)
            
            retrieved = retriever.retrieve(dummy_emb, q['user_id'], private_data, top_k=10)
            retrieved_ids = [r.chunk_id for r in retrieved]
            
            # The core parity assertion: verify encrypted stores retrieve the exact same IDs
            if q['query_id'] in top_k_history:
                assert retrieved_ids == top_k_history[q['query_id']], f"Parity failed for {store_name}"
            else:
                top_k_history[q['query_id']] = retrieved_ids

            recall, mrr, ndcg = calc_metrics(retrieved_ids, q['gold_chunk_ids'])
            store_metrics.append((recall, mrr, ndcg))

        # 3. Aggregate
        avg_recall = sum(m[0] for m in store_metrics) / len(store_metrics)
        avg_mrr = sum(m[1] for m in store_metrics) / len(store_metrics)
        avg_ndcg = sum(m[2] for m in store_metrics) / len(store_metrics)
        
        results.append({
            "Store": store_name,
            "Recall@10": f"{avg_recall:.4f}",
            "MRR@10": f"{avg_mrr:.4f}",
            "nDCG@10": f"{avg_ndcg:.4f}"
        })

    # 4. Save to CSV
    eval_dir = Path("eval/quality")
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "parity_results.csv"
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Store", "Recall@10", "MRR@10", "nDCG@10"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Parity evaluation complete. Saved to {out_path}")

if __name__ == "__main__":
    run_quality_parity()