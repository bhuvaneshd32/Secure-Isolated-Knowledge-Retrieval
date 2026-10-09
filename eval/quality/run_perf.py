import time
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from privrag.store import ChunkRecord
from privrag.store.plaintext import PlaintextStore
from privrag.rag.retriever import Retriever

def generate_dummy_chunks(num_chunks):
    # Using small dummy vectors to simulate data loading speeds
    return [ChunkRecord(text=f"Performance test chunk {i}", metadata={"source": "perf"}) for i in range(num_chunks)]

def measure_throughput_and_overhead(store_class, num_chunks=5000):
    db_path = f"perf_test_{store_class.__name__}.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    store = store_class(db_path)
    store.register_user("perf_user", "pass")
    session = store.unlock("perf_user", "pass")
    
    chunks = generate_dummy_chunks(num_chunks)
    
    # 1. Ingestion Throughput
    start_time = time.perf_counter()
    store.add_chunks(session, chunks)
    end_time = time.perf_counter()
    
    throughput = num_chunks / (end_time - start_time)
    
    # 2. Disk Overhead per chunk
    disk_size = os.path.getsize(db_path)
    overhead_per_chunk = disk_size / num_chunks
    
    if os.path.exists(db_path):
        os.remove(db_path)
        
    return throughput, overhead_per_chunk

def measure_latency(store_class, corpus_sizes, queries=200):
    latencies = {size: [] for size in corpus_sizes}
    retriever = Retriever()
    dummy_query = np.zeros(384, dtype=np.float32)
    
    for size in corpus_sizes:
        # Using in-memory DB to isolate raw query/crypto logic speed from disk I/O
        store = store_class(":memory:") 
        store.register_user("perf_user", "pass")
        session = store.unlock("perf_user", "pass")
        
        chunks = generate_dummy_chunks(size)
        store.add_chunks(session, chunks)
        
        for _ in range(queries):
            start = time.perf_counter()
            # The core retrieval path: Load -> Retrieve
            user_data = store.load_user_data(session)
            _ = retriever.retrieve(dummy_query, "perf_user", user_data, top_k=5)
            end = time.perf_counter()
            
            latencies[size].append((end - start) * 1000) # Convert to milliseconds
            
    # Calculate p50 and p95 metrics
    results = {}
    for size in corpus_sizes:
        arr = np.array(latencies[size])
        results[size] = {
            "p50": np.percentile(arr, 50),
            "p95": np.percentile(arr, 95)
        }
    return results

def run_benchmarks():
    print("Starting performance benchmarks...")
    
    # Person 1 will add MethodAStore and MethodBStore here tomorrow
    stores = {
        "Plaintext RAG": PlaintextStore
    }
    
    corpus_sizes = [100, 1000, 5000]
    out_dir = Path("eval/quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    throughput_data = {}
    overhead_data = {}
    latency_data = {}
    
    for name, cls in stores.items():
        print(f"Benchmarking {name}...")
        tp, oh = measure_throughput_and_overhead(cls, num_chunks=5000)
        throughput_data[name] = tp
        overhead_data[name] = oh
        
        lat = measure_latency(cls, corpus_sizes, queries=200)
        latency_data[name] = lat
        
        print(f"  Throughput: {tp:.2f} chunks/s | Overhead: {oh:.2f} bytes/chunk")
        
    # Plot 1: Latency Scaling vs Corpus Size
    plt.figure(figsize=(8, 5))
    for name in stores.keys():
        x = corpus_sizes
        y_p50 = [latency_data[name][s]["p50"] for s in x]
        y_p95 = [latency_data[name][s]["p95"] for s in x]
        
        plt.plot(x, y_p50, marker='o', label=f"{name} (p50)")
        plt.plot(x, y_p95, marker='x', linestyle='--', label=f"{name} (p95)")
        
    plt.title("Query Latency vs Private Corpus Size")
    plt.xlabel("Number of Private Chunks")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "latency_scaling.png")
    
    # Plot 2: Storage Overhead per Chunk
    plt.figure(figsize=(8, 5))
    names = list(stores.keys())
    overheads = [overhead_data[n] for n in names]
    
    plt.bar(names, overheads, color='skyblue')
    plt.title("On-Disk Storage Overhead per Chunk")
    plt.ylabel("Bytes per Chunk")
    for i, v in enumerate(overheads):
        plt.text(i, v + 2, f"{v:.0f}", ha='center')
    plt.savefig(out_dir / "storage_overhead.png")
    
    print(f"Benchmarks complete! Plots saved to {out_dir}/")

if __name__ == "__main__":
    run_benchmarks()