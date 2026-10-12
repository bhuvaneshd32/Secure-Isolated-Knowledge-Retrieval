"""
Performance: ingest throughput, per-chunk storage overhead and query latency
vs. private corpus size, for plaintext, Method A and Method B and Method B improved.

Embeddings are random unit vectors here (we measure store + crypto cost, not
the embedding model).

    python -m evaluation.perf
"""

import os
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from privrag.rag.retriever import Retriever
from privrag.store import ChunkRecord
from privrag.store.method_a import MethodAStore
from privrag.store.method_b import MethodBStore
from privrag.store.plaintext import PlaintextStore

DIM = 384
STORES = {
    "Plaintext": lambda path: PlaintextStore(path),
    "Method A": lambda path: MethodAStore(path),
    "Method B": lambda path: MethodBStore(path, mode="base"),
    "Method B improved": lambda path: MethodBStore(path, mode="improved"),
}


def make_chunks(n, rng):
    embs = rng.normal(size=(n, DIM)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return [
        ChunkRecord(
            text=f"Performance test chunk {i} " + "lorem ipsum " * 20,
            metadata={"source": "perf"},
            embedding=embs[i],
        )
        for i in range(n)
    ]


def measure_throughput_and_overhead(factory, n=2000):
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "perf.db")
        store = factory(path)
        store.register_user("perf_user", "password-1")
        s = store.unlock("perf_user", "password-1")
        chunks = make_chunks(n, rng)
        t0 = time.perf_counter()
        store.add_chunks(s, chunks)
        dt = time.perf_counter() - t0
        store.conn.commit()
        size = os.path.getsize(path)
        store.conn.close()
    return n / dt, size / n


def measure_latency(factory, sizes, queries=50):
    rng = np.random.default_rng(1)
    retriever = Retriever()
    q = rng.normal(size=DIM).astype(np.float32)
    q /= np.linalg.norm(q)
    out = {}
    for size in sizes:
        store = factory(":memory:")
        store.register_user("perf_user", "password-1")
        s = store.unlock("perf_user", "password-1")
        store.add_chunks(s, make_chunks(size, rng))
        lat = []
        for _ in range(queries):
            t0 = time.perf_counter()
            data = store.load_user_data(s)
            retriever.retrieve(q, "perf_user", data, top_k=5)
            lat.append((time.perf_counter() - t0) * 1000)
        out[size] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
    return out


def run_benchmarks():
    sizes = [100, 500, 1000, 2000]
    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tp, oh, lat = {}, {}, {}
    for name, f in STORES.items():
        print(f"Benchmarking {name} ...")
        tp[name], oh[name] = measure_throughput_and_overhead(f)
        lat[name] = measure_latency(f, sizes)
        print(
            f"  ingest {tp[name]:.0f} chunks/s | {oh[name]:.0f} bytes/chunk | "
            + " ".join(f"n={s}: p50={lat[name][s]['p50']:.1f}ms" for s in sizes)
        )

    plt.figure(figsize=(8, 5))
    for name in STORES:
        plt.plot(sizes, [lat[name][s]["p50"] for s in sizes], marker="o", label=f"{name} (p50)")
        plt.plot(
            sizes,
            [lat[name][s]["p95"] for s in sizes],
            marker="x",
            linestyle="--",
            label=f"{name} (p95)",
        )
    plt.title("Query latency (load + decrypt + retrieve) vs private corpus size")
    plt.xlabel("Private chunks")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "latency_scaling.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    names = list(STORES)
    vals = [oh[n] for n in names]
    plt.bar(names, vals, color=["#8aa4c8", "#2f6f4f", "#b5651d", "#8e44ad"])
    for i, v in enumerate(vals):
        plt.text(i, v + 20, f"{v:.0f}", ha="center")
    plt.title("On-disk storage per chunk (384-d embedding + ~300 B text)")
    plt.ylabel("Bytes per chunk")
    plt.savefig(out_dir / "storage_overhead.png", dpi=120)
    plt.close()

    import csv

    with open(out_dir / "perf_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["Store", "ingest_chunks_per_s", "bytes_per_chunk"]
            + [f"p50_ms@{s}" for s in sizes]
            + [f"p95_ms@{s}" for s in sizes]
        )
        for n in names:
            w.writerow(
                [n, f"{tp[n]:.1f}", f"{oh[n]:.0f}"]
                + [f"{lat[n][s]['p50']:.2f}" for s in sizes]
                + [f"{lat[n][s]['p95']:.2f}" for s in sizes]
            )
    print(f"Done. Plots + perf_results.csv in {out_dir}/")


if __name__ == "__main__":
    run_benchmarks()
