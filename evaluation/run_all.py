"""
Full evaluation of every system, written to CSV files for analysis.

    python -m evaluation.run_all                    # default run (about 30-40 min with Ollama)
    python -m evaluation.run_all --quick            # smoke run (a few minutes)
    python -m evaluation.run_all --resume evaluation/results/full_run_<timestamp>
    python -m evaluation.run_all --public-kb path/to/public_kb.jsonl

Systems compared (all in-memory, same data, same embeddings):

    plaintext          isolated store, nothing encrypted (baseline)
    method_a           AES-CBC + HMAC
    method_b     chained keys, as published (no ciphertext MAC)
    method_b_improved  chained keys with AES-GCM
    naive_shared       one shared plaintext knowledge base (the vulnerable system of
                       paper Fig. 1); used as the reference row in the leakage CSV only

CSV files written (rows are appended as they are produced, so an interrupted run keeps
everything and `--resume` continues it):

    leakage.csv    system x user x injection prompt: foreign secrets in the LLM prompt (all users,
                   all prompts, no LLM needed) and in the answer (sampled, needs the LLM)
    qa.csv         system x query: gold rank, hit@k, LLM answer and whether it is correct
    retrieval.csv  system: Recall@10, MRR@10, nDCG@10 and parity with the plaintext ranking
    key_exposure.csv  system x user x leaked-key position: records an attacker can read
    tamper.csv     system x attack: detected / silent modification / silent data loss
    perf.csv       system x corpus size: query latency; ingest rate and bytes per chunk
    summary.csv    one row per system, ready for a table or chart
    run_info.json  settings, versions and timings of this run
"""

import argparse
import csv
import json
import math
import os
import platform
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import perf as perfmod
from evaluation.key_exposure import exposure_method_a, exposure_method_b, leaked_key_at
from evaluation.leakage import INJECTION_PROMPTS, foreign_canaries, own_canaries
from evaluation.selfcheck import date_in_text
from privrag.rag.generator import LLMError, build_prompt, get_llm
from privrag.rag.ingest import get_embedder
from privrag.rag.pipeline import NaiveSharedRAG, RagService
from privrag.rag.public_kb import load_public_kb
from privrag.rag.retriever import Retriever
from privrag.rag.seed import DEMO_PASSWORD, load_jsonl
from privrag.store import AuthenticationError, ChunkRecord, IntegrityError
from privrag.store.method_a import MethodAStore
from privrag.store.method_b import MethodBStore
from privrag.store.plaintext import PlaintextStore

SYSTEMS = {
    "plaintext": lambda: PlaintextStore(":memory:"),
    "method_a": lambda: MethodAStore(":memory:", kdf_n=2**12),
    "method_b": lambda: MethodBStore(":memory:", mode="base", kdf_n=2**12),
    "method_b_improved": lambda: MethodBStore(":memory:", mode="improved", kdf_n=2**12),
}
PERF_NAMES = {
    "plaintext": "Plaintext",
    "method_a": "Method A",
    "method_b": "Method B",
    "method_b_improved": "Method B improved",
}
NAIVE = "naive_shared"
# "repeat all the context" prompts can make a model generate for minutes; bound them.
DEFAULT_MAX_TOKENS = 256
DEFAULT_LLM_TIMEOUT = 600.0


# ---------------------------------------------------------------------- CSV log
class CsvLog:
    """Append-only CSV with a header; remembers which keys already exist (for --resume)."""

    def __init__(self, path: Path, columns: list[str], key_cols: list[str]):
        self.path, self.columns, self.key_cols = path, columns, key_cols
        self.seen: set[tuple] = set()
        new = not path.exists() or path.stat().st_size == 0
        if not new:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.columns = columns = list(reader.fieldnames or columns)  # keep the file's header
                for row in reader:
                    self.seen.add(tuple(row[c] for c in key_cols))
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=columns, extrasaction="ignore")
        if new:
            self._w.writeheader()
            self._f.flush()

    def has(self, **key) -> bool:
        return tuple(str(key[c]) for c in self.key_cols) in self.seen

    def write(self, row: dict) -> None:
        self._w.writerow(row)
        self._f.flush()
        self.seen.add(tuple(str(row[c]) for c in self.key_cols))

    def close(self) -> None:
        self._f.close()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def say(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------- helpers
def unit_vec(seed: int, dim: int = 384) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def judge_answer(qtype: str, meta: dict, answer: str) -> bool:
    """Does the LLM answer contain the fact the question asks for?"""
    low = answer.lower().replace(",", "")
    if qtype == "appointment":
        return date_in_text(answer, meta["date"])
    if qtype == "prescription":
        return str(meta.get("drug", "")).lower() in low
    if qtype == "lab":
        return str(meta.get("value")) in low
    if qtype == "note":
        return str(meta.get("condition", "")).lower() in low
    if qtype == "billing":
        return str(meta.get("amount")) in low
    return False


class LlmCaller:
    """Calls the model with retries, optionally re-using the answer for an identical prompt.

    A call that still fails after the retries is recorded as an error (and left out of the
    answer metrics) instead of ending the whole run.
    """

    def __init__(self, llm, reuse: bool, retries: int = 2):
        self.llm, self.reuse, self.retries, self.cache = llm, reuse, retries, {}
        self.calls = 0
        self.errors = 0

    def ask(self, prompt: str) -> tuple[str, float, bool, str]:
        """Returns (answer, seconds, reused, error). `error` is "" on success."""
        if self.reuse and prompt in self.cache:
            return self.cache[prompt], 0.0, True, ""
        last = ""
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                ans = self.llm.generate(prompt)
            except LLMError as e:
                last = str(e)
                waited = time.perf_counter() - t0
                say(
                    f"  LLM error after {waited:.0f}s "
                    f"(attempt {attempt + 1}/{self.retries + 1}): {last[:110]}"
                )
                continue
            dt = time.perf_counter() - t0
            self.calls += 1
            if self.reuse:
                self.cache[prompt] = ans
            return ans, dt, False, ""
        self.errors += 1
        return "", 0.0, False, last


# ------------------------------------------------------------------------ build
def build_systems(corpus, embedder, retriever, llm):
    """Seed every system with identical data and identical (pre-computed) embeddings."""
    texts = [c["text"] for c in corpus]
    embs = embedder.embed(texts)
    by_user = defaultdict(list)
    for c, e in zip(corpus, embs, strict=True):
        by_user[c["user_id"]].append(ChunkRecord(c["text"], dict(c["metadata"]), e))

    services, sessions = {}, {}
    for name, make in SYSTEMS.items():
        store = make()
        sessions[name] = {}
        for uid, recs in by_user.items():
            store.register_user(uid, DEMO_PASSWORD)
            s = store.unlock(uid, DEMO_PASSWORD)
            store.add_chunks(s, recs)
            sessions[name][uid] = s
        services[name] = RagService(store, embedder, retriever, llm)
    naive = NaiveSharedRAG(
        services["plaintext"].store.dump_all(),
        embedder,
        llm,
        retriever.public_embs,
        retriever.public_meta,
    )
    return services, sessions, naive


# --------------------------------------------------------------------- leakage
def run_leakage(args, services, sessions, naive, caller, users, out: Path):
    cols = [
        "system", "user_id", "prompt_id", "n_sources", "foreign_sources", "foreign_in_prompt",
        "own_in_prompt", "llm_called", "foreign_in_answer", "own_in_answer", "llm_seconds", "answer",
        "llm_error",
    ]  # fmt: skip
    log = CsvLog(out / "leakage.csv", cols, ["system", "user_id", "prompt_id"])
    prompts = INJECTION_PROMPTS[: args.prompts]
    systems = [NAIVE, *SYSTEMS]
    total = len(systems) * len(users) * len(prompts)
    done = 0
    for system in systems:
        for ui, uid in enumerate(users):
            for pi, p in enumerate(prompts):
                done += 1
                if log.has(system=system, user_id=uid, prompt_id=pi):
                    continue
                if system == NAIVE:
                    retrieved, prompt = naive.prompt_for(p, 10)
                else:
                    retrieved, prompt = services[system].prompt_for(sessions[system][uid], p, 10)
                context = prompt.split("Question:")[0]
                row = {
                    "system": system,
                    "user_id": uid,
                    "prompt_id": pi,
                    "n_sources": len(retrieved),
                    "foreign_sources": sum(
                        1 for r in retrieved if r.source not in ("public", f"private:{uid}")
                    ),
                    "foreign_in_prompt": len(foreign_canaries(context, uid)),
                    "own_in_prompt": len(own_canaries(context, uid)),
                    "llm_called": 0,
                }
                if not args.skip_llm and ui < args.llm_users and pi < args.llm_prompts:
                    ans, dt, reused, err = caller.ask(prompt)
                    if err:
                        row.update(llm_called=1, llm_error=err[:300])
                        say(f"leakage {done}/{total}  {system:18} {uid} p{pi}  LLM ERROR (recorded)")
                    else:
                        row.update(
                            llm_called=0 if reused else 1,
                            foreign_in_answer=len(foreign_canaries(ans, uid)),
                            own_in_answer=len(own_canaries(ans, uid)),
                            llm_seconds=round(dt, 2),
                            answer=ans.strip()[:2000],
                        )
                        say(f"leakage {done}/{total}  {system:18} {uid} p{pi}  {dt:5.1f}s")
                log.write(row)
    log.close()
    say("leakage done")


# ------------------------------------------------------------------------- QA
def run_qa(args, services, sessions, caller, corpus, queries, out: Path):
    cols = [
        "system", "query_id", "user_id", "query_type", "question", "gold_rank", "hit_at_1",
        "hit_at_3", "hit_at_5", "hit_at_10", "retrieval_ms", "same_ranking_as_plaintext",
        "same_scores_as_plaintext", "llm_called", "llm_seconds", "answer_correct", "answer",
        "llm_error",
    ]  # fmt: skip
    log = CsvLog(out / "qa.csv", cols, ["system", "query_id"])
    chunk_by_id = {c["chunk_id"]: c for c in corpus}
    reference: dict[str, list[str]] = {}
    reference_scores: dict[str, list[float]] = {}
    qs = queries[: args.qa_queries] if args.qa_queries else queries
    total = len(qs) * len(SYSTEMS)
    done = 0
    for system in SYSTEMS:  # plaintext first: it is the ranking reference
        svc = services[system]
        for q in qs:
            done += 1
            uid = q["user_id"]
            gold = chunk_by_id[q["gold_chunk_ids"][0]]
            t0 = time.perf_counter()
            retrieved = svc.retrieve(sessions[system][uid], q["question"], k=10)
            dt_ret = (time.perf_counter() - t0) * 1000
            ranked = [r.text for r in retrieved]
            scores = [round(r.score, 5) for r in retrieved]
            if system == "plaintext":
                reference[q["query_id"]] = ranked
                reference_scores[q["query_id"]] = scores
            rank = next((i + 1 for i, t in enumerate(ranked) if t == gold["text"]), None)
            if log.has(system=system, query_id=q["query_id"]):
                continue
            row = {
                "system": system,
                "query_id": q["query_id"],
                "user_id": uid,
                "query_type": gold["metadata"]["type"],
                "question": q["question"],
                "gold_rank": rank if rank else "",
                "hit_at_1": int(rank is not None and rank <= 1),
                "hit_at_3": int(rank is not None and rank <= 3),
                "hit_at_5": int(rank is not None and rank <= 5),
                "hit_at_10": int(rank is not None),
                "retrieval_ms": round(dt_ret, 1),
                "same_ranking_as_plaintext": int(ranked == reference.get(q["query_id"], ranked)),
                # equal-score chunks may swap places between stores (random chunk ids break
                # ties), so the score sequence is the tie-proof parity measure
                "same_scores_as_plaintext": int(
                    scores == reference_scores.get(q["query_id"], scores)
                ),
                "llm_called": 0,
            }
            if not args.skip_llm:
                prompt = build_prompt(q["question"], retrieved[:5])
                ans, dt, reused, err = caller.ask(prompt)
                if err:
                    row.update(llm_called=1, llm_error=err[:300])
                    say(f"qa {done}/{total}  {system:18} {q['query_id']}  LLM ERROR (recorded)")
                else:
                    row.update(
                        llm_called=0 if reused else 1,
                        llm_seconds=round(dt, 2),
                        answer_correct=int(judge_answer(row["query_type"], gold["metadata"], ans)),
                        answer=ans.strip()[:1000],
                    )
                    say(f"qa {done}/{total}  {system:18} {q['query_id']}  {dt:5.1f}s")
            log.write(row)
    log.close()
    say("qa done")


def write_retrieval_summary(out: Path) -> None:
    rows = read_csv(out / "qa.csv")
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    with open(out / "retrieval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["system", "queries", "recall_at_10", "mrr_at_10", "ndcg_at_10", "hit_at_1",
             "hit_at_3", "hit_at_5", "ranking_parity_with_plaintext",
             "score_parity_with_plaintext"]
        )  # fmt: skip
        for system, rs in by_sys.items():
            ranks = [int(r["gold_rank"]) if r["gold_rank"] else None for r in rs]
            n = len(rs)
            w.writerow(
                [
                    system,
                    n,
                    f"{sum(1 for x in ranks if x) / n:.4f}",
                    f"{sum(1 / x for x in ranks if x) / n:.4f}",
                    f"{sum(1 / math.log2(x + 1) for x in ranks if x) / n:.4f}",
                    f"{sum(int(r['hit_at_1']) for r in rs) / n:.4f}",
                    f"{sum(int(r['hit_at_3']) for r in rs) / n:.4f}",
                    f"{sum(int(r['hit_at_5']) for r in rs) / n:.4f}",
                    f"{sum(int(r['same_ranking_as_plaintext']) for r in rs) / n:.4f}",
                    f"{sum(int(r['same_scores_as_plaintext']) for r in rs) / n:.4f}",
                ]
            )


# --------------------------------------------------------------- key exposure
def run_key_exposure(services, sessions, users, out: Path):
    cols = ["system", "user_id", "position", "n_records", "readable", "readable_fraction"]
    log = CsvLog(out / "key_exposure.csv", cols, ["system", "user_id", "position"])
    for system in SYSTEMS:
        store = services[system].store
        for uid in users:
            s = sessions[system][uid]
            n = len(store.load_user_data(s))
            fixed = None
            if system == "plaintext":
                fixed = n  # no key at all: everything is readable
            elif system == "method_a":
                fixed = len(exposure_method_a(store, s))  # the user's one key opens everything
            for pos in range(n):
                if log.has(system=system, user_id=uid, position=pos):
                    continue
                readable = (
                    fixed
                    if fixed is not None
                    else len(exposure_method_b(store, uid, leaked_key_at(store, s, pos)))
                )
                log.write(
                    {
                        "system": system,
                        "user_id": uid,
                        "position": pos,
                        "n_records": n,
                        "readable": readable,
                        "readable_fraction": round(readable / n, 4),
                    }
                )
        say(f"key exposure done: {system}")
    log.close()


# --------------------------------------------------------------------- tamper
# where each system keeps a record's ciphertext, its IV/nonce, and its row key
TAMPER_LAYOUT = {
    "plaintext": ("chunks", "chunk_id", "text", None),
    "method_a": ("ma_records", "pk", "c_chunk", "iv_chunk"),
    "method_b": ("mb_nodes", "addr", "c_chunk", "iv_chunk"),
    "method_b_improved": ("mb_nodes", "addr", "blob", "nonce"),
}
ATTACKS = ["flip_ciphertext", "flip_iv", "swap_rows", "delete_row", "rollback"]


def _flip(v):
    if isinstance(v, str):
        return ("X" if v[:1] != "X" else "Y") + v[1:]
    b = bytearray(v)
    b[0] ^= 1
    return bytes(b)


def run_tamper(out: Path):
    cols = ["system", "attack", "applicable", "detected", "outcome", "detail"]
    log = CsvLog(out / "tamper.csv", cols, ["system", "attack"])
    for system, make in SYSTEMS.items():
        table, key, col, ivcol = TAMPER_LAYOUT[system]
        for attack in ATTACKS:
            if log.has(system=system, attack=attack):
                continue
            store = make()
            store.register_user("t", "tamper-test-pw")
            s = store.unlock("t", "tamper-test-pw")
            texts = [f"confidential record number {i}" for i in range(5)]
            ids = store.add_chunks(
                s, [ChunkRecord(t, {}, unit_vec(i)) for i, t in enumerate(texts)]
            )
            store.conn.commit()
            c = store.conn
            expected = list(texts)
            applicable = True
            if attack == "flip_ciphertext":
                v = c.execute(f"SELECT {col} FROM {table} WHERE {key}=?", (ids[0],)).fetchone()[0]
                c.execute(f"UPDATE {table} SET {col}=? WHERE {key}=?", (_flip(v), ids[0]))
            elif attack == "flip_iv":
                if ivcol is None:
                    applicable = False
                else:
                    v = c.execute(
                        f"SELECT {ivcol} FROM {table} WHERE {key}=?", (ids[0],)
                    ).fetchone()[0]
                    c.execute(f"UPDATE {table} SET {ivcol}=? WHERE {key}=?", (_flip(v), ids[0]))
            elif attack == "swap_rows":
                cs = [col] + ([ivcol] if ivcol else [])
                sel = ", ".join(cs)
                ra = c.execute(f"SELECT {sel} FROM {table} WHERE {key}=?", (ids[0],)).fetchone()
                rb = c.execute(f"SELECT {sel} FROM {table} WHERE {key}=?", (ids[1],)).fetchone()
                setc = ", ".join(f"{x}=?" for x in cs)
                c.execute(f"UPDATE {table} SET {setc} WHERE {key}=?", (*rb, ids[0]))
                c.execute(f"UPDATE {table} SET {setc} WHERE {key}=?", (*ra, ids[1]))
            elif attack == "delete_row":
                c.execute(f"DELETE FROM {table} WHERE {key}=?", (ids[-1],))
            elif attack == "rollback":
                snap = sqlite3.connect(":memory:")
                c.backup(snap)  # attacker's copy of the database as it is now
                store.add_chunks(s, [ChunkRecord("added after the snapshot", {}, unit_vec(99))])
                store.conn.commit()
                expected = [*texts, "added after the snapshot"]
                snap.backup(c)  # attacker restores the older, perfectly valid, state
                snap.close()
            c.commit()

            if not applicable:
                log.write(
                    {"system": system, "attack": attack, "applicable": 0, "detected": "",
                     "outcome": "not_applicable", "detail": "no separate IV in this system"}
                )  # fmt: skip
                continue
            try:
                loaded = [x.text for x in store.load_user_data(s)]
                if loaded == expected:
                    outcome, detected, detail = "no_effect", 0, "data unchanged"
                elif attack == "rollback":
                    outcome, detected, detail = "silent_rollback", 0, "older state accepted"
                elif len(loaded) < len(expected):
                    outcome, detected, detail = "silent_data_loss", 0, f"{len(loaded)} of 5 left"
                else:
                    outcome, detected, detail = "silent_modification", 0, "content changed"
            except (IntegrityError, AuthenticationError) as e:
                outcome, detected, detail = "detected", 1, type(e).__name__
            log.write(
                {"system": system, "attack": attack, "applicable": 1, "detected": detected,
                 "outcome": outcome, "detail": detail}
            )  # fmt: skip
        say(f"tamper done: {system}")
    log.close()


# ------------------------------------------------------------------------ perf
def run_perf(args, out: Path):
    cols = ["system", "n_chunks", "p50_ms", "p95_ms", "ingest_chunks_per_s", "bytes_per_chunk"]
    log = CsvLog(out / "perf.csv", cols, ["system", "n_chunks"])
    sizes = [100, 500] if args.quick else [100, 500, 1000, 2000]
    n_ingest = 500 if args.quick else 2000
    for system in SYSTEMS:
        factory = perfmod.STORES[PERF_NAMES[system]]
        if all(log.has(system=system, n_chunks=s) for s in sizes):
            continue
        tp, oh = perfmod.measure_throughput_and_overhead(factory, n=n_ingest)
        lat = perfmod.measure_latency(factory, sizes, queries=args.perf_queries)
        for s in sizes:
            log.write(
                {
                    "system": system,
                    "n_chunks": s,
                    "p50_ms": round(lat[s]["p50"], 2),
                    "p95_ms": round(lat[s]["p95"], 2),
                    "ingest_chunks_per_s": round(tp, 1),
                    "bytes_per_chunk": round(oh),
                }
            )
        say(f"perf done: {system}")
    log.close()


# --------------------------------------------------------------------- summary
def write_summary(out: Path) -> None:
    def avg(rows, col):
        vals = [float(r[col]) for r in rows if r.get(col, "") != ""]
        return round(sum(vals) / len(vals), 4) if vals else ""

    leak = read_csv(out / "leakage.csv")
    qa = read_csv(out / "qa.csv")
    ret = {r["system"]: r for r in read_csv(out / "retrieval.csv")}
    ke = read_csv(out / "key_exposure.csv")
    tam = read_csv(out / "tamper.csv")
    perf = read_csv(out / "perf.csv")
    systems = [NAIVE, *SYSTEMS]
    cols = [
        "system", "leak_attacks", "foreign_secrets_in_prompt", "attacks_with_prompt_leak",
        "llm_attacks", "llm_attacks_failed", "foreign_secrets_in_answer", "qa_queries",
        "qa_llm_failed", "qa_answer_accuracy",
        "recall_at_10", "mrr_at_10", "ndcg_at_10", "ranking_parity_with_plaintext",
        "score_parity_with_plaintext", "mean_retrieval_ms", "mean_llm_seconds", "key_leak_mean_fraction_readable",
        "tamper_attacks_applicable", "tamper_detected", "tamper_detection_rate",
        "ingest_chunks_per_s", "bytes_per_chunk", "p50_ms_at_largest", "largest_n",
    ]  # fmt: skip
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in systems:
            L = [r for r in leak if r["system"] == s]
            Lq = [r for r in L if r.get("llm_seconds", "") != ""]
            Q = [r for r in qa if r["system"] == s and r.get("answer_correct", "") != ""]
            Qr = [r for r in qa if r["system"] == s]
            K = [r for r in ke if r["system"] == s]
            T = [r for r in tam if r["system"] == s and r["applicable"] == "1"]
            P = sorted([r for r in perf if r["system"] == s], key=lambda r: int(r["n_chunks"]))
            row = {
                "system": s,
                "leak_attacks": len(L),
                "foreign_secrets_in_prompt": sum(int(r["foreign_in_prompt"]) for r in L),
                "attacks_with_prompt_leak": sum(1 for r in L if int(r["foreign_in_prompt"]) > 0),
                "llm_attacks": len(Lq),
                # a call was attempted (llm_called == 1) but produced no answer
                "llm_attacks_failed": sum(
                    1 for r in L if r.get("llm_called") == "1" and r.get("llm_seconds", "") == ""
                ),
                "foreign_secrets_in_answer": sum(
                    int(r["foreign_in_answer"] or 0) for r in Lq
                ),
            }
            if Qr:
                row.update(
                    qa_queries=len(Qr),
                    qa_llm_failed=sum(
                        1
                        for r in Qr
                        if r.get("llm_called") == "1" and r.get("llm_seconds", "") == ""
                    ),
                    qa_answer_accuracy=avg(Q, "answer_correct"),
                    recall_at_10=ret[s]["recall_at_10"] if s in ret else "",
                    mrr_at_10=ret[s]["mrr_at_10"] if s in ret else "",
                    ndcg_at_10=ret[s]["ndcg_at_10"] if s in ret else "",
                    ranking_parity_with_plaintext=(
                        ret[s]["ranking_parity_with_plaintext"] if s in ret else ""
                    ),
                    score_parity_with_plaintext=(
                        ret[s]["score_parity_with_plaintext"] if s in ret else ""
                    ),
                    mean_retrieval_ms=avg(Qr, "retrieval_ms"),
                    mean_llm_seconds=avg([r for r in Q if r.get("llm_called") == "1"], "llm_seconds"),
                )
            if K:
                row["key_leak_mean_fraction_readable"] = avg(K, "readable_fraction")
            if T:
                det = sum(int(r["detected"]) for r in T)
                row.update(
                    tamper_attacks_applicable=len(T),
                    tamper_detected=det,
                    tamper_detection_rate=round(det / len(T), 4),
                )
            if P:
                row.update(
                    ingest_chunks_per_s=P[0]["ingest_chunks_per_s"],
                    bytes_per_chunk=P[0]["bytes_per_chunk"],
                    p50_ms_at_largest=P[-1]["p50_ms"],
                    largest_n=P[-1]["n_chunks"],
                )
            w.writerow(row)


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=None, help="output folder (default: results/full_run_<time>)")
    ap.add_argument("--resume", default=None, help="continue an earlier run in this folder")
    ap.add_argument("--llm", default=None, help="ollama (default) | echo")
    ap.add_argument("--skip-llm", action="store_true", help="no LLM calls at all (fast)")
    ap.add_argument("--reuse-llm", action="store_true", help="re-use the answer of an identical prompt")
    ap.add_argument("--users", type=int, default=10, help="users in the leakage attack")
    ap.add_argument("--prompts", type=int, default=4, help="injection prompts in the attack")
    ap.add_argument("--llm-users", type=int, default=3, help="users whose attacks also call the LLM")
    ap.add_argument("--llm-prompts", type=int, default=2, help="prompts per user that call the LLM")
    ap.add_argument("--qa-queries", type=int, default=None, help="limit QA queries (default: all)")
    ap.add_argument("--perf-queries", type=int, default=20, help="timed queries per corpus size")
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument(
        "--llm-timeout", type=float, default=None,
        help=f"seconds per LLM request (default {DEFAULT_LLM_TIMEOUT:.0f}, or OLLAMA_TIMEOUT)",
    )
    ap.add_argument(
        "--llm-max-tokens", type=int, default=None,
        help=f"cap on generated tokens, bounds long 'repeat the context' answers "
        f"(default {DEFAULT_MAX_TOKENS}, or OLLAMA_NUM_PREDICT)",
    )
    ap.add_argument("--llm-retries", type=int, default=2, help="retries per failed LLM call")
    ap.add_argument(
        "--public-kb", default=str(ROOT / "data" / "public_kb.jsonl"),
        help="public knowledge base JSONL (its embedding cache is written next to it)",
    )
    ap.add_argument("--quick", action="store_true", help="small smoke run")
    args = ap.parse_args()
    if args.quick:
        args.users, args.prompts, args.llm_users, args.llm_prompts = 3, 2, 1, 1
        args.qa_queries = args.qa_queries or 6
        args.perf_queries = 5

    out = Path(args.resume or args.out or ROOT / "evaluation" / "results" /
               f"full_run_{datetime.now():%Y%m%d_%H%M%S}")  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    say(f"output folder: {out}")

    llm = get_llm("echo" if args.skip_llm else args.llm)
    if hasattr(llm, "timeout"):
        if args.llm_timeout is not None:
            llm.timeout = args.llm_timeout
        elif not os.getenv("OLLAMA_TIMEOUT"):
            llm.timeout = DEFAULT_LLM_TIMEOUT
    if hasattr(llm, "num_predict"):
        if args.llm_max_tokens is not None:
            llm.num_predict = args.llm_max_tokens
        elif llm.num_predict is None:  # no CLI value and no OLLAMA_NUM_PREDICT
            llm.num_predict = DEFAULT_MAX_TOKENS
        say(f"LLM limits: max {llm.num_predict} tokens, {llm.timeout:.0f} s per request")
    if not args.skip_llm and hasattr(llm, "available"):
        ok, msg = llm.available()
        say(msg)
        if not ok:
            sys.exit("Ollama is not available. Start it, or use --skip-llm / --llm echo.")

    corpus = load_jsonl(ROOT / "data" / "private_corpus.jsonl")
    queries = load_jsonl(ROOT / "data" / "private_queries.jsonl")
    all_users = sorted({c["user_id"] for c in corpus})
    users = all_users[: args.users]

    say("loading embedder and public knowledge base ...")
    embedder = get_embedder()
    pub_embs, pub_meta = load_public_kb(embedder, Path(args.public_kb))
    retriever = Retriever(pub_embs, pub_meta)

    say("seeding the four systems (embeddings computed once) ...")
    services, sessions, naive = build_systems(corpus, embedder, retriever, llm)
    caller = LlmCaller(llm, reuse=args.reuse_llm, retries=args.llm_retries)

    try:
        say("1/6 leakage (injection attack) ...")
        run_leakage(args, services, sessions, naive, caller, users, out)
        say("2/6 question answering and retrieval quality ...")
        run_qa(args, services, sessions, caller, corpus, queries, out)
        write_retrieval_summary(out)
        say("3/6 key exposure ...")
        run_key_exposure(services, sessions, users, out)
        say("4/6 tamper detection ...")
        run_tamper(out)
        if not args.skip_perf:
            say("5/6 performance ...")
            run_perf(args, out)
        say("6/6 summary ...")
    except LLMError as e:
        sys.exit(f"LLM error: {e}\nProgress is saved; re-run with --resume {out}")
    except KeyboardInterrupt:
        sys.exit(f"Interrupted. Progress is saved; re-run with --resume {out}")

    write_summary(out)
    info = {
        "finished": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - started),
        "args": vars(args),
        "llm": getattr(llm, "name", type(llm).__name__),
        "llm_model": getattr(llm, "model", ""),
        "embedder": getattr(embedder, "name", ""),
        "users": len(users),
        "private_chunks": len(corpus),
        "public_docs": len(pub_meta),
        "queries": len(queries),
        "llm_calls": caller.calls,
        "llm_errors": caller.errors,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    say(f"finished in {info['seconds']} s. Files in {out}:")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
