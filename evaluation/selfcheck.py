"""
One-shot functional self-check of an encrypted store through the whole RAG stack
(embedder -> encrypted store -> retriever -> LLM). Works for the naive shared
baseline, Method A and Method B and Method B improved. Every system runs the
same cases under the same names, so results can be compared side by side.

    python -m evaluation.selfcheck [--store naive|A|B|B_improved] [--llm ollama|echo]
                                   [--user user_000] [--skip-llm]

Used by the demo UI ("Proof" tab). Does not modify the demo databases: the tamper
check runs on a throw-away in-memory store.
"""

import argparse
import re
import sys
from collections.abc import Callable
from datetime import date

from evaluation.key_exposure import exposure_method_a, exposure_method_b, leaked_key_at
from evaluation.leakage import INJECTION_PROMPTS, foreign_canaries, own_canaries
from privrag.rag.pipeline import NaiveSharedRAG, RagService
from privrag.rag.seed import DEMO_PASSWORD, load_jsonl
from privrag.store import ChunkRecord, IntegrityError
from privrag.store.method_a import MethodAStore
from privrag.store.method_b import MethodBStore
from privrag.store.plaintext import PlaintextStore


def date_in_text(text: str, iso: str) -> bool:
    """True if the answer states the ISO date in a common human format."""
    if iso in text:
        return True
    y, m, d = (int(x) for x in iso.split("-"))
    month = date(y, m, d).strftime("%B").lower()
    low = text.lower()
    return (
        str(y) in low
        and month[:3] in low
        and re.search(rf"\b0?{d}(st|nd|rd|th)?\b", low) is not None
    )


def make_empty_store(kind: str, kdf_n: int = 2**12):
    """In-memory store of the given kind: "naive", "A", "B" or "B_improved"."""
    if kind == "naive":
        return PlaintextStore(":memory:")
    if kind == "A":
        return MethodAStore(":memory:", kdf_n=kdf_n)
    if kind in ("B", "B_improved"):
        return MethodBStore(":memory:", mode="improved" if kind == "B_improved" else "base", kdf_n=kdf_n)
    raise ValueError(f"unsupported store {kind!r}; expected naive, A, B or B_improved")


def run_selfchecks(
    enc: RagService,
    user_id: str = "user_000",
    other_user: str = "user_001",
    corpus_path: str = "data/private_corpus.jsonl",
    queries_path: str = "data/private_queries.jsonl",
    progress: Callable[[str], None] | None = None,
    include_llm: bool = True,
) -> list[dict]:
    results: list[dict] = []

    def check(name: str, fn):
        if progress:
            progress(name)
        try:
            ok, detail = fn()
        except Exception as e:  # a crashing check is a failed check
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append({"check": name, "passed": bool(ok), "detail": detail})

    store = enc.store
    is_b = isinstance(store, MethodBStore)
    is_naive = isinstance(store, PlaintextStore)
    mode = getattr(store, "mode", None)
    if is_naive:
        store_kind = "naive"
    elif is_b:
        store_kind = "B_improved" if mode == "improved" else "B"
    else:
        store_kind = "A"
    if is_b:
        tables = ("mb_users", "mb_trapdoors", "mb_nodes")
    elif is_naive:
        tables = ("users", "chunks")
    else:
        tables = ("ma_users", "ma_records")
    # The naive system answers from ONE shared pool (paper Fig. 1), not per user.
    naive = (
        NaiveSharedRAG(
            store.dump_all(),
            enc.embedder,
            enc.llm,
            enc.retriever.public_embs,
            enc.retriever.public_meta,
        )
        if is_naive
        else None
    )
    corpus = load_jsonl(corpus_path)
    queries = load_jsonl(queries_path)
    chunk_by_id = {c["chunk_id"]: c for c in corpus}
    session = store.unlock(user_id, DEMO_PASSWORD)

    def context(question: str, k: int):
        """(retrieved chunks, prompt) exactly as this system would build them."""
        return naive.prompt_for(question, k) if naive else enc.prompt_for(session, question, k)

    def answer(question: str, k: int) -> str:
        return (naive.answer(question, k) if naive else enc.answer(session, question, k))[0]

    q = next(x for x in queries if x["user_id"] == user_id)
    gold = chunk_by_id[q["gold_chunk_ids"][0]]

    # 2 --- confidentiality at rest ---------------------------------------
    def at_rest():
        blob = b""
        for table in tables:
            for row in store.conn.execute(f"SELECT * FROM {table}"):
                blob += b"".join(c for c in row if isinstance(c, bytes))
                blob += " ".join(str(c) for c in row if isinstance(c, str)).encode()
        needles = [
            "CANARY",
            "Appointment",
            "Confidential",
            gold["metadata"]["doctor"],
            DEMO_PASSWORD,
        ]
        found = [n for n in needles if n.encode() in blob]
        n = len(store.dump_all()) if is_naive else store.record_count()
        return (not found), (
            f"{n} encrypted records scanned; no plaintext found"
            if not found
            else f"plaintext found in database: {found}"
        )

    check("Database holds no plaintext (text, doctor names, canaries, password)", at_rest)

    # 5 --- isolation: nothing foreign in the pool ---------------------------
    def isolation():
        bad = 0
        for p in INJECTION_PROMPTS:
            ret, prompt = context(p, 10)
            bad += len(foreign_canaries(prompt.split("Question:")[0], user_id))
            bad += sum(1 for r in ret if r.source not in ("public", f"private:{user_id}"))
        return (
            bad == 0,
            f"{len(INJECTION_PROMPTS)} injection prompts; foreign items in LLM context: {bad}",
        )

    check("LLM context never contains another user's data", isolation)

    # 7 --- tamper detection (throw-away store) ------------------------------
    def tamper():
        t = make_empty_store(store_kind, kdf_n=2**10)
        t.register_user("t", "tamper-test-pw")
        s = t.unlock("t", "tamper-test-pw")
        texts = ["first secret", "second secret"]
        ids = t.add_chunks(
            s,
            [
                ChunkRecord(text=x, metadata={}, embedding=e)
                for x, e in zip(texts, enc.embedder.embed(texts), strict=True)
            ],
        )
        assert len(t.load_user_data(s)) == 2
        if is_naive:
            # plaintext rows: an attacker simply edits the text; nothing can notice
            t.conn.execute("UPDATE chunks SET text = text || ' (edited)' WHERE chunk_id=?", (ids[0],))
            got = [c.text for c in t.load_user_data(s)]
            if any(x.endswith("(edited)") for x in got):
                return False, "edited record was served as genuine: no integrity protection"
            return True, "edit detected"
        if mode == "improved":
            table, key_col, col, how = "mb_nodes", "addr", "blob", "AES-GCM tag"
        elif is_b:
            table, key_col, col, how = "mb_nodes", "addr", "c_h", "H(K_i) / padding check"
        else:
            table, key_col, col, how = "ma_records", "pk", "c_chunk", "HMAC"
        sql = f"SELECT {col} FROM {table} WHERE {key_col}=?"
        val = bytearray(t.conn.execute(sql, (ids[0],)).fetchone()[0])
        val[0] ^= 1
        t.conn.execute(f"UPDATE {table} SET {col}=? WHERE {key_col}=?", (bytes(val), ids[0]))
        try:
            t.load_user_data(s)
        except IntegrityError:
            return True, f"1-bit flip in {col} -> IntegrityError (bottom), caught by {how}"
        return False, "tampered ciphertext was accepted"

    check("Tampered stored data is rejected (integrity check)", tamper)

    # 7a --- structured attacks on stored rows (what separates Method B from Method B improved) ---
    kind = store_kind
    table, key_col = {
        "naive": ("chunks", "chunk_id"),
        "A": ("ma_records", "pk"),
        "B": ("mb_nodes", "addr"),
        "B_improved": ("mb_nodes", "addr"),
    }[kind]
    swap_cols = {
        "naive": ["text", "embedding"],
        "A": ["iv_emb", "iv_chunk", "iv_meta", "c_emb", "c_chunk", "c_meta", "tag"],
        "B": [
            "iv_emb", "c_emb", "iv_chunk", "c_chunk", "iv_meta", "c_meta",
            "iv_key", "c_key", "iv_h", "c_h",
        ],  # next_addr stays: it is the chain structure, not ciphertext
        "B_improved": ["nonce", "blob"],
    }[kind]
    iv_col = {"naive": "text", "A": "iv_chunk", "B": "iv_chunk", "B_improved": "nonce"}[kind]

    def fresh_three():
        t = make_empty_store(kind, kdf_n=2**10)
        t.register_user("t", "tamper-test-pw")
        sess = t.unlock("t", "tamper-test-pw")
        texts = ["first secret", "second secret", "third secret"]
        ids = t.add_chunks(
            sess,
            [
                ChunkRecord(text=x, metadata={}, embedding=e)
                for x, e in zip(texts, enc.embedder.embed(texts), strict=True)
            ],
        )
        return t, sess, ids, texts

    def verdict(t, sess, texts, what):
        try:
            got = [c.text for c in t.load_user_data(sess)]
        except IntegrityError:
            return True, f"{what} -> IntegrityError (bottom): refused"
        return False, f"{what} NOT detected: store returned {len(got)} record(s) as genuine"

    def iv_flip():
        t, sess, ids, texts = fresh_three()
        if is_naive:
            t.conn.execute(f"UPDATE {table} SET text = 'x' || text WHERE {key_col}=?", (ids[0],))
        else:
            v = bytearray(
                t.conn.execute(f"SELECT {iv_col} FROM {table} WHERE {key_col}=?", (ids[0],))
                .fetchone()[0]
            )
            v[0] ^= 1
            t.conn.execute(
                f"UPDATE {table} SET {iv_col}=? WHERE {key_col}=?", (bytes(v), ids[0])
            )
        return verdict(t, sess, texts, f"1-bit flip in {iv_col}")

    def drop_row():
        t, sess, ids, texts = fresh_three()
        t.conn.execute(f"DELETE FROM {table} WHERE {key_col}=?", (ids[2],))
        return verdict(t, sess, texts, "deleting one stored row")

    def swap_rows():
        t, sess, ids, texts = fresh_three()
        cols = ", ".join(swap_cols)
        ra, rb = (
            t.conn.execute(f"SELECT {cols} FROM {table} WHERE {key_col}=?", (i,)).fetchone()
            for i in ids[:2]
        )
        setter = ", ".join(f"{c}=?" for c in swap_cols)
        t.conn.execute(f"UPDATE {table} SET {setter} WHERE {key_col}=?", (*rb, ids[0]))
        t.conn.execute(f"UPDATE {table} SET {setter} WHERE {key_col}=?", (*ra, ids[1]))
        return verdict(t, sess, texts, "swapping two stored rows")

    def drop_or_swap():
        ok_d, det_d = drop_row()
        ok_s, det_s = swap_rows()
        return ok_d and ok_s, f"{det_d}; {det_s}"

    check("Silent edit through the IV / nonce is rejected", iv_flip)
    check("Deleted or swapped stored rows are detected", drop_or_swap)

    # 7b --- key leak: how much does ONE leaked key give away? ------------------
    def key_leak():
        if naive:
            n = len(store.dump_all())
            return False, f"no encryption: anyone with the database file reads all {n} records"
        full = store.load_user_data(session)
        if is_b:
            pos = len(full) // 2
            readable = exposure_method_b(store, user_id, leaked_key_at(store, session, pos))
            expected = [c.text for c in full][pos:]
            return (
                readable == expected and len(readable) < len(full),
                f"key of node {pos} of {len(full)} opened {len(readable)} records "
                f"(nodes {pos}..{len(full) - 1}); the {pos} older records stay sealed",
            )
        readable = exposure_method_a(store, session)
        return (
            len(readable) < len(full),
            f"the user's single key opened {len(readable)} of {len(full)} records "
            f"(one key protects the whole account)",
        )

    check("One leaked key exposes only part of the user's records", key_leak)

    if not include_llm:
        return results

    # 8 --- real LLM: injection ----------------------------------------------
    llm_name = getattr(enc.llm, "name", type(enc.llm).__name__)

    def llm_injection():
        leaked, own = 0, 0
        for p in INJECTION_PROMPTS[:2]:
            ans = answer(p, 10)
            leaked += len(foreign_canaries(ans, user_id))
            own += len(own_canaries(ans, user_id))
        return (
            leaked == 0,
            f"LLM={llm_name}; foreign canaries in answers: {leaked}; own canaries echoed: {own}",
        )

    check(
        f"Injection attack through the LLM leaks no other user's data (LLM: {llm_name})",
        llm_injection,
    )

    # 9 --- real LLM: grounded answer ------------------------------------------
    def llm_answer():
        ans = answer(q["question"], 5)
        ok = date_in_text(ans, gold["metadata"]["date"]) or gold["metadata"]["doctor"] in ans
        return ok, f"expected date {gold['metadata']['date']}; LLM answered: {ans.strip()[:200]!r}"

    check(f"LLM answers the question from the decrypted chunk (LLM: {llm_name})", llm_answer)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None)
    ap.add_argument("--user", default="user_000")
    ap.add_argument("--store", default="A", help="naive | A | B | B_improved")
    ap.add_argument("--skip-llm", action="store_true", help="skip the checks that call the LLM")
    args = ap.parse_args()

    from privrag.rag.generator import get_llm
    from privrag.rag.ingest import get_embedder
    from privrag.rag.public_kb import load_public_kb
    from privrag.rag.retriever import Retriever
    from privrag.rag.seed import seed_store

    emb = get_embedder()
    pe, pm = load_public_kb(emb)
    llm = get_llm(args.llm)
    if hasattr(llm, "available"):
        ok, msg = llm.available()
        print(msg)
        if not ok:
            sys.exit(3)
    svc = RagService(make_empty_store(args.store), emb, Retriever(pe, pm), llm)
    seed_store(svc, load_jsonl("data/private_corpus.jsonl"))
    res = run_selfchecks(
        svc,
        args.user,
        progress=lambda n: print("..", n, flush=True),
        include_llm=not args.skip_llm,
    )
    print()
    for r in res:
        print(("PASS  " if r["passed"] else "FAIL  ") + r["check"])
        print("      " + r["detail"])
    failed = [r for r in res if not r["passed"]]
    print(f"\n{len(res) - len(failed)}/{len(res)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
