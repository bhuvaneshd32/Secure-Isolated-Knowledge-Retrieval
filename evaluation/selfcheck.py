"""
One-shot functional self-check of an encrypted store through the whole RAG stack
(embedder -> encrypted store -> retriever -> LLM). Works for Method A and for
Method B (paper and hardened modes).

    python -m evaluation.selfcheck [--store A|B_paper|B_hardened] [--llm ollama|echo]
                                   [--user user_000] [--skip-llm]

Used by the demo UI ("Proof" tab). Does not modify the demo databases: the tamper
check runs on a throw-away in-memory store.
"""

import argparse
import re
import sys
from collections.abc import Callable
from datetime import date

from evaluation.key_exposure import exposure_method_b, leaked_key_at
from evaluation.leakage import INJECTION_PROMPTS, foreign_canaries, own_canaries
from privrag.rag.pipeline import RagService
from privrag.rag.seed import DEMO_PASSWORD, load_jsonl
from privrag.store import AuthenticationError, ChunkRecord, IntegrityError, IsolationViolation
from privrag.store.method_a import MethodAStore
from privrag.store.method_b import MethodBStore


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
    """In-memory store of the given kind: "A", "B_paper" or "B_hardened"."""
    if kind == "A":
        return MethodAStore(":memory:", kdf_n=kdf_n)
    if kind in ("B_paper", "B_hardened"):
        return MethodBStore(":memory:", mode=kind.split("_", 1)[1], kdf_n=kdf_n)
    raise ValueError(f"unsupported store {kind!r}; expected A, B_paper or B_hardened")


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
    mode = getattr(store, "mode", None)
    tables = ("mb_users", "mb_trapdoors", "mb_nodes") if is_b else ("ma_users", "ma_records")
    corpus = load_jsonl(corpus_path)
    queries = load_jsonl(queries_path)
    chunk_by_id = {c["chunk_id"]: c for c in corpus}
    session = store.unlock(user_id, DEMO_PASSWORD)
    q = next(x for x in queries if x["user_id"] == user_id)
    gold = chunk_by_id[q["gold_chunk_ids"][0]]

    # 1 --- authentication -------------------------------------------------
    def wrong_pw():
        try:
            store.unlock(user_id, "definitely-wrong")
        except AuthenticationError:
            return True, "wrong password -> AuthenticationError"
        return False, "wrong password was accepted"

    check("Wrong password is rejected", wrong_pw)

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
        n = store.record_count()
        return (not found), (
            f"{n} encrypted records scanned; no plaintext found"
            if not found
            else f"plaintext found in database: {found}"
        )

    check("Database holds no plaintext (text, doctor names, canaries, password)", at_rest)

    # 3 --- round trip -------------------------------------------------------
    def round_trip():
        data = store.load_user_data(session)
        expected = {c["text"] for c in corpus if c["user_id"] == user_id}
        got = {c.text for c in data}
        dims = {len(c.embedding) for c in data}
        return (
            (got == expected and dims == {384}),
            f"{len(data)} chunks decrypted; texts match originals: {got == expected}; "
            f"embedding dims: {sorted(dims)}",
        )

    check("Decrypt round-trip returns the original chunks + 384-d embeddings", round_trip)

    # 4 --- retrieval --------------------------------------------------------
    def retrieval():
        ret = enc.retrieve(session, q["question"], k=5)
        rank = next((i + 1 for i, r in enumerate(ret) if gold["text"] == r.text), None)
        return (
            rank is not None and rank <= 3,
            f"question: {q['question']!r} -> gold chunk rank {rank}",
        )

    check("Retrieval finds the right appointment (rank <= 3)", retrieval)

    # 5 --- isolation: nothing foreign in the pool ---------------------------
    def isolation():
        bad = 0
        for p in INJECTION_PROMPTS:
            ret, prompt = enc.prompt_for(session, p, 10)
            bad += len(foreign_canaries(prompt.split("Question:")[0], user_id))
            bad += sum(1 for r in ret if r.source not in ("public", f"private:{user_id}"))
        return (
            bad == 0,
            f"{len(INJECTION_PROMPTS)} injection prompts; foreign items in LLM context: {bad}",
        )

    check("LLM context never contains another user's data", isolation)

    # 6 --- cross-user delete -------------------------------------------------
    def cross_delete():
        other = store.unlock(other_user, DEMO_PASSWORD)
        target = store.load_user_data(other)[0].chunk_id
        try:
            store.delete_chunk(session, target)
        except IsolationViolation:
            still = any(c.chunk_id == target for c in store.load_user_data(other))
            return still, "IsolationViolation raised; victim's chunk still present"
        return False, "deletion of another user's chunk succeeded"

    check("Cannot delete another user's chunk", cross_delete)

    # 7 --- tamper detection (throw-away store) ------------------------------
    def tamper():
        t = make_empty_store(f"B_{mode}" if is_b else "A", kdf_n=2**10)
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
        if mode == "hardened":
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

    tamper_name = {
        None: "Tampered ciphertext is rejected (HMAC)",
        "hardened": "Tampered node is rejected (AES-GCM tag)",
        "paper": "Corrupted key-hash field is rejected (H(K_i) check)",
    }[mode]
    check(tamper_name, tamper)

    # 7b --- Method B: a leaked chain key opens only its own node and later ones ---
    if is_b:

        def chain_key_leak():
            full = store.load_user_data(session)
            pos = len(full) // 2
            readable = exposure_method_b(store, user_id, leaked_key_at(store, session, pos))
            expected = [c.text for c in full][pos:]
            return (
                readable == expected,
                f"key of node {pos} of {len(full)} opened {len(readable)} records "
                f"(nodes {pos}..{len(full) - 1}); the {pos} older records stay sealed",
            )

        check("A leaked node key cannot open older records (chain is one-way)", chain_key_leak)

    if not include_llm:
        return results

    # 8 --- real LLM: injection ----------------------------------------------
    llm_name = getattr(enc.llm, "name", type(enc.llm).__name__)

    def llm_injection():
        leaked, own = 0, 0
        for p in INJECTION_PROMPTS[:2]:
            ans, _, _ = enc.answer(session, p, 10)
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
        ans, _, _ = enc.answer(session, q["question"], 5)
        ok = date_in_text(ans, gold["metadata"]["date"]) or gold["metadata"]["doctor"] in ans
        return ok, f"expected date {gold['metadata']['date']}; LLM answered: {ans.strip()[:200]!r}"

    check(f"LLM answers the question from the decrypted chunk (LLM: {llm_name})", llm_answer)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None)
    ap.add_argument("--user", default="user_000")
    ap.add_argument("--store", default="A", help="A | B_paper | B_hardened")
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
