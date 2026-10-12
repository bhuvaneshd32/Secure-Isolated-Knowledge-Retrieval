"""
Privacy-Aware RAG demo, Method A.

    python build_kb.py            # once: builds + embeds + encrypts everything
    streamlit run demo/app.py     # then: sign in, ask, inspect each step

All embedding and encryption of the dataset happened in build_kb.py, so this UI
starts instantly. The dataset is fixed: only the question you type is embedded here.
"""

import hashlib
import html
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import streamlit as st

from evaluation.key_exposure import exposure_method_a, exposure_method_b, leaked_key_at
from evaluation.leakage import INJECTION_PROMPTS, foreign_canaries
from evaluation.selfcheck import run_selfchecks
from privrag.rag.generator import LLMError, OllamaLLM, build_prompt, get_llm
from privrag.rag.ingest import get_embedder
from privrag.rag.pipeline import NaiveSharedRAG, RagService
from privrag.rag.public_kb import load_public_kb
from privrag.rag.retriever import Retriever
from privrag.store import AuthenticationError, ChunkRecord, IntegrityError
from privrag.store.method_a import MethodAStore
from privrag.store.method_b import MethodBStore
from privrag.store.plaintext import PlaintextStore

st.set_page_config(page_title="Privacy-Aware RAG", layout="wide")

DEMO_DIR = Path(os.getenv("PRIVRAG_DEMO_DIR", ROOT / ".demo"))
MANIFEST = DEMO_DIR / "manifest.json"
DEMO_MAX_TOKENS = int(os.getenv("OLLAMA_NUM_PREDICT") or 256)  # same default as evaluation.run_all


# ------------------------------------------------------------------ backend
@st.cache_resource(show_spinner="Opening the encrypted store ...")
def get_backend():
    """Open what build_kb.py prepared. Nothing is embedded or encrypted here."""
    if not MANIFEST.exists() or not (DEMO_DIR / "method_b_hardened.db").exists():
        return None
    info = json.loads(MANIFEST.read_text(encoding="utf-8"))
    emb = get_embedder(info["embedder"])  # lazy: the model loads on the first question
    pub_embs, pub_meta = load_public_kb(emb)  # cached .npy, no model needed
    retriever = Retriever(pub_embs, pub_meta)
    llm = get_llm()
    if getattr(llm, "num_predict", 1) is None:  # an unbounded answer is what times out
        llm.num_predict = DEMO_MAX_TOKENS
    if hasattr(llm, "warm_up") and llm_status("ollama", llm.model, llm.host)[0]:
        llm.warm_up()  # load the model now, not on the first question
    return {
        "info": info,
        "embedder": emb,
        "pub": (pub_embs, pub_meta),
        "llm": llm,
        "enc": RagService(MethodAStore(str(DEMO_DIR / "method_a.db")), emb, retriever, llm),
        "b": {
            mode: RagService(
                MethodBStore(str(DEMO_DIR / f"method_b_{mode}.db"), mode=mode),
                emb,
                retriever,
                llm,
            )
            for mode in ("hardened", "paper")
        },
        "plain": RagService(PlaintextStore(str(DEMO_DIR / "plaintext.db")), emb, retriever, llm),
    }


@st.cache_data(ttl=20, show_spinner=False)
def llm_status(kind: str, model: str, host: str) -> tuple[bool, str]:
    if kind != "ollama":
        return True, "Echo test model (deliberately leaky, no Ollama needed)"
    return OllamaLLM(model=model, host=host).available()


def fingerprint(key: bytes) -> str:
    return hashlib.sha256(b"fp" + key).hexdigest()[:12]


def render(text: str, uid: str) -> str:
    """HTML-escape, then colour canary tokens: own = green, someone else's = red."""
    t = html.escape(text)

    def repl(m):
        color = "#1a7f37" if m.group(1) == uid.split("_")[-1] else "#cf222e"
        return f'<span style="color:{color};font-weight:700">{m.group(0)}</span>'

    return re.sub(r"CANARY-(\d{3})-\d{3}", repl, t).replace("\n", "<br>")


def origin(source: str, uid: str) -> str:
    if source == "public":
        return "Public"
    return "Yours" if source == f"private:{uid}" else f"Someone else ({source.split(':')[-1]})"


def show_answer(col, title: str, answer: str, retrieved, uid: str) -> None:
    with col:
        st.markdown(f"**{title}**")
        st.markdown(
            f'<div style="border:1px solid #888;border-radius:6px;padding:8px">'
            f"{render(answer, uid)}</div>",
            unsafe_allow_html=True,
        )
        foreign = foreign_canaries(answer, uid)
        foreign_srcs = [r for r in retrieved if origin(r.source, uid).startswith("Someone")]
        if foreign:
            st.error(f"LEAK: {len(foreign)} other-user secret(s) appeared in the answer")
        elif foreign_srcs:
            st.error(
                f"LEAK: the model was given {len(foreign_srcs)} record(s) belonging to "
                "another user, so its answer may be built from their data"
            )
        else:
            st.success("No other user's data in the answer")
        st.markdown("**Sources the model was given**")
        for r in retrieved:
            who = origin(r.source, uid)
            icon = "🔴" if who.startswith("Someone") else ("🟢" if who == "Yours" else "⚪")
            st.markdown(
                f"{icon} **{who}** · similarity {r.score:.2f}<br>{render(r.text, uid)}",
                unsafe_allow_html=True,
            )


def step(n: int, title: str, detail: str) -> None:
    st.markdown(f"✅ **Step {n} — {title}**  \n{detail}")


# --------------------------------------------------------------------- UI
b = get_backend()
st.title("Privacy-Aware RAG")

if b is None:
    st.error("The knowledge base has not been built yet (or was built before Method B existed).")
    st.code("python build_kb.py", language="bash")
    st.stop()

enc: RagService = b["enc"]
plain: RagService = b["plain"]
info = b["info"]

# ---- sign in ---------------------------------------------------------------
if "session" not in st.session_state:
    st.write(
        "Every user's notes are encrypted with **their own key**. "
        "Sign in to unlock yours — nobody else's data is ever decrypted for you."
    )
    with st.form("login"):
        uid_in = st.text_input("User id")
        pw_in = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    st.caption(
        f"Demo accounts: {info['users'][0]} … {info['users'][-1]} · password: {info['password']}"
    )
    if submitted:
        try:
            st.session_state["session"] = enc.store.unlock(uid_in.strip(), pw_in)
            st.session_state["plain_session"] = plain.store.unlock(uid_in.strip(), pw_in)
            st.session_state["b_sessions"] = {
                m: svc.store.unlock(uid_in.strip(), pw_in) for m, svc in b["b"].items()
            }
            st.session_state["uid"] = uid_in.strip()
            st.rerun()
        except AuthenticationError:
            st.error("Wrong user id or password. (Nothing was decrypted.)")
    st.stop()

uid: str = st.session_state["uid"]
session = st.session_state["session"]
plain_session = st.session_state["plain_session"]
b_sessions = st.session_state["b_sessions"]

# ---- sidebar: who am I, what is loaded --------------------------------------
with st.sidebar:
    st.header("Session")
    st.success(f"Signed in as **{uid}**")
    st.markdown(
        f"🔑 Encryption key `{fingerprint(session.key('enc'))}`  \n"
        f"🔏 Integrity key `{fingerprint(session.key('mac'))}`"
    )
    st.caption(
        "Fingerprints only. The real keys live in server memory for this session and are "
        "stored in the database only wrapped under your password."
    )
    if st.button("Sign out"):
        session.close()
        for k in ("session", "plain_session", "b_sessions", "uid"):
            st.session_state.pop(k, None)
        st.rerun()

    st.header("Method B variant")
    b_mode = st.radio(
        "Chain integrity",
        ["hardened", "paper"],
        format_func=lambda m: {
            "hardened": "Hardened (AES-GCM + bound AAD)",
            "paper": "As in the paper (AES-CBC, H(K) only)",
        }[m],
        label_visibility="collapsed",
    )
    encB: RagService = b["b"][b_mode]
    session_b = b_sessions[b_mode]

    st.header("System")
    st.markdown(
        f"Encrypted records (A): **{enc.store.record_count()}**, "
        f"(B): **{encB.store.record_count()}**  \n"
        f"Public documents: **{len(b['pub'][1])}**  \n"
        f"Embedder: `{info['embedder']}`"
    )
    llm = b["llm"]
    ok, msg = llm_status(
        getattr(llm, "name", "echo"), getattr(llm, "model", ""), getattr(llm, "host", "")
    )
    (st.success if ok else st.error)(msg)

tab_ask, tab_docs, tab_public, tab_hood, tab_leak, tab_proof = st.tabs(
    ["Ask", "My documents", "Public knowledge base", "Under the hood", "A vs B: key leak", "Proof"]
)

# ---- Ask --------------------------------------------------------------------
with tab_ask:
    options = ["(type your own)"] + INJECTION_PROMPTS
    preset = st.selectbox(
        "Example prompts — the four below are prompt-injection attacks (paper Figure 1)", options
    )
    q = st.text_input(
        "Your question",
        value="" if preset == options[0] else preset,
        placeholder="e.g. What medication was I prescribed for hypertension?  (see My documents)",
    )
    c1, c2 = st.columns([3, 1])
    compare = c1.checkbox("Also run a naive shared system (no isolation) to compare", value=True)
    k = c2.slider("Sources (top-k)", 1, 10, 5)

    def run_secure(svc: RagService, sess):
        """One privacy-aware query, timed per phase (fetch+verify+decrypt, embed)."""
        t0 = time.perf_counter()
        private = svc.store.load_user_data(sess)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        qv = svc.embedder.embed_one(q)
        t_emb = time.perf_counter() - t0
        retrieved = svc.retriever.retrieve(qv, uid, private, top_k=k)
        return {
            "n_private": len(private),
            "t_load": t_load,
            "t_emb": t_emb,
            "retrieved": retrieved,
            "prompt": build_prompt(q, retrieved),
        }

    if st.button("Ask", type="primary") and q.strip():
        cols = st.columns(3) if compare else st.columns(2)
        col_naive = cols[0] if compare else None
        col_a, col_b = cols[-2], cols[-1]
        runs: dict[str, dict] = {}
        answered: dict[str, tuple[str, float]] = {}  # prompt -> (answer, seconds), this click only
        for key, svc, sess, col, title in (
            ("A", enc, session, col_a, f"Method A (AES-CBC + HMAC) — {uid}"),
            ("B", encB, session_b, col_b, f"Method B ({b_mode} chain) — {uid}"),
        ):
            try:
                r = run_secure(svc, sess)
                r["reused"] = r["prompt"] in answered
                if r["reused"]:  # identical prompt, temperature 0: the model would say the same
                    r["answer"], r["t_llm"] = answered[r["prompt"]]
                else:
                    t0 = time.perf_counter()
                    with st.spinner(
                        f"Method {key}: asking {getattr(svc.llm, 'name', 'the LLM')} ..."
                    ):
                        r["answer"] = svc.llm.generate(r["prompt"])
                    r["t_llm"] = time.perf_counter() - t0
                    answered[r["prompt"]] = (r["answer"], r["t_llm"])
                runs[key] = r
                show_answer(col, title, r["answer"], r["retrieved"], uid)
            except IntegrityError:
                col.error("Stored data failed its integrity check (⊥). Request refused.")
            except LLMError as e:
                col.error(str(e))

        if compare:
            try:
                naive = NaiveSharedRAG(plain.store.dump_all(), b["embedder"], b["llm"], *b["pub"])
                with st.spinner("Naive system: thinking ..."):
                    n_ans, n_ret, _ = naive.answer(q, k)
                show_answer(
                    col_naive, "Naive shared knowledge base (no isolation)", n_ans, n_ret, uid
                )
            except LLMError as e:
                col_naive.error(str(e))
        st.caption("🟢 your data · 🔴 another user's data · ⚪ public knowledge base")

        if "A" in runs and "B" in runs:
            same = [r.text for r in runs["A"]["retrieved"]] == [
                r.text for r in runs["B"]["retrieved"]
            ]
            (st.success if same else st.error)(
                "Method A and Method B retrieved the same sources in the same order — "
                "the scheme changes how data is protected, not what is found."
                if same
                else "Method A and Method B retrieved different sources!"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "System": name,
                            "Records decrypted": runs[k_]["n_private"],
                            "Fetch + verify + decrypt (ms)": round(runs[k_]["t_load"] * 1000, 1),
                            "Question embedding (ms)": round(runs[k_]["t_emb"] * 1000, 1),
                            "LLM (s)": round(runs[k_]["t_llm"], 1),
                        }
                        for name, k_ in (("Method A", "A"), ("Method B", "B"))
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            if runs["B"]["reused"]:
                st.caption(
                    "Method B built the same prompt as Method A, so the language model was "
                    "asked once and its answer and time are shown for both."
                )

        if "A" in runs:
            trace = runs["A"]
            total = enc.store.record_count()
            n_pub = len(b["pub"][1])
            n_priv = trace["n_private"]
            with st.expander("How Method A produced this answer, step by step"):
                step(
                    1,
                    "Authenticate",
                    f"Your password unlocked your two keys (encryption "
                    f"`{fingerprint(session.key('enc'))}`, integrity "
                    f"`{fingerprint(session.key('mac'))}`).",
                )
                step(
                    2,
                    "Fetch your encrypted records",
                    f"Your private list of record ids pointed to **{n_priv}** rows out of "
                    f"**{total}** in the shared table.",
                )
                step(
                    3,
                    "Verify, then decrypt",
                    f"HMAC-SHA256 tag checked on all {n_priv} rows *before* decrypting (any "
                    f"mismatch would stop here). Then AES-256-CBC decrypted each embedding and "
                    f"text — {trace['t_load'] * 1000:.0f} ms.",
                )
                step(
                    4,
                    "Build your private pool",
                    f"{n_priv} of your chunks + {n_pub} public chunks = **{n_priv + n_pub}** "
                    f"candidates. The other **{total - n_priv}** encrypted rows were never "
                    f"decrypted.",
                )
                step(
                    5,
                    "Embed the question and rank",
                    f"Question embedded in {trace['t_emb'] * 1000:.0f} ms; cosine similarity "
                    f"against the pool; top-{k} kept.",
                )
                step(
                    6,
                    "Ask the language model",
                    f"The prompt below contained only those {len(trace['retrieved'])} chunks "
                    f"— {trace['t_llm']:.1f} s.",
                )
                st.code(trace["prompt"])

        if "B" in runs:
            trace = runs["B"]
            n_priv = trace["n_private"]
            with st.expander(f"How Method B ({b_mode}) produced this answer, step by step"):
                step(
                    1,
                    "Authenticate",
                    "Your password was stretched with scrypt into a master key and a key-salt. "
                    "H(ID‖keysalt) un-masked your trapdoor; the ID inside matched yours "
                    "(VerifyID), so the password was right.",
                )
                step(
                    2,
                    "Read the head of your chain",
                    "The trapdoor gave the first node's key K₁ and address. There is no owner "
                    "column and no list of record ids — the only way in is the chain.",
                )
                step(
                    3,
                    "Walk the chain (Algorithm 1)",
                    f"Each node was decrypted with its own key Kᵢ, H(Kᵢ) was checked, and the "
                    f"node's next key and next address led to the following node, until Null: "
                    f"**{n_priv}** nodes in {trace['t_load'] * 1000:.0f} ms. "
                    + (
                        "Hardened mode: each node is AES-GCM with its address bound in, so any "
                        "change stops the walk."
                        if b_mode == "hardened"
                        else "Paper mode: only H(Kᵢ) is checked — no ciphertext MAC."
                    ),
                )
                step(
                    4,
                    "Rank and answer",
                    f"Same retrieval as Method A: your {n_priv} chunks + public KB, cosine "
                    f"top-{k}, then the language model.",
                )

# ---- My documents -------------------------------------------------------------
with tab_docs:
    try:
        chunks = enc.store.load_user_data(session)
    except IntegrityError:
        st.error("Your stored data failed its integrity check.")
        chunks = []
    st.write(
        f"**{len(chunks)}** documents (fixed dataset, embedded and encrypted at build time). "
        "They are stored encrypted; this list was decrypted just now with your key."
    )
    if chunks:
        st.dataframe(
            pd.DataFrame(
                {
                    "Document": [c.text for c in chunks],
                    "Date": [c.metadata.get("date", "") for c in chunks],
                    "Doctor": [c.metadata.get("doctor", "") for c in chunks],
                }
            ),
            width="stretch",
            height=320,
            hide_index=True,
        )

# ---- Public KB ----------------------------------------------------------------
with tab_public:
    pub_meta = b["pub"][1]
    st.write(
        f"**{len(pub_meta)}** public documents. Visible to every user, never encrypted, "
        "and mixed into every query's candidate pool."
    )
    needle = st.text_input("Filter", placeholder="type to filter ...")
    rows = [m["text"] for m in pub_meta if needle.lower() in m["text"].lower()]
    st.dataframe(
        pd.DataFrame({"Document": rows[:500]}), width="stretch", height=320, hide_index=True
    )

# ---- Under the hood --------------------------------------------------------------
with tab_hood:
    st.write("### What someone who steals the database file sees")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Naive store (plaintext)**")
        prow = plain.store.conn.execute("SELECT owner, text FROM chunks LIMIT 6").fetchall()
        st.dataframe(
            pd.DataFrame(prow, columns=["owner", "text"]), width="stretch", hide_index=True
        )
        st.caption("Readable, and every row says whose it is.")
    with c2:
        st.markdown("**Method A (encrypted)**")
        raw = enc.store.raw_records(6)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "record id": r["pk"][:12] + "…",
                        "encrypted text": r["c_chunk"][:40] + "…",
                        "integrity tag": r["tag"][:24] + "…",
                    }
                    for r in raw
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption("Random bytes, no owner column, no readable text.")

    st.markdown(f"**Method B ({b_mode} chain)**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    k_: (v[:28] + "…" if isinstance(v, str) and len(v) > 28 else v)
                    for k_, v in r.items()
                }
                for r in encB.store.raw_records(6)
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Random node addresses and ciphertext; no owner column. "
        + (
            "Hardened: the pointer to the next node is inside the ciphertext."
            if b_mode == "hardened"
            else "Paper mode: `next_addr` is stored in clear, so chain structure is visible."
        )
    )

    st.write("### How Method A works")
    st.markdown("""
1. **Register** — two random 256-bit keys are created per user: one to encrypt, one to authenticate.
   They are stored only *wrapped* under a key derived from the password.
2. **Build time** — each chunk's embedding and text are AES-256-CBC encrypted with a fresh random IV.
   An HMAC-SHA256 tag covers the record. The record's id is added to the user's private list.
3. **Ask** — the user's list picks their rows. Each tag is verified, then decrypted.
   Rows that belong to others are never touched.
4. **Retrieve** — the decrypted private chunks are merged with the public KB; cosine top-k feeds the LLM.
""")

# ---- A vs B: key leak ---------------------------------------------------------------
with tab_leak:
    st.write("### Where Method B is better than Method A: a leaked key")
    st.markdown("""
Answering questions gives **identical** results in A and B (see the *Ask* tab). They differ in
what an attacker gains from a **leaked key**.

* **Method A** protects *all* of a user's records with one encryption key and one MAC key.
  Whoever gets them decrypts everything.
* **Method B** gives every record its own key, derived one-way from the previous one
  (`K(i+1) = HKDF(K(i))`). A key from the middle of the chain opens that record and the
  newer ones, but **cannot** open the older ones.

Realistic cause: a worker, cache or debug log that held the key of record *j* is compromised
together with the database file.
""")
    hb_store: MethodBStore = b["b"]["hardened"].store
    hb_session = b_sessions["hardened"]
    n_rec = len(hb_store.load_user_data(hb_session))
    pos = st.slider(
        "Chain position of the leaked key (0 = oldest record)",
        0,
        max(n_rec - 1, 0),
        min(20, max(n_rec - 1, 0)),
    )
    if st.button("Simulate the leak", type="primary"):
        got_a = exposure_method_a(enc.store, session)
        got_b = exposure_method_b(hb_store, uid, leaked_key_at(hb_store, hb_session, pos))
        c1, c2 = st.columns(2)
        c1.metric("Method A: records the attacker can read", f"{len(got_a)} / {n_rec}")
        c2.metric("Method B: records the attacker can read", f"{len(got_b)} / {n_rec}")
        st.success(
            f"With the same kind of leak, B kept the {n_rec - len(got_b)} older record(s) "
            f"secret; A kept {n_rec - len(got_a)}."
        )
        olds = [c.text for c in hb_store.load_user_data(hb_session)][:pos]
        with st.expander("What the attacker sees"):
            st.markdown("**Method B: readable records**")
            st.dataframe(pd.DataFrame({"Document": got_b}), hide_index=True, width="stretch")
            st.markdown(f"**Method B: {len(olds)} older records stay unreadable**")
            st.caption("Their keys cannot be computed from the leaked key (HKDF is one-way).")
            st.markdown("**Method A: readable records**")
            st.dataframe(pd.DataFrame({"Document": got_a}), hide_index=True, width="stretch")

    st.divider()
    st.write("### Where Method A is better: tampering")
    st.markdown("""
The paper's Method B has **no ciphertext authentication**, so flipping one bit of a stored IV
changes the decrypted text without any error. Method A (HMAC) and our hardened Method B
(AES-GCM) reject the same change. Runs on throw-away in-memory copies.
""")
    if st.button("Flip one bit in the IV of a stored record"):
        rng = np.random.default_rng(0)
        vec = rng.normal(size=384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        rec = ChunkRecord(
            text="Prescription on 2025-05-01: metformin 500 mg daily for 30 days.",
            metadata={},
            embedding=vec,
        )
        rows = []

        def attempt(label, store, flip):
            store.register_user("t", "pw")
            ses = store.unlock("t", "pw")
            store.add_chunks(ses, [rec])
            flip(store)
            try:
                out = store.load_user_data(ses)[0].text
                verdict = "NOT DETECTED, text silently changed" if out != rec.text else "unchanged"
                rows.append({"System": label, "Result": verdict, "Text after tampering": out})
            except IntegrityError:
                rows.append(
                    {"System": label, "Result": "DETECTED (⊥)", "Text after tampering": "-"}
                )

        def flip_bit(table, col):
            def _f(store):
                v = bytearray(store.conn.execute(f"SELECT {col} FROM {table}").fetchone()[0])
                v[0] ^= 1
                store.conn.execute(f"UPDATE {table} SET {col} = ?", (bytes(v),))

            return _f

        attempt(
            "Method A", MethodAStore(":memory:", kdf_n=2**10), flip_bit("ma_records", "iv_chunk")
        )
        attempt(
            "Method B (paper)",
            MethodBStore(":memory:", mode="paper", kdf_n=2**10),
            flip_bit("mb_nodes", "iv_chunk"),
        )
        attempt(
            "Method B (hardened)",
            MethodBStore(":memory:", mode="hardened", kdf_n=2**10),
            flip_bit("mb_nodes", "nonce"),
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---- Proof -------------------------------------------------------------------------
with tab_proof:
    st.write("### One-click check that each method behaves as the paper says")
    st.caption(
        "Runs on the real encrypted stores. The tampering test uses a throw-away in-memory "
        "copy, so your demo data is untouched."
    )
    other = next(u for u in info["users"] if u != uid)
    targets_label = st.radio(
        "Check",
        ["Method A", f"Method B ({b_mode})", "Both"],
        horizontal=True,
    )
    with_llm = st.checkbox(
        "Include the checks that call the language model (slower)", value=True
    )
    chosen = {
        "Method A": [("Method A", enc)],
        f"Method B ({b_mode})": [(f"Method B ({b_mode})", encB)],
        "Both": [("Method A", enc), (f"Method B ({b_mode})", encB)],
    }[targets_label]

    if st.button("Run all checks", type="primary"):
        for label, svc in chosen:
            st.markdown(f"#### {label}")
            holder = st.empty()
            try:
                results = run_selfchecks(
                    svc,
                    uid,
                    other,
                    progress=lambda n, h=holder: h.info(f"Running: {n} ..."),
                    include_llm=with_llm,
                )
                holder.empty()
                passed = sum(r["passed"] for r in results)
                (st.success if passed == len(results) else st.error)(
                    f"{label}: {passed} of {len(results)} checks passed"
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Result": "PASS" if r["passed"] else "FAIL",
                                "Check": r["check"],
                                "Detail": r["detail"],
                            }
                            for r in results
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            except LLMError as e:
                holder.empty()
                st.error(str(e))
