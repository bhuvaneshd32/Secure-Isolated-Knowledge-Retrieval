"""
Privacy-Aware RAG demo, Method A.

    python build_kb.py            # once: builds + embeds + encrypts everything
    streamlit run demo/app.py     # then: sign in, ask, inspect each step

All embedding and encryption of the dataset happened in build_kb.py, so this UI
starts instantly. The dataset is fixed: only the question you type is embedded here.
"""

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
    if not MANIFEST.exists() or not (DEMO_DIR / "method_b_improved.db").exists():
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
            for mode in ("improved", "base")
        },
        "plain": RagService(PlaintextStore(str(DEMO_DIR / "plaintext.db")), emb, retriever, llm),
    }


@st.cache_data(ttl=20, show_spinner=False)
def llm_status(kind: str, model: str, host: str) -> tuple[bool, str]:
    if kind != "ollama":
        return True, "Echo test model (deliberately leaky, no Ollama needed)"
    return OllamaLLM(model=model, host=host).available()


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
    with st.form("login"):
        uid_in = st.text_input("User id")
        pw_in = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
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

# ---- sidebar: who am I, is the LLM up --------------------------------------
with st.sidebar:
    st.header("Session")
    st.success(f"Signed in as **{uid}**")
    if st.button("Sign out"):
        session.close()
        for k in ("session", "plain_session", "b_sessions", "uid"):
            st.session_state.pop(k, None)
        st.rerun()

    llm = b["llm"]
    ok, msg = llm_status(
        getattr(llm, "name", "echo"), getattr(llm, "model", ""), getattr(llm, "host", "")
    )
    (st.success if ok else st.error)(msg)

tab_ask, tab_docs, tab_public, tab_leak, tab_proof = st.tabs(
    ["Ask", "My documents", "Public knowledge base", "Key leak & tampering", "Proof"]
)

# ---- Ask --------------------------------------------------------------------
with tab_ask:
    options = ["(type your own)"] + INJECTION_PROMPTS
    preset = st.selectbox(
        "Example prompts for prompt injection", options
    )
    q = st.text_input(
        "Your question",
        value="" if preset == options[0] else preset,
        placeholder="e.g. What medication was I prescribed for hypertension?  (see My documents)",
    )
    c1, c2 = st.columns([3, 1])
    compare = c1.checkbox("Also run Naive RAG (no isolation) to compare", value=True)
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
        cols = st.columns(4) if compare else st.columns(3)
        col_naive = cols[0] if compare else None
        col_a, col_b, col_bh = cols[-3], cols[-2], cols[-1]
        svc_b, sess_b = b["b"]["base"], b_sessions["base"]
        svc_bh, sess_bh = b["b"]["improved"], b_sessions["improved"]
        runs: dict[str, dict] = {}
        answered: dict[str, tuple[str, float]] = {}  # prompt -> (answer, seconds), this click only
        for key, svc, sess, col, title in (
            ("A", enc, session, col_a, f"Method A — {uid}"),
            ("B", svc_b, sess_b, col_b, f"Method B — {uid}"),
            ("BH", svc_bh, sess_bh, col_bh, f"Method B improved — {uid}"),
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
                with st.spinner("Naive RAG: thinking ..."):
                    n_ans, n_ret, _ = naive.answer(q, k)
                show_answer(
                    col_naive, "Naive RAG (no isolation)", n_ans, n_ret, uid
                )
            except LLMError as e:
                col_naive.error(str(e))
        st.caption("🟢 your data · 🔴 another user's data · ⚪ public knowledge base")


# ---- My documents -------------------------------------------------------------
with tab_docs:
    try:
        chunks = enc.store.load_user_data(session)
    except IntegrityError:
        st.error("Your stored data failed its integrity check.")
        chunks = []
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
    needle = st.text_input("Filter", placeholder="type to filter ...")
    rows = [m["text"] for m in pub_meta if needle.lower() in m["text"].lower()]
    st.dataframe(
        pd.DataFrame({"Document": rows[:500]}), width="stretch", height=320, hide_index=True
    )

# ---- Key leak & tampering ---------------------------------------------------------------
with tab_leak:
    st.write("### Leaked key")
    b_store: MethodBStore = b["b"]["base"].store
    b_session = b_sessions["base"]
    hb_store: MethodBStore = b["b"]["improved"].store
    hb_session = b_sessions["improved"]
    n_rec = len(hb_store.load_user_data(hb_session))
    pos = st.slider(
        "Chain position of the leaked key (0 = oldest record)",
        0,
        max(n_rec - 1, 0),
        min(20, max(n_rec - 1, 0)),
    )
    if st.button("Simulate the leak", type="primary"):
        got = {
            "Method A": exposure_method_a(enc.store, session),
            "Method B": exposure_method_b(b_store, uid, leaked_key_at(b_store, b_session, pos)),
            "Method B improved": exposure_method_b(
                hb_store, uid, leaked_key_at(hb_store, hb_session, pos)
            ),
        }
        for col, (name, rows_) in zip(st.columns(3), got.items(), strict=True):
            col.metric(f"{name}: records the attacker can read", f"{len(rows_)} / {n_rec}")
        st.success(
            "Older records kept secret: "
            + ", ".join(f"{name} {n_rec - len(rows_)}" for name, rows_ in got.items())
            + "."
        )
        with st.expander("What the attacker sees"):
            for name, rows_ in got.items():
                st.markdown(f"**{name}: readable records**")
                st.dataframe(pd.DataFrame({"Document": rows_}), hide_index=True, width="stretch")
                if name != "Method A":
                    st.caption(
                        f"{n_rec - len(rows_)} older records stay unreadable: their keys cannot "
                        "be computed from the leaked key (HKDF is one-way)."
                    )

    st.divider()
    st.write("### IV bit-flip")
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
            "Method B",
            MethodBStore(":memory:", mode="base", kdf_n=2**10),
            flip_bit("mb_nodes", "iv_chunk"),
        )
        attempt(
            "Method B improved",
            MethodBStore(":memory:", mode="improved", kdf_n=2**10),
            flip_bit("mb_nodes", "nonce"),
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---- Proof -------------------------------------------------------------------------
with tab_proof:
    other = next(u for u in info["users"] if u != uid)
    systems = {
        "Naive RAG": plain,
        "Method A": enc,
        "Method B": b["b"]["base"],
        "Method B improved": b["b"]["improved"],
    }
    with_llm = st.checkbox(
        "Also test with the live LLM: injection attack and answer quality (slower)", value=True
    )
    chosen = list(systems.items())

    if st.button("Run all checks", type="primary"):
        all_results: dict[str, list[dict]] = {}
        for label, svc in chosen:
            holder = st.empty()
            try:
                all_results[label] = run_selfchecks(
                    svc,
                    uid,
                    other,
                    progress=lambda n, h=holder, lb=label: h.info(f"{lb}: running {n} ..."),
                    include_llm=with_llm,
                )
            except LLMError as e:
                st.error(f"{label}: {e}")
            holder.empty()

        if len(all_results) > 1:
            names = [r["check"] for r in next(iter(all_results.values()))]
            matrix = []
            for n_ in names:
                row = {"Case": n_}
                for label, res in all_results.items():
                    hit = next((r for r in res if r["check"] == n_), None)
                    row[label] = "—" if hit is None else ("PASS" if hit["passed"] else "FAIL")
                matrix.append(row)
            row = {"Case": "Total passed"}
            for label, res in all_results.items():
                row[label] = f"{sum(r['passed'] for r in res)} / {len(res)}"
            matrix.append(row)
            st.table(pd.DataFrame(matrix).set_index("Case"))
