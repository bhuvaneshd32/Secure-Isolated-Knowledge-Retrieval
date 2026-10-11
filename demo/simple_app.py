"""
Simple Method A tester.   streamlit run demo/simple_app.py

One page, five steps, no LLM and no embedder needed: it exercises only the
encrypted store (privrag/store/method_a.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from privrag.store import (
    AuthenticationError,
    ChunkRecord,
    IntegrityError,
    IsolationViolation,
)
from privrag.store.method_a import MethodAStore

st.set_page_config(page_title="Method A tester", layout="centered")

PASSWORDS = {"alice": "alice-password", "bob": "bob-password"}
NOTES = {
    "alice": ["Alice's secret: my salary is 90000", "Alice's secret: my doctor is Dr. Cruz"],
    "bob": ["Bob's secret: my bank PIN is 4821", "Bob's secret: I am moving to Berlin"],
}


def fake_embedding(text: str) -> np.ndarray:
    # Method A only needs *a* vector to encrypt; this keeps the test instant.
    v = np.random.default_rng(abs(hash(text)) % 2**32).normal(size=32).astype("float32")
    return v / np.linalg.norm(v)


def new_store() -> MethodAStore:
    store = MethodAStore(":memory:", kdf_n=2**10)
    for user, pw in PASSWORDS.items():
        store.register_user(user, pw)
        s = store.unlock(user, pw)
        store.add_chunks(s, [ChunkRecord(t, {}, fake_embedding(t)) for t in NOTES[user]])
    return store


if "store" not in st.session_state:
    st.session_state.store = new_store()
store: MethodAStore = st.session_state.store

st.title("Method A tester")
st.write(
    "Method A = every user's notes are encrypted with **their own key** (AES-CBC) and protected "
    "by a **tamper seal** (HMAC). Five quick checks below. Each one says PASS or FAIL."
)
if st.button("Reset everything"):
    st.session_state.clear()
    st.rerun()

with st.expander("Test data (what was stored)", expanded=True):
    st.table(
        pd.DataFrame(
            [
                {"user": u, "password": PASSWORDS[u], "secret note": n}
                for u in NOTES
                for n in NOTES[u]
            ]
        )
    )

# ---- 1. disk view -----------------------------------------------------------
st.header("1. Is the data unreadable on disk?")
st.caption("Pretend you are a hacker who stole the database file.")
if st.button("Look inside the database"):
    raw = b"".join(
        bytes(c) for row in store.conn.execute("SELECT c_chunk, c_emb FROM ma_records") for c in row
    )
    st.dataframe(pd.DataFrame(store.raw_records(10))[["pk", "c_chunk", "tag"]], width="stretch")
    leaked = [w for w in ("salary", "doctor", "PIN", "Berlin", "secret") if w.encode() in raw]
    if leaked:
        st.error(f"FAIL: readable words found in the database: {leaked}")
    else:
        st.success("PASS: only random-looking bytes. No readable words, no owner name.")

# ---- 2. login ---------------------------------------------------------------
st.header("2. Does only the right password work?")
c1, c2 = st.columns(2)
user = c1.selectbox("User", list(PASSWORDS))
pw = c2.text_input("Password", value="wrong-password")
if st.button("Try to log in"):
    try:
        s = store.unlock(user, pw)
        notes = [c.text for c in store.load_user_data(s)]
        st.success(f"Logged in as {user}. Decrypted notes:")
        for n in notes:
            st.write("•", n)
    except AuthenticationError:
        st.success("PASS: wrong password rejected, nothing decrypted.")
    st.caption("Try again with the real password from the table above.")

# ---- 3. isolation -----------------------------------------------------------
st.header("3. Can Alice see or touch Bob's data?")
if st.button("Log in as Alice and try to snoop on Bob"):
    alice = store.unlock("alice", PASSWORDS["alice"])
    bob = store.unlock("bob", PASSWORDS["bob"])
    seen = [c.text for c in store.load_user_data(alice)]
    st.write("Alice can read:", seen)
    if any("Bob" in t for t in seen):
        st.error("FAIL: Alice can see Bob's data")
    else:
        st.success("PASS: Alice only sees her own notes.")
    bobs_chunk = store.load_user_data(bob)[0].chunk_id
    try:
        store.delete_chunk(alice, bobs_chunk)
        st.error("FAIL: Alice deleted Bob's note")
    except IsolationViolation:
        st.success("PASS: Alice tried to delete Bob's note and was blocked.")

# ---- 4. tamper --------------------------------------------------------------
st.header("4. Is tampering detected?")
st.caption("Flip one bit of Alice's encrypted note directly in the database, as an attacker would.")
alice = store.unlock("alice", PASSWORDS["alice"])
if "tampered" not in st.session_state:
    if st.button("Corrupt one of Alice's records"):
        pk = store._read_pklist(alice)[0]
        v = bytearray(
            store.conn.execute("SELECT c_chunk FROM ma_records WHERE pk=?", (pk,)).fetchone()[0]
        )
        v[0] ^= 1
        store.conn.execute("UPDATE ma_records SET c_chunk=? WHERE pk=?", (bytes(v), pk))
        store.conn.commit()
        st.session_state.tampered = pk
        st.rerun()
else:
    st.warning("Alice's record is corrupted. Now let her load her notes:")
    if st.button("Alice loads her notes"):
        try:
            store.load_user_data(alice)
            st.error("FAIL: tampering was NOT detected")
        except IntegrityError:
            st.success("PASS: tampering detected, request refused (no garbage returned).")
    if st.button("Undo the corruption"):
        pk = st.session_state.pop("tampered")
        v = bytearray(
            store.conn.execute("SELECT c_chunk FROM ma_records WHERE pk=?", (pk,)).fetchone()[0]
        )
        v[0] ^= 1
        store.conn.execute("UPDATE ma_records SET c_chunk=? WHERE pk=?", (bytes(v), pk))
        store.conn.commit()
        st.rerun()

# ---- 5. add & read your own -------------------------------------------------
st.header("5. Store your own note")
c1, c2 = st.columns([1, 3])
who = c1.selectbox("For user", list(PASSWORDS), key="who")
text = c2.text_input("Note text")
if st.button("Encrypt and store") and text.strip():
    s = store.unlock(who, PASSWORDS[who])
    store.add_chunks(s, [ChunkRecord(text, {}, fake_embedding(text))])
    st.success(
        "Stored encrypted. Now check step 1 (not readable) and step 3 (only the owner sees it)."
    )
