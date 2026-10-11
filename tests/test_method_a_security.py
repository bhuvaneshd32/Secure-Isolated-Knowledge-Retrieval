"""
Security tests for Method A, mapped to the paper's §4 properties:
  * confidentiality (IND-CPA of AES-CBC)  -> randomised ciphertext, no plaintext at rest
  * integrity (INT-CTXT via HMAC)         -> any bit flip / swap / splice => ⊥
  * isolation                             -> other users' keys cannot open a record
"""

import pathlib

import numpy as np
import pytest

from privrag.store import AuthenticationError, IntegrityError, IsolationViolation
from privrag.store.method_a import MethodAStore

CANARY = "CANARY-777-001"
SECRET_WORDS = "Appointment with Dr. Zebulon Quackenbush regarding lumbago"


@pytest.fixture
def store():
    return MethodAStore(":memory:", kdf_n=2**10)


@pytest.fixture
def alice(store, make_record):
    store.register_user("alice", "alice-password")
    s = store.unlock("alice", "alice-password")
    ids = store.add_chunks(
        s,
        [
            make_record(f"{SECRET_WORDS}. Ref: {CANARY}.", {"date": "2026-01-01"}),
            make_record("second chunk"),
        ],
    )
    return s, ids


def _flip(store, pk, column, offset=0):
    val = store.conn.execute(f"SELECT {column} FROM ma_records WHERE pk=?", (pk,)).fetchone()[0]
    b = bytearray(val)
    b[offset] ^= 0x01
    store.conn.execute(f"UPDATE ma_records SET {column}=? WHERE pk=?", (bytes(b), pk))
    store.conn.commit()


# ------------------------------------------------------------ confidentiality
def test_no_plaintext_in_database_file(tmp_path, make_record):
    path = str(tmp_path / "a.db")
    st = MethodAStore(path, kdf_n=2**10)
    st.register_user("alice", "alice-password")
    s = st.unlock("alice", "alice-password")
    st.add_chunks(
        s, [make_record(f"{SECRET_WORDS}. Ref: {CANARY}.", {"doctor": "Zebulon Quackenbush"})]
    )
    st.conn.commit()
    st.conn.close()
    raw = pathlib.Path(path).read_bytes()
    for needle in (CANARY, "Quackenbush", "lumbago", "alice-password", "Appointment"):
        assert needle.encode() not in raw, f"{needle!r} found in database file"


def test_embedding_not_stored_in_clear(tmp_path, make_record):
    path = str(tmp_path / "a.db")
    st = MethodAStore(path, kdf_n=2**10)
    st.register_user("alice", "alice-password")
    s = st.unlock("alice", "alice-password")
    rec = make_record("x")
    st.add_chunks(s, [rec])
    st.conn.commit()
    st.conn.close()
    raw = pathlib.Path(path).read_bytes()
    assert np.asarray(rec.embedding, dtype="<f4").tobytes()[:64] not in raw


def test_same_plaintext_gives_different_ciphertext(store, make_record):
    store.register_user("alice", "pw-alice-1")
    s = store.unlock("alice", "pw-alice-1")
    r = make_record("identical text")
    store.add_chunks(s, [r, r])
    rows = store.conn.execute("SELECT c_chunk, iv_chunk, c_emb, iv_emb FROM ma_records").fetchall()
    assert rows[0][0] != rows[1][0] and rows[0][1] != rows[1][1]
    assert rows[0][2] != rows[1][2] and rows[0][3] != rows[1][3]


def test_chunk_and_embedding_use_distinct_ivs(alice, store):
    iv_emb, iv_chunk, iv_meta = store.conn.execute(
        "SELECT iv_emb, iv_chunk, iv_meta FROM ma_records LIMIT 1"
    ).fetchone()
    assert len({iv_emb, iv_chunk, iv_meta}) == 3


def test_keys_are_not_stored_in_clear(alice, store):
    s, _ = alice
    k_enc, k_mac = s.key("enc"), s.key("mac")
    for table in ("ma_users", "ma_records"):
        for row in store.conn.execute(f"SELECT * FROM {table}").fetchall():
            for cell in row:
                if isinstance(cell, bytes):
                    assert k_enc not in cell and k_mac not in cell


def test_no_owner_column_in_records(store):
    cols = [r[1] for r in store.conn.execute("PRAGMA table_info(ma_records)")]
    assert not any("owner" in c or "user" in c for c in cols)


# ------------------------------------------------------------------ integrity
@pytest.mark.parametrize(
    "column", ["c_chunk", "c_emb", "c_meta", "iv_chunk", "iv_emb", "iv_meta", "tag"]
)
def test_any_bit_flip_is_detected(alice, store, column):
    s, ids = alice
    _flip(store, ids[0], column)
    with pytest.raises(IntegrityError):
        store.load_user_data(s)


def test_record_swap_between_primary_keys_detected(alice, store):
    s, ids = alice
    r0 = store.conn.execute(
        "SELECT iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta, tag FROM ma_records WHERE pk=?",
        (ids[0],),
    ).fetchone()
    store.conn.execute(
        "UPDATE ma_records SET iv_emb=?, iv_chunk=?, iv_meta=?, c_emb=?, c_chunk=?, "
        "c_meta=?, tag=? WHERE pk=?",
        (*r0, ids[1]),
    )
    store.conn.commit()
    with pytest.raises(IntegrityError):
        store.load_user_data(s)


def test_deleted_record_detected(alice, store):
    s, ids = alice
    store.conn.execute("DELETE FROM ma_records WHERE pk=?", (ids[0],))
    store.conn.commit()
    with pytest.raises(IntegrityError):
        store.load_user_data(s)


def test_pklist_tamper_detected(alice, store):
    s, _ = alice
    val = store.conn.execute("SELECT pk_cipher FROM ma_users WHERE user_id='alice'").fetchone()[0]
    b = bytearray(val)
    b[0] ^= 1
    store.conn.execute("UPDATE ma_users SET pk_cipher=? WHERE user_id='alice'", (bytes(b),))
    store.conn.commit()
    with pytest.raises(IntegrityError):
        store.load_user_data(s)


def test_wrapped_key_tamper_prevents_login(alice, store):
    val = store.conn.execute("SELECT wrapped_keys FROM ma_users WHERE user_id='alice'").fetchone()[
        0
    ]
    b = bytearray(val)
    b[20] ^= 1
    store.conn.execute("UPDATE ma_users SET wrapped_keys=? WHERE user_id='alice'", (bytes(b),))
    store.conn.commit()
    with pytest.raises(AuthenticationError):
        store.unlock("alice", "alice-password")


# ------------------------------------------------------------------ isolation
def test_other_users_key_cannot_open_foreign_record(alice, store, make_record):
    _, ids = alice
    store.register_user("mallory", "mallory-password")
    m = store.unlock("mallory", "mallory-password")
    row = store.conn.execute(
        "SELECT pk, iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta, tag "
        "FROM ma_records WHERE pk=?",
        (ids[0],),
    ).fetchone()
    with pytest.raises(IntegrityError):
        store._open_record(m, row)


def test_attacker_injecting_foreign_pk_into_own_list_fails(alice, store, make_record):
    """Even if a PK of Alice's is forced into Mallory's PKlist, MAC under K_mac(mallory) fails."""
    _, ids = alice
    store.register_user("mallory", "mallory-password")
    m = store.unlock("mallory", "mallory-password")
    iv, ct, tag = store._seal_pklist("mallory", m.key("enc"), m.key("mac"), [ids[0]])
    store.conn.execute(
        "UPDATE ma_users SET pk_iv=?, pk_cipher=?, pk_tag=? WHERE user_id='mallory'", (iv, ct, tag)
    )
    store.conn.commit()
    with pytest.raises(IntegrityError):
        store.load_user_data(m)


def test_mallory_cannot_delete_alices_chunk(alice, store):
    _, ids = alice
    store.register_user("mallory", "mallory-password")
    m = store.unlock("mallory", "mallory-password")
    with pytest.raises(IsolationViolation):
        store.delete_chunk(m, ids[0])


def test_session_without_keys_rejected(store):
    from privrag.store import UserSession

    store.register_user("alice", "alice-password")
    with pytest.raises(AuthenticationError):
        store.load_user_data(UserSession("alice"))


def test_closed_session_cannot_load(alice, store):
    s, _ = alice
    s.close()
    with pytest.raises(AuthenticationError):
        store.load_user_data(s)


def test_session_repr_hides_keys(alice):
    s, _ = alice
    assert s.key("enc").hex() not in repr(s)


def test_persistence_across_reopen(tmp_path, make_record):
    path = str(tmp_path / "p.db")
    st = MethodAStore(path, kdf_n=2**10)
    st.register_user("alice", "alice-password")
    s = st.unlock("alice", "alice-password")
    st.add_chunks(s, [make_record("persisted")])
    st.conn.close()
    st2 = MethodAStore(path)
    s2 = st2.unlock("alice", "alice-password")
    assert [c.text for c in st2.load_user_data(s2)] == ["persisted"]


def test_large_user_over_in_clause_limit(store, make_record):
    store.register_user("big", "big-password")
    s = store.unlock("big", "big-password")
    ids = store.add_chunks(s, [make_record(f"chunk {i}") for i in range(1100)])
    loaded = store.load_user_data(s)
    assert [c.chunk_id for c in loaded] == ids
