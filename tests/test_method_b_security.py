"""Method B (chained key derivation): chain semantics, tamper behaviour, key exposure."""

import pytest

from privrag.store import AuthenticationError, IntegrityError
from privrag.store.method_b import MethodBStore

SECRET = "Prescription on 2025-05-01: Zebulon-Quackenbush metformin CANARY-777-001"
BOTH = pytest.mark.parametrize("mode", ["paper", "hardened"])


def make(mode, path=":memory:"):
    return MethodBStore(path, mode=mode, kdf_n=2**10)


def fill(store, make_record, n=6, user="alice"):
    store.register_user(user, "pw-" + user)
    s = store.unlock(user, "pw-" + user)
    ids = store.add_chunks(s, [make_record(f"note number {i}") for i in range(n)])
    return s, ids


# ---------------------------------------------------------------- at rest
@BOTH
def test_no_plaintext_on_disk(mode, tmp_path, make_record):
    path = str(tmp_path / "b.db")
    st = make(mode, path)
    st.register_user("alice", "alice-password")
    s = st.unlock("alice", "alice-password")
    st.add_chunks(s, [make_record(SECRET, {"doctor": "Quackenbush"})])
    st.conn.commit()
    st.conn.close()
    with open(path, "rb") as f:
        raw = f.read()
    for needle in (b"CANARY", b"Quackenbush", b"metformin", b"alice-password", b"Prescription"):
        assert needle not in raw


@BOTH
def test_same_plaintext_gives_different_ciphertext(mode, make_record):
    st = make(mode)
    s, _ = fill(st, make_record, n=0)
    st.add_chunks(s, [make_record("same text")])
    st.add_chunks(s, [make_record("same text")])
    rows = st.raw_records()
    key = "c_chunk" if mode == "paper" else "blob"
    assert rows[0][key] != rows[1][key]


@BOTH
def test_wrong_password_reveals_nothing(mode, make_record):
    st = make(mode)
    fill(st, make_record)
    with pytest.raises(AuthenticationError):
        st.unlock("alice", "not-the-password")


# ------------------------------------------------------------ chain logic
@BOTH
def test_append_after_relogin_keeps_order(mode, make_record):
    st = make(mode)
    s, _ = fill(st, make_record, n=3)
    s2 = st.unlock("alice", "pw-alice")
    st.add_chunks(s2, [make_record("note number 3")])
    assert [c.text for c in st.load_user_data(s2)] == [f"note number {i}" for i in range(4)]


@BOTH
@pytest.mark.parametrize("victim", [0, 2, 5])  # head, middle, tail
def test_delete_head_middle_tail_then_append(mode, victim, make_record):
    st = make(mode)
    s, ids = fill(st, make_record, n=6)
    st.delete_chunk(s, ids[victim])
    expected = [f"note number {i}" for i in range(6) if i != victim]
    assert [c.text for c in st.load_user_data(s)] == expected
    st.add_chunks(s, [make_record("after delete")])
    assert [c.text for c in st.load_user_data(s)] == expected + ["after delete"]
    # and it all still decrypts after a fresh login
    assert len(st.load_user_data(st.unlock("alice", "pw-alice"))) == 6


@BOTH
def test_delete_everything_then_reuse(mode, make_record):
    st = make(mode)
    s, ids = fill(st, make_record, n=2)
    for i in ids:
        st.delete_chunk(s, i)
    assert st.load_user_data(s) == []
    st.add_chunks(s, [make_record("fresh")])
    assert [c.text for c in st.load_user_data(s)] == ["fresh"]
    assert st.record_count() == 1


@BOTH
def test_users_have_independent_chains(mode, make_record):
    st = make(mode)
    fill(st, make_record, n=3, user="alice")
    s_bob, _ = fill(st, make_record, n=2, user="bob")
    assert len(st.load_user_data(s_bob)) == 2
    assert st.record_count() == 5


# ---------------------------------------------------------------- hardened
def _nodes(st):
    return st.conn.execute("SELECT addr, nonce, blob FROM mb_nodes").fetchall()


def test_hardened_detects_any_flipped_byte(make_record):
    st = make("hardened")
    s, _ = fill(st, make_record, n=3)
    addr, _, blob = _nodes(st)[1]
    for pos in (0, len(blob) // 2, len(blob) - 1):
        bad = bytearray(blob)
        bad[pos] ^= 0x01
        st.conn.execute("UPDATE mb_nodes SET blob = ? WHERE addr = ?", (bytes(bad), addr))
        with pytest.raises(IntegrityError):
            st.load_user_data(s)
    st.conn.execute("UPDATE mb_nodes SET blob = ? WHERE addr = ?", (blob, addr))
    assert len(st.load_user_data(s)) == 3  # restoring the original works


def test_hardened_detects_truncation_and_swapping(make_record):
    st = make("hardened")
    s, ids = fill(st, make_record, n=3)
    saved = st.conn.execute("SELECT * FROM mb_nodes WHERE addr = ?", (ids[2],)).fetchone()
    st.conn.execute("DELETE FROM mb_nodes WHERE addr = ?", (ids[2],))  # drop the tail row
    with pytest.raises(IntegrityError):
        st.load_user_data(s)
    st.conn.execute("INSERT INTO mb_nodes VALUES (?, ?, ?)", saved)
    # swap two rows' ciphertext: the address is bound into the AAD
    a, b = ids[0], ids[1]
    ra = st.conn.execute("SELECT nonce, blob FROM mb_nodes WHERE addr = ?", (a,)).fetchone()
    rb = st.conn.execute("SELECT nonce, blob FROM mb_nodes WHERE addr = ?", (b,)).fetchone()
    st.conn.execute("UPDATE mb_nodes SET nonce=?, blob=? WHERE addr=?", (*rb, a))
    st.conn.execute("UPDATE mb_nodes SET nonce=?, blob=? WHERE addr=?", (*ra, b))
    with pytest.raises(IntegrityError):
        st.load_user_data(s)


def test_hardened_trapdoor_is_authenticated(make_record):
    st = make("hardened")
    s, _ = fill(st, make_record)
    t = bytearray(st.conn.execute("SELECT trapdoor FROM mb_trapdoors").fetchone()[0])
    t[40] ^= 0x01  # inside the K_head part
    st.conn.execute("UPDATE mb_trapdoors SET trapdoor = ?", (bytes(t),))
    with pytest.raises(IntegrityError):
        st.load_user_data(s)
    with pytest.raises(AuthenticationError):
        st.unlock("alice", "pw-alice")


def test_hardened_hides_chain_structure(make_record):
    st = make("hardened")
    fill(st, make_record)
    cols = [d[0] for d in st.conn.execute("SELECT * FROM mb_nodes LIMIT 0").description]
    assert "next_addr" not in cols  # the pointer lives inside the ciphertext


# ------------------------------------------------------------------- paper
def test_paper_mode_detects_wrong_key_but_not_iv_malleation(make_record):
    """
    Documents the paper's Method B weakness (no ciphertext authentication):
    flipping a bit of the IV changes the first plaintext block silently.
    The hardened mode (above) rejects the same attack.
    """
    st = make("paper")
    s, ids = fill(st, make_record, n=2)
    before = [c.text for c in st.load_user_data(s)]
    iv = bytearray(
        st.conn.execute("SELECT iv_chunk FROM mb_nodes WHERE addr = ?", (ids[0],)).fetchone()[0]
    )
    iv[0] ^= 0x01
    st.conn.execute("UPDATE mb_nodes SET iv_chunk = ? WHERE addr = ?", (bytes(iv), ids[0]))
    after = [c.text for c in st.load_user_data(s)]  # no exception: silent modification
    assert after[0] != before[0] and after[1] == before[1]


def test_paper_mode_rejects_corrupted_key_field(make_record):
    st = make("paper")
    s, ids = fill(st, make_record, n=2)
    st.conn.execute("UPDATE mb_nodes SET c_h = ? WHERE addr = ?", (b"\x00" * 32, ids[0]))
    with pytest.raises(IntegrityError):  # H(K_i) check / padding check
        st.load_user_data(s)


# ----------------------------------------------------------- key exposure
@BOTH
def test_leaked_node_key_opens_only_that_node_and_later_ones(mode, make_record):
    st = make(mode)
    s, ids = fill(st, make_record, n=10)
    chain, _ = st._walk(s)
    j = 6
    leaked_key = chain[j][1]
    got = [c.text for c in st.read_forward_with_key("alice", leaked_key)]
    assert got == [f"note number {i}" for i in range(j, 10)]  # 4 of 10, never 0..5


@BOTH
def test_unrelated_key_opens_nothing(mode, make_record):
    st = make(mode)
    fill(st, make_record, n=3)
    assert st.read_forward_with_key("alice", b"\x07" * 32) == []


def test_db_mode_mismatch_is_refused(tmp_path):
    path = str(tmp_path / "m.db")
    make("paper", path).conn.commit()
    with pytest.raises(ValueError):
        MethodBStore(path, mode="hardened", kdf_n=2**10)
