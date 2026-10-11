import numpy as np
import pytest

from privrag.store import AuthenticationError, IntegrityError, IsolationViolation
from privrag.store.method_a import MethodAStore
from privrag.store.method_b import MethodBStore
from privrag.store.plaintext import PlaintextStore

# Every store implementation must pass the identical contract.
FACTORIES = {
    "plaintext": lambda: PlaintextStore(":memory:"),
    "method_a": lambda: MethodAStore(":memory:", kdf_n=2**10),  # low scrypt cost for test speed
    "method_b_paper": lambda: MethodBStore(":memory:", mode="paper", kdf_n=2**10),
    "method_b_hardened": lambda: MethodBStore(":memory:", mode="hardened", kdf_n=2**10),
}


@pytest.fixture(params=list(FACTORIES))
def store(request):
    return FACTORIES[request.param]()


def test_register_and_unlock(store):
    store.register_user("alice", "pass123")
    session = store.unlock("alice", "pass123")
    assert session.user_id == "alice"


def test_duplicate_registration_rejected(store):
    store.register_user("alice", "pass123")
    with pytest.raises(IntegrityError):
        store.register_user("alice", "other")


def test_wrong_password_raises_auth_error(store):
    store.register_user("bob", "securepass")
    with pytest.raises(AuthenticationError):
        store.unlock("bob", "wrongpass")


def test_unknown_user_raises_auth_error(store):
    with pytest.raises(AuthenticationError):
        store.unlock("nobody", "x")


def test_add_and_load_round_trip(store, make_record):
    store.register_user("charlie", "pass")
    session = store.unlock("charlie", "pass")

    rec = make_record("Project meeting at 10 AM", {"source": "notes.txt"})
    chunk_ids = store.add_chunks(session, [rec])

    loaded = store.load_user_data(session)
    assert len(loaded) == 1
    assert loaded[0].chunk_id == chunk_ids[0]
    assert loaded[0].text == "Project meeting at 10 AM"
    assert loaded[0].metadata["source"] == "notes.txt"
    assert loaded[0].metadata["owner"] == "charlie"
    assert isinstance(loaded[0].embedding, np.ndarray)
    assert loaded[0].embedding.dtype == np.float32
    np.testing.assert_array_equal(loaded[0].embedding, rec.embedding)


def test_data_survives_relogin(store, make_record):
    store.register_user("dana", "pw")
    s1 = store.unlock("dana", "pw")
    store.add_chunks(s1, [make_record("first"), make_record("second")])
    s2 = store.unlock("dana", "pw")
    assert [c.text for c in store.load_user_data(s2)] == ["first", "second"]


def test_cross_user_isolation(store, make_record):
    store.register_user("userA", "passA")
    sessionA = store.unlock("userA", "passA")
    store.add_chunks(sessionA, [make_record("User A Secret")])

    store.register_user("userB", "passB")
    sessionB = store.unlock("userB", "passB")

    assert store.load_user_data(sessionB) == []
    assert len(store.load_user_data(sessionA)) == 1


def test_delete_own_chunk(store, make_record):
    store.register_user("erin", "pw")
    s = store.unlock("erin", "pw")
    ids = store.add_chunks(s, [make_record("a"), make_record("b")])
    store.delete_chunk(s, ids[0])
    assert [c.text for c in store.load_user_data(s)] == ["b"]
    store.delete_chunk(s, ids[0])  # idempotent


def test_delete_isolation_violation(store, make_record):
    store.register_user("hacker", "badpass")
    store.register_user("victim", "goodpass")

    sessionV = store.unlock("victim", "goodpass")
    ids = store.add_chunks(sessionV, [make_record("Victim Data")])

    sessionH = store.unlock("hacker", "badpass")
    with pytest.raises(IsolationViolation):
        store.delete_chunk(sessionH, ids[0])
    assert len(store.load_user_data(sessionV)) == 1  # victim's data untouched


def test_embedding_is_required(store):
    from privrag.store import ChunkRecord

    store.register_user("frank", "pw")
    s = store.unlock("frank", "pw")
    with pytest.raises(ValueError):
        store.add_chunks(s, [ChunkRecord(text="x", metadata={})])
