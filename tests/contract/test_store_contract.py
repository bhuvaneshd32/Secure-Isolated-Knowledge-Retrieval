import pytest
import numpy as np
from privrag.store import (
    ChunkRecord, AuthenticationError, IntegrityError, IsolationViolation
)
from privrag.store.plaintext import PlaintextStore

# Parametrize allows Person 1 to easily add MethodAStore and MethodBStore to this list later.
@pytest.fixture(params=[PlaintextStore])
def store(request):
    # Using an in-memory database for fast, isolated tests
    return request.param(db_path=":memory:")

def test_register_and_unlock(store):
    store.register_user("alice", "pass123")
    session = store.unlock("alice", "pass123")
    assert session.user_id == "alice"

def test_wrong_password_raises_auth_error(store):
    store.register_user("bob", "securepass")
    with pytest.raises(AuthenticationError):
        store.unlock("bob", "wrongpass")

def test_add_and_load_round_trip(store):
    store.register_user("charlie", "pass")
    session = store.unlock("charlie", "pass")
    
    chunks_in = [ChunkRecord(text="Project meeting at 10 AM", metadata={"source": "notes.txt"})]
    chunk_ids = store.add_chunks(session, chunks_in)
    
    loaded = store.load_user_data(session)
    assert len(loaded) == 1
    assert loaded[0].chunk_id == chunk_ids[0]
    assert loaded[0].text == "Project meeting at 10 AM"
    assert loaded[0].metadata["source"] == "notes.txt"
    assert isinstance(loaded[0].embedding, np.ndarray)

def test_cross_user_isolation(store):
    store.register_user("userA", "passA")
    sessionA = store.unlock("userA", "passA")
    store.add_chunks(sessionA, [ChunkRecord(text="User A Secret", metadata={})])
    
    store.register_user("userB", "passB")
    sessionB = store.unlock("userB", "passB")
    
    # B should never see A's chunks
    loadedB = store.load_user_data(sessionB)
    assert len(loadedB) == 0

def test_delete_isolation_violation(store):
    store.register_user("hacker", "badpass")
    store.register_user("victim", "goodpass")
    
    sessionV = store.unlock("victim", "goodpass")
    ids = store.add_chunks(sessionV, [ChunkRecord(text="Victim Data", metadata={})])
    target_id = ids[0]
    
    sessionH = store.unlock("hacker", "badpass")
    with pytest.raises(IsolationViolation):
        store.delete_chunk(sessionH, target_id)