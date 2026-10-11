import pytest

from privrag.sessions import SessionManager
from privrag.store import AuthenticationError, UserSession


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_token_roundtrip_and_expiry():
    clk = Clock()
    sm = SessionManager(ttl_seconds=10, clock=clk)
    s = UserSession("alice", {"enc": b"e" * 32, "mac": b"m" * 32})
    tok = sm.create(s)
    assert sm.get(tok) is s
    clk.t = 11
    assert sm.get(tok) is None
    with pytest.raises(AuthenticationError):
        s.key("enc")  # keys dropped on expiry


def test_revoke_drops_keys():
    sm = SessionManager()
    s = UserSession("alice", {"enc": b"e" * 32})
    tok = sm.create(s)
    assert sm.revoke(tok) is True
    assert sm.get(tok) is None
    assert sm.revoke(tok) is False


def test_tokens_are_unique_and_unguessable():
    sm = SessionManager()
    toks = {sm.create(UserSession("u")) for _ in range(50)}
    assert len(toks) == 50 and all(len(t) >= 40 for t in toks)


def test_lockout_after_failures_and_reset():
    clk = Clock()
    sm = SessionManager(max_failures=3, lockout_seconds=30, clock=clk)
    for _ in range(3):
        sm.record_failure("alice")
    with pytest.raises(AuthenticationError):
        sm.check_not_locked("alice")
    clk.t = 31
    sm.check_not_locked("alice")
    sm.record_success("alice")
    sm.check_not_locked("alice")


# -- lockout is per (user, client) ------------------------------------------
def test_attacker_cannot_lock_victim_out_from_another_client():
    sm = SessionManager(max_failures=3, lockout_seconds=30, clock=Clock())
    for _ in range(3):
        sm.record_failure("victim", client="10.0.0.66")
    with pytest.raises(AuthenticationError):
        sm.check_not_locked("victim", client="10.0.0.66")  # the attacker is locked...
    sm.check_not_locked("victim", client="10.0.0.5")  # ...the real user is not


def test_one_client_guessing_many_accounts_is_capped():
    sm = SessionManager(max_failures=3, max_client_failures=6, lockout_seconds=30, clock=Clock())
    for i in range(6):
        sm.record_failure(f"user{i}", client="10.0.0.66")
    with pytest.raises(AuthenticationError):
        sm.check_not_locked("some-new-user", client="10.0.0.66")
    sm.check_not_locked("some-new-user", client="10.0.0.5")


# -- opt-in encrypted persistence --------------------------------------------
def _keys():
    return {"enc": b"E" * 32, "mac": b"M" * 32}


def test_sessions_survive_restart_when_persistence_is_enabled(tmp_path):
    path = str(tmp_path / "sessions.enc")
    a = SessionManager(ttl_seconds=600, persist_path=path, secret="server-secret")
    token = a.create(UserSession("alice", _keys()))

    b = SessionManager(ttl_seconds=600, persist_path=path, secret="server-secret")  # "restart"
    restored = b.get(token)
    assert restored is not None and restored.user_id == "alice"
    assert restored.key("enc") == b"E" * 32 and restored.key("mac") == b"M" * 32


def test_persisted_file_holds_no_token_or_key_in_clear(tmp_path):
    path = tmp_path / "sessions.enc"
    sm = SessionManager(persist_path=str(path), secret="server-secret")
    token = sm.create(UserSession("alice", _keys()))
    raw = path.read_bytes()
    for needle in (token.encode(), b"alice", b"E" * 32, (b"E" * 32).hex().encode()):
        assert needle not in raw


def test_revoke_and_expiry_are_persisted(tmp_path):
    path = str(tmp_path / "sessions.enc")
    clk = Clock()
    a = SessionManager(ttl_seconds=10, persist_path=path, secret="s", clock=clk)
    t_revoked = a.create(UserSession("alice", _keys()))
    t_expiring = a.create(UserSession("bob", _keys()))
    a.revoke(t_revoked)
    clk.t = 11
    b = SessionManager(ttl_seconds=10, persist_path=path, secret="s", clock=clk)
    assert b.get(t_revoked) is None
    assert b.get(t_expiring) is None


def test_wrong_secret_or_corrupt_file_yields_no_sessions(tmp_path):
    path = tmp_path / "sessions.enc"
    token = SessionManager(persist_path=str(path), secret="right").create(
        UserSession("alice", _keys())
    )
    assert SessionManager(persist_path=str(path), secret="wrong").get(token) is None
    path.write_bytes(b"PRS1" + b"\x00" * 40)
    assert SessionManager(persist_path=str(path), secret="right").get(token) is None


def test_without_a_secret_nothing_is_written(tmp_path):
    path = tmp_path / "sessions.enc"
    SessionManager(persist_path=str(path)).create(UserSession("alice", _keys()))
    assert not path.exists()
