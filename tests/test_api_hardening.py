import logging

import pytest
from fastapi.testclient import TestClient

from privrag.api import MAX_BODY, RedactingFilter, Settings, app, create_app
from privrag.rag.generator import EchoLLM
from privrag.rag.ingest import HashEmbedder
from privrag.store.method_a import MethodAStore

client = TestClient(app)


def test_redacting_filter_drops_canary():
    log_filter = RedactingFilter()

    # Record containing a canary string must be filtered out
    bad_record = logging.LogRecord(
        name="privrag",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Processing query containing CANARY-001-005 in text",
        args=(),
        exc_info=None,
    )
    assert log_filter.filter(bad_record) is False

    # Normal record must pass through
    good_record = logging.LogRecord(
        name="privrag",
        level=logging.INFO,
        pathname=__file__,
        lineno=20,
        msg="Processing query for hash 8f4b2c1d",
        args=(),
        exc_info=None,
    )
    assert log_filter.filter(good_record) is True


def test_request_logging_no_canary_leak(caplog):
    # Attach our filter to caplog handler during the test run
    logger = logging.getLogger("privrag.api")
    logger.addFilter(RedactingFilter())

    canary_token = "CANARY-999-001"

    with caplog.at_level(logging.INFO):
        # Trigger an endpoint with a payload holding a canary
        client.post(
            "/documents",
            headers={"Authorization": "Bearer non-existent-token"},
            json=[{"text": f"Confidential meeting: {canary_token}", "metadata": {}}],
        )

        # Explicitly log a message with the canary through the filtered logger
        logger.info(f"Received document with token {canary_token}")

    # Assert that no canary string leaked into the captured logs
    assert "CANARY" not in caplog.text


def test_payload_size_limit_middleware():
    # Simulate a payload header exceeding the 2MB threshold
    large_size = (2 * 1024 * 1024) + 1
    response = client.post("/documents", headers={"Content-Length": str(large_size)}, json=[])
    assert response.status_code == 413
    assert response.json()["detail"] == "Payload Too Large"


# -- request size and login throttling through the real app ---------------------------
@pytest.fixture
def api(tmp_path):
    app = create_app(
        store=MethodAStore(str(tmp_path / "h.db"), kdf_n=2**10),
        embedder=HashEmbedder(),
        llm=EchoLLM(),
        settings=Settings(privrag_public_kb=str(tmp_path / "none.jsonl")),
    )
    return app


def _token(c, uid="alice", pw="password-123"):
    c.post("/register", json={"user_id": uid, "password": pw})
    r = c.post("/login", data={"username": uid, "password": pw})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_chunked_request_without_length_is_refused(api):
    c = TestClient(api)

    def body():
        yield b'[{"text": "x"}]'

    r = c.post("/documents", content=body(), headers={"Authorization": "Bearer t"})
    assert r.status_code == 411


def test_oversized_upload_is_refused(api):
    c = TestClient(api)
    h = _token(c)
    big = b"a" * (MAX_BODY + 10)
    r = c.post("/documents/upload", files={"file": ("big.txt", big)}, headers=h)
    assert r.status_code == 413


def test_upload_just_under_the_limit_is_accepted(api):
    c = TestClient(api)
    h = _token(c)
    r = c.post("/documents/upload", files={"file": ("ok.txt", b"hello world")}, headers=h)
    assert r.status_code == 200


def test_attacker_address_cannot_lock_out_the_real_user(api):
    attacker = TestClient(api, client=("6.6.6.6", 1111))
    owner = TestClient(api, client=("5.5.5.5", 2222))
    _token(owner, "victim", "correct-password")
    for _ in range(5):
        attacker.post("/login", data={"username": "victim", "password": "wrong-password"})
    assert (
        attacker.post(
            "/login", data={"username": "victim", "password": "correct-password"}
        ).status_code
        == 429
    )
    ok = owner.post("/login", data={"username": "victim", "password": "correct-password"})
    assert ok.status_code == 200
