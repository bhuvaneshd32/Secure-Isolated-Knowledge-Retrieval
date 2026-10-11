import io
import logging

import pytest
from fastapi.testclient import TestClient

from privrag.api import Settings, create_app, install_log_redaction
from privrag.rag.generator import EchoLLM
from privrag.rag.ingest import DocumentLoader, HashEmbedder, TextChunker
from privrag.rag.public_kb import load_public_kb
from privrag.sessions import SessionManager
from privrag.store.plaintext import PlaintextStore


@pytest.fixture
def client(tmp_path):
    app = create_app(
        PlaintextStore(":memory:"),
        HashEmbedder(),
        EchoLLM(),
        Settings(privrag_public_kb=str(tmp_path / "none.jsonl"), privrag_session_file=""),
    )
    c = TestClient(app, raise_server_exceptions=False)
    c.post("/register", json={"user_id": "u", "password": "password1"})
    tok = c.post("/login", data={"username": "u", "password": "password1"}).json()["access_token"]
    c.headers["Authorization"] = f"Bearer {tok}"
    return c


def test_upload_bad_or_empty_files_are_client_errors(client):
    assert client.post("/documents/upload", files={"file": ("a.txt", b"\xff\xfe\x00")}).status_code == 422
    assert client.post("/documents/upload", files={"file": ("a.pdf", b"not a pdf")}).status_code == 422
    assert client.post("/documents/upload", files={"file": ("a.txt", b"  \n")}).status_code == 422
    ok = client.post("/documents/upload", files={"file": ("a.txt", b"hello world")})
    assert ok.status_code == 200 and len(ok.json()["inserted_ids"]) == 1


def test_upload_never_writes_a_temp_file(client, monkeypatch):
    import tempfile

    def boom(*a, **k):
        raise AssertionError("temp file created")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", boom)
    assert client.post("/documents/upload", files={"file": ("a.txt", b"x y z")}).status_code == 200


def test_load_bytes_roundtrip():
    assert DocumentLoader.load_bytes("héllo".encode(), ".md") == "héllo"


def test_chunker_splits_long_paragraph_with_overlap():
    text = " ".join(f"w{i}" for i in range(3000))  # one 15k-char paragraph
    ch = TextChunker(chunk_size=1200, overlap=200).chunk(text)
    assert len(ch) > 10 and all(len(c) <= 1300 for c in ch)
    assert ch[0].split()[-1] in ch[1]  # overlap carries the boundary word into the next chunk
    assert TextChunker().chunk("") == []


def test_lockout_window_resets():
    t = [0.0]
    sm = SessionManager(clock=lambda: t[0], max_failures=3, lockout_seconds=60)
    for _ in range(3):
        sm.record_failure("u", "c")
    with pytest.raises(Exception):
        sm.check_not_locked("u", "c")
    t[0] = 100
    sm.check_not_locked("u", "c")
    sm.record_failure("u", "c")  # one failure after the window must not re-lock
    sm.check_not_locked("u", "c")


def test_log_redaction_applies_to_handlers_of_child_loggers():
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    root = logging.getLogger()
    root.addHandler(h)
    try:
        install_log_redaction()
        child = logging.getLogger("privrag.ingest")
        child.setLevel(logging.INFO)
        child.info("secret CANARY-001-001")
        child.info("fine")
    finally:
        root.removeHandler(h)
    assert "CANARY" not in buf.getvalue() and "fine" in buf.getvalue()


def test_public_kb_cache_follows_content(tmp_path):
    p = tmp_path / "kb.jsonl"
    emb = HashEmbedder()
    p.write_text('{"chunk_id": "a", "text": "alpha beta"}\n', encoding="utf-8")
    e1, _ = load_public_kb(emb, p)
    p.write_text('{"chunk_id": "a", "text": "gamma delta"}\n', encoding="utf-8")  # same count
    e2, _ = load_public_kb(emb, p)
    assert not (e1 == e2).all()
