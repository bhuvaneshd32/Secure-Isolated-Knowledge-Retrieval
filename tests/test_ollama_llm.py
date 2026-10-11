import pytest
import requests

from privrag.rag.generator import EchoLLM, LLMError, OllamaLLM, get_llm


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._p


def test_default_llm_is_ollama(monkeypatch):
    monkeypatch.delenv("PRIVRAG_LLM", raising=False)
    assert isinstance(get_llm(), OllamaLLM)
    assert isinstance(get_llm("echo"), EchoLLM)
    with pytest.raises(ValueError):
        get_llm("gpt-nope")


def test_generate_sends_chat_request_and_parses(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen.update(url=url, body=json, timeout=timeout)
        return FakeResp({"message": {"role": "assistant", "content": "  Monday at 10.  "}})

    monkeypatch.setattr(requests, "post", fake_post)
    llm = OllamaLLM(model="m1", host="http://h:1/")
    assert llm.generate("PROMPT") == "Monday at 10."
    assert seen["url"] == "http://h:1/api/chat"
    assert seen["body"]["model"] == "m1" and seen["body"]["stream"] is False
    assert seen["body"]["options"]["temperature"] == 0.0
    assert seen["body"]["messages"][-1] == {"role": "user", "content": "PROMPT"}
    assert "injection" not in seen["body"]["messages"][0]["content"].lower()


def test_connection_error_becomes_llmerror(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(LLMError):
        OllamaLLM().generate("x")


def test_malformed_reply_becomes_llmerror(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp({"oops": 1}))
    with pytest.raises(LLMError):
        OllamaLLM().generate("x")


def test_available_probe(monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: FakeResp({"models": [{"name": "llama3:instruct"}]})
    )
    assert OllamaLLM(model="llama3:instruct").available()[0] is True
    assert OllamaLLM(model="llama3").available()[0] is True
    ok, msg = OllamaLLM(model="mistral").available()
    assert ok is False and "pull" in msg

    def down(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(requests, "get", down)
    assert OllamaLLM().available()[0] is False


def test_api_returns_503_when_llm_down(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from privrag.api import Settings, create_app
    from privrag.rag.ingest import HashEmbedder
    from privrag.store.method_a import MethodAStore

    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    app = create_app(
        store=MethodAStore(str(tmp_path / "x.db"), kdf_n=2**10),
        embedder=HashEmbedder(),
        llm=OllamaLLM(),
        settings=Settings(privrag_public_kb=str(tmp_path / "n.jsonl")),
    )
    c = TestClient(app)
    c.post("/register", json={"user_id": "a", "password": "password-1"})
    tok = c.post("/login", data={"username": "a", "password": "password-1"}).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    c.post("/documents", headers=h, json=[{"text": "hello"}])
    r = c.post("/query", headers=h, json={"question": "hi"})
    assert r.status_code == 503 and r.json() == {"detail": "Language model unavailable"}
