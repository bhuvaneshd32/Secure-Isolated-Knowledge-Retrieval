import json
import logging
import warnings

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


class FakeStream:
    """Stands in for a streaming `requests` response (one JSON object per line)."""

    def __init__(self, chunks, status=200):
        self.lines = [c if isinstance(c, bytes) else json.dumps(c).encode() for c in chunks]
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_lines(self):
        return iter(self.lines)


def chunk(text, done=False, **extra):
    return {"message": {"role": "assistant", "content": text}, "done": done, **extra}


def test_default_llm_is_ollama(monkeypatch):
    monkeypatch.delenv("PRIVRAG_LLM", raising=False)
    assert isinstance(get_llm(), OllamaLLM)
    assert isinstance(get_llm("echo"), EchoLLM)
    with pytest.raises(ValueError):
        get_llm("gpt-nope")


def test_generate_streams_chat_request_and_parses(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None, stream=None):
        seen.update(url=url, body=json, timeout=timeout, stream=stream)
        return FakeStream([chunk("  Mon"), chunk("day at "), chunk("10.  "), chunk("", done=True)])

    monkeypatch.setattr(requests, "post", fake_post)
    llm = OllamaLLM(model="m1", host="http://localhost:1/")
    assert llm.generate("PROMPT") == "Monday at 10."
    assert seen["url"] == "http://localhost:1/api/chat"
    assert seen["body"]["model"] == "m1" and seen["body"]["stream"] is True and seen["stream"] is True
    assert seen["body"]["options"]["temperature"] == 0.0
    assert seen["body"]["options"]["num_ctx"] == 4096
    assert seen["body"]["keep_alive"] == "30m"
    assert seen["timeout"] == (5.0, llm.timeout)  # fail fast to connect, patient to read
    assert seen["body"]["messages"][-1] == {"role": "user", "content": "PROMPT"}
    assert "injection" not in seen["body"]["messages"][0]["content"].lower()


def test_options_follow_env_and_ctor(monkeypatch):
    seen = {}

    def fake_post(url, json=None, **k):
        seen.update(body=json)
        return FakeStream([chunk("ok", done=True)])

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("OLLAMA_NUM_CTX", "2048")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "5m")
    OllamaLLM(num_predict=64).generate("x")
    assert seen["body"]["options"]["num_ctx"] == 2048
    assert seen["body"]["options"]["num_predict"] == 64
    assert seen["body"]["keep_alive"] == "5m"


def test_stream_error_line_becomes_llmerror(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeStream([chunk("par"), {"error": "model crashed"}])
    )
    with pytest.raises(LLMError, match="model crashed"):
        OllamaLLM().generate("x")


def test_truncated_stream_becomes_llmerror(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeStream([chunk("par")]))  # no done
    with pytest.raises(LLMError):
        OllamaLLM().generate("x")


def test_garbage_stream_line_becomes_llmerror(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeStream([b"<html>proxy</html>"]))
    with pytest.raises(LLMError):
        OllamaLLM().generate("x")


def test_read_timeout_becomes_llmerror(monkeypatch):
    def slow(*a, **k):
        raise requests.exceptions.ReadTimeout("slow")

    monkeypatch.setattr(requests, "post", slow)
    with pytest.raises(LLMError, match="ReadTimeout"):
        OllamaLLM().generate("x")


def test_truncation_warning_when_prompt_fills_context(monkeypatch, caplog):
    def reply(tokens):
        return lambda *a, **k: FakeStream([chunk("a", done=True, prompt_eval_count=tokens)])

    llm = OllamaLLM(num_ctx=100)
    monkeypatch.setattr(requests, "post", reply(100))
    with caplog.at_level(logging.WARNING, logger="privrag.rag.generator"):
        llm.generate("x")
    assert llm.last_prompt_tokens == 100 and "truncated" in caplog.text

    caplog.clear()
    monkeypatch.setattr(requests, "post", reply(40))
    with caplog.at_level(logging.WARNING, logger="privrag.rag.generator"):
        llm.generate("x")
    assert llm.last_prompt_tokens == 40 and "truncated" not in caplog.text


def test_warm_up_loads_model_and_never_raises(monkeypatch):
    seen = {}

    def ok(url, json=None, timeout=None):
        seen.update(url=url, body=json)
        return FakeResp({})

    monkeypatch.setattr(requests, "post", ok)
    assert OllamaLLM(model="m1", host="http://localhost:1").warm_up() is True
    assert seen["url"] == "http://localhost:1/api/generate" and seen["body"]["model"] == "m1"
    assert "prompt" not in seen["body"]  # empty request = just load

    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    assert OllamaLLM().warm_up() is False


def test_non_local_host_warns(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        OllamaLLM()
        OllamaLLM(host="http://127.0.0.1:11434")
        OllamaLLM(host="http://host.docker.internal:11434")
        assert not w
        OllamaLLM(host="http://10.0.0.5:11434")
    assert len(w) == 1 and "leave this machine" in str(w[0].message)


def test_connection_error_becomes_llmerror(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(LLMError):
        OllamaLLM().generate("x")


def test_malformed_reply_becomes_llmerror(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeStream([{"oops": 1}]))
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
