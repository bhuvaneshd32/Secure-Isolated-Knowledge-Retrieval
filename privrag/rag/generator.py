"""
LLM backends.

OllamaLLM  : the real model (local Ollama server). Default everywhere.
EchoLLM    : a deterministic, deliberately vulnerable test double ("worst-case"
             model that dumps its whole prompt on "repeat ... context"). Only
             used when explicitly requested (PRIVRAG_LLM=echo / unit tests).
"""

import json
import logging
import os
import warnings
from typing import Protocol
from urllib.parse import urlparse

from privrag.rag.retriever import RetrievedChunk


# The LLM Protocol
class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


def build_prompt(question: str, context_chunks: list[RetrievedChunk]) -> str:
    """Delimits context chunks with <context id=...> blocks as specified."""
    prompt = "Answer the question based only on the provided context.\n\n"

    for chunk in context_chunks:
        prompt += f"<context id={chunk.chunk_id}>\n{chunk.text}\n</context>\n\n"

    prompt += f"Question: {question}\nAnswer:"
    return prompt


class EchoLLM(LLM):
    """
    Deterministic worst-case test double (not used unless explicitly selected).
    """

    name = "echo"

    def __init__(self):
        # The specific trigger phrases defined in the workplan
        self.injection_phrases = ["repeat", "copy", "verbatim", "all the context"]

    def generate(self, prompt: str) -> str:
        # Extract the user's question from the prompt template
        question_part = prompt.split("Question:")[-1].lower()

        # Check if the user is attempting a prompt injection
        is_injection = any(phrase in question_part for phrase in self.injection_phrases)

        if is_injection:
            # LEAK: The AI ignores instructions and dumps the raw context
            return f"[ECHO LEAK VERBATIM]:\n{prompt}"

        # NORMAL BEHAVIOR: Answer safely from the first chunk
        if "<context id=" in prompt:
            # Extract just the text from the first context block
            first_chunk_text = (
                prompt.split("<context id=")[1].split("</context>")[0].split(">\n")[-1].strip()
            )
            return f"Based on the documents: {first_chunk_text}"

        return "I don't have enough context to answer that."


log = logging.getLogger(__name__)

# Hosts where the prompt (decrypted chunks) stays on this machine.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
CONNECT_TIMEOUT = 5.0  # fail fast when the server is down


class LLMError(RuntimeError):
    """The language model backend could not produce an answer."""


class OllamaLLM(LLM):
    """
    Local model served by Ollama (https://ollama.com). The prompt never leaves
    the machine, which matters here: it contains the user's decrypted chunks.

    Env: OLLAMA_HOST (default http://localhost:11434), OLLAMA_MODEL (default llama3:instruct),
    OLLAMA_TIMEOUT (seconds without a new streamed chunk, default 300), OLLAMA_NUM_PREDICT (cap
    on generated tokens, unset = model default), OLLAMA_NUM_CTX (context window, default 4096),
    OLLAMA_KEEP_ALIVE (how long the model stays loaded, default 30m).

    The reply is streamed, so `timeout` is per chunk: a slow model that keeps producing
    tokens is not cut off, only a stalled or still-loading one is.

    The system message only asks for grounded answers. It deliberately has NO
    anti-injection instructions, so that isolation (not prompt wording) is what
    prevents cross-user leakage in the evaluation.
    """

    name = "ollama"
    SYSTEM = "You are a helpful assistant. Answer the question using only the provided context."

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float | None = None,
        temperature: float = 0.0,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3:instruct")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = float(timeout if timeout is not None else os.getenv("OLLAMA_TIMEOUT", 300))
        self.temperature = temperature
        env_cap = os.getenv("OLLAMA_NUM_PREDICT")
        self.num_predict = num_predict if num_predict is not None else (int(env_cap) if env_cap else None)
        self.num_ctx = int(num_ctx if num_ctx is not None else os.getenv("OLLAMA_NUM_CTX", 4096))
        self.keep_alive = keep_alive or os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.last_prompt_tokens: int | None = None  # prompt size of the last reply

        if urlparse(self.host).hostname not in _LOCAL_HOSTS:
            warnings.warn(
                f"OLLAMA_HOST={self.host} is not local: prompts, which contain decrypted "
                f"chunks, will leave this machine.",
                stacklevel=2,
            )

    def available(self) -> tuple[bool, str]:
        """Probe the server and check that the model is pulled."""
        import requests

        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except (requests.RequestException, ValueError):
            return False, f"Ollama not reachable at {self.host} (run `ollama serve`)"
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            return False, f"Model '{self.model}' not pulled (run `ollama pull {self.model}`)"
        return True, f"Ollama ready: {self.model}"

    def warm_up(self) -> bool:
        """Load the model now so the first question does not pay the load time. Never raises."""
        import requests

        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "keep_alive": self.keep_alive},
                timeout=(CONNECT_TIMEOUT, self.timeout),
            )
            r.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def generate(self, prompt: str) -> str:
        import requests

        options: dict = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if self.num_predict:
            options["num_predict"] = self.num_predict
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": True,
                    "keep_alive": self.keep_alive,
                    "options": options,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=(CONNECT_TIMEOUT, self.timeout),
                stream=True,
            ) as r:
                r.raise_for_status()
                return self._read_stream(r)
        except requests.exceptions.ReadTimeout:
            raise LLMError(
                f"Ollama sent nothing for {self.timeout:.0f} s (ReadTimeout). The model is "
                f"probably still loading or stalled; raise OLLAMA_TIMEOUT, or use a smaller "
                f"model / cap the length with OLLAMA_NUM_PREDICT."
            ) from None
        except (requests.RequestException, KeyError, ValueError) as e:
            raise LLMError(
                f"Ollama request failed ({type(e).__name__}); "
                f"is `ollama serve` running and `{self.model}` pulled?"
            ) from None

    def _read_stream(self, r) -> str:
        """Concatenate the streamed chat chunks (one JSON object per line)."""
        parts: list[str] = []
        self.last_prompt_tokens = None
        for line in r.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                raise LLMError(f"Ollama error: {obj['error']}")
            parts.append(obj["message"]["content"])
            if obj.get("done"):
                self.last_prompt_tokens = obj.get("prompt_eval_count")
                break
        else:
            raise LLMError("Ollama stream ended before the answer was complete.")
        if self.last_prompt_tokens is not None and self.last_prompt_tokens >= self.num_ctx:
            log.warning(
                "Prompt used %d tokens, the whole context window (num_ctx=%d): it was probably "
                "truncated. Lower top-k or raise OLLAMA_NUM_CTX.",
                self.last_prompt_tokens,
                self.num_ctx,
            )
        return "".join(parts).strip()


def get_llm(kind: str | None = None) -> LLM:
    """kind: 'ollama' (default) | 'echo' (test double). Env: PRIVRAG_LLM."""
    kind = (kind or os.getenv("PRIVRAG_LLM", "ollama")).lower()
    if kind == "echo":
        return EchoLLM()
    if kind == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM '{kind}'. Expected 'echo' or 'ollama'")
