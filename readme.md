# Secure and Isolated Knowledge Retrieval

A privacy-aware Retrieval-Augmented Generation (RAG) system with cryptographic per-user data isolation,
implementing *Privacy-Aware RAG: Secure and Isolated Knowledge Retrieval* (arXiv 2503.15548).

| | Status |
|---|---|
| Method A (AES-CBC + HMAC, `privrag/store/method_a.py`) | implemented and tested |
| Method B (chained key derivation, `privrag/store/method_b.py`; `paper` and `hardened` modes) | implemented and tested |
| LLM | local Ollama (default) |

Documentation (architecture, code walkthrough, crypto design, tests, known issues, roadmap):
**[plans/README.md](plans/README.md)**.

## Quick start (Windows PowerShell, from the repo root)

```powershell
# 1. Python deps
python -m venv venv ; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Local LLM (once)
ollama pull llama3:instruct        # any chat model works: set OLLAMA_MODEL
ollama serve                       # skip if the Ollama app is already running

# 3. Build once (~40 s): generate data, embed public + private, encrypt with Method A and Method B
python build_kb.py

# 4. Demo UI (starts instantly; only your question is embedded at runtime)
streamlit run demo/app.py

# 5. Checks from the command line
python -m pytest tests                              # 147 tests, no Ollama needed
python -m evaluation.selfcheck                      # end-to-end Method A checks with a real LLM answer
python -m evaluation.selfcheck --store B_hardened   # the same for Method B (also B_paper)
python -m evaluation.quality                        # retrieval parity: plaintext = A = B
python -m evaluation.key_exposure --position 20     # one leaked key: Method A opens 30/30, Method B 10/30
python -m evaluation.leakage --users 1 --prompts 2  # prompt-injection attack, naive vs A vs B (real LLM)
python -m evaluation.run_all                        # full evaluation of all 4 systems -> CSV files (30-40 min)
```

In the UI, sign in with `user_000` / `demo-password` (any `user_00N`):

- **Ask** shows three answers side by side: the naive shared system (leaks other users' data under the
  prompt-injection presets), Method A and Method B (they do not). The sidebar switches the Method B variant.
- **A vs B: key leak** shows where B is better (one leaked key exposes fewer records) and where paper-mode B is
  worse (an IV bit-flip goes undetected; A and hardened B detect it).
- **My documents**, **Public knowledge base**, **Under the hood** (what a DB thief sees) and **Proof** (self-checks for Method A, Method B or both).

`demo/simple_app.py` is a small step-by-step Method A tester. After `python -m evaluation.run_all`, `streamlit run demo/results_dashboard.py` shows the evaluation CSVs as charts and tables. The dataset is fixed; it is built by `build_kb.py`.

## Running Ollama on a laptop GPU

`ollama ps` should show `100% GPU`. If it shows a CPU share (the model plus context does not fit in VRAM), generation
becomes very slow and `run_all` can hit `ReadTimeout`.

```powershell
$env:OLLAMA_NUM_PARALLEL = "1"          # set before `ollama serve`, avoids reserving extra KV space
$env:OLLAMA_MODEL = "llama3.2:3b"       # optional: a 3B model fits a 6 GB GPU entirely
python -m evaluation.run_all --llm-max-tokens 256 --llm-timeout 600   # these are the defaults
python -m evaluation.run_all --resume evaluation/results/full_run_<timestamp>   # continue after a failure
```

`run_all` caps generated tokens at 256 and waits up to 600 s per request by default; override with `--llm-max-tokens`,
`--llm-timeout` or `OLLAMA_NUM_PREDICT` / `OLLAMA_TIMEOUT`.

## API

```powershell
$env:PRIVRAG_STORE = "B_hardened"   # plaintext | A | B_paper | B_hardened
$env:PRIVRAG_LLM   = "ollama"       # ollama | echo (leaky test double)
uvicorn privrag.api:app --port 8000      # docs at http://localhost:8000/docs
python deploy/smoke_test.py              # in a second terminal
```

Endpoints: `/register`, `/login`, `/logout`, `/documents` (POST/GET), `/documents/upload`,
`/documents/{id}` (DELETE), `/query`, `/health`.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PRIVRAG_STORE` | `A` | `plaintext`, `A`, `B_paper`, `B_hardened` |
| `PRIVRAG_LLM` | `ollama` | `ollama` or `echo` |
| `OLLAMA_MODEL` / `OLLAMA_HOST` | `llama3:instruct` / `http://localhost:11434` | local model server |
| `PRIVRAG_EMBEDDER` | `auto` | `minilm` (sentence-transformers), `hash` (no download), `auto` |
| `PRIVRAG_DB_PATH` | `data/privrag_<store>.db` | SQLite file |
| `PRIVRAG_PUBLIC_KB` | `data/public_kb.jsonl` | public knowledge base |
| `PRIVRAG_SESSION_SECRET` | unset | set to persist sessions across API restarts (encrypted file); unset = in memory only |

## Layout

```
privrag/
  crypto.py       AES-CBC, AES-GCM, PKCS7, HMAC, HKDF, scrypt, SHA-256
  sessions.py     token sessions, TTL, lockout
  api.py          FastAPI app
  store/          plaintext baseline, method_a.py, method_b.py, factory
  rag/            ingest (chunk/embed), retriever, generator (Ollama/Echo), pipeline
evaluation/       selfcheck, leakage, quality parity, key exposure, performance (+ results/)
demo/             app.py (main Streamlit UI), simple_app.py (Method A tester)
deploy/           Dockerfile, compose, smoke test
data/             dataset + public KB builders and generated JSONL
tests/            crypto, store contract, Method A / B security, API
build_kb.py       one-shot build of data, embeddings and encrypted stores
plans/            documentation
```
