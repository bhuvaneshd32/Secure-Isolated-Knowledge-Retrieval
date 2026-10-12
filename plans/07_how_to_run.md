# 7. How to run

All commands from the repository root, PowerShell. Python 3.11+.

## 7.1 Install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first use of the MiniLM embedder downloads `sentence-transformers/all-MiniLM-L6-v2` (internet needed once).

## 7.2 Local LLM (Ollama)

```powershell
ollama pull llama3:instruct     # or any chat model; then set OLLAMA_MODEL
ollama serve                    # skip if the Ollama app is already running
```

## 7.3 Build the knowledge base (once)

```powershell
python build_kb.py                       # data + embeddings + encrypted Method A and B stores, about 40 s
python build_kb.py --users 5 --chunks 20 --public-limit 100     # smaller build
```

This generates the synthetic corpus, builds the public KB, embeds everything and writes `.demo/method_a.db`,
`.demo/method_b_base.db`, `.demo/method_b_improved.db`, `.demo/plaintext.db` and `.demo/manifest.json`. Delete
`.demo/` to reset. The data scripts can also be run on their own:

```powershell
python data/make_dataset.py              # 300 chunks + 30 queries (reproducible)
python data/make_public_kb.py --limit 500       # BEIR SciFact via Hugging Face; --limit 5183 for all
```

## 7.4 Demo UI

```powershell
streamlit run demo/app.py
```

Sign in as `user_000` … `user_009` with password `demo-password`.

| Tab | What to try |
|---|---|
| Ask | `When is my appointment with Dr. <name>?`, `What was my HbA1c result?`; the four prompt-injection presets. Left: naive system (leaks). Middle: Method A. Right: Method B (sidebar switches `improved` / `base`). |
| My documents | your decrypted records (read-only; the dataset is fixed) |
| Public knowledge base | the shared documents |
| Key leak & tampering | slider: one leaked key, how many records each method exposes; button: IV bit-flip on A, B, B improved |
| Proof | self-checks for Method A, Method B or both (9 and 10 checks; optional LLM checks) |

Each question calls Ollama three times (naive, A, B), so expect 1 to 1.5 minutes on a CPU. Set
`PRIVRAG_LLM=echo` for an instant, deterministic (deliberately leaky) test model.

## 7.5 Tests and evaluation

```powershell
python -m pytest tests                                  # 147 tests, no Ollama or internet needed
$env:PYTHONIOENCODING = "utf-8"                         # Windows: scripts print the symbol for bottom
python -m evaluation.selfcheck                          # Method A checks (uses Ollama)
python -m evaluation.selfcheck --store B_improved       # Method B checks (also B); --skip-llm is faster
python -m evaluation.quality                            # parity across the 4 stores
python -m evaluation.key_exposure --position 20         # one leaked key, A vs B
python -m evaluation.leakage --users 1 --prompts 2      # real LLM, quick
python -m evaluation.leakage --llm echo                 # full set, seconds
python -m evaluation.perf                               # throughput, latency, storage -> CSV + PNGs
python -m evaluation.run_all                            # everything, all 4 systems -> CSV folder (30-40 min)
python -m evaluation.run_all --quick                    # same pipeline, small and fast
streamlit run demo/results_dashboard.py                 # charts and tables from the CSV folder
python -m evaluation.run_all --resume evaluation\results\full_run_<timestamp>   # continue an interrupted run
```

Exit codes of `selfcheck` / `leakage`: 0 pass, 1 failure, 2 baseline did not leak (harness invalid), 3 Ollama
unavailable.

## 7.6 API

```powershell
$env:PRIVRAG_STORE = "B_improved"     # plaintext | A | B | B_improved
$env:PRIVRAG_LLM   = "ollama"         # ollama | echo
uvicorn privrag.api:app --port 8000
# Swagger UI: http://localhost:8000/docs      second terminal:
python deploy/smoke_test.py
```

Flow: `POST /register` → `POST /login` (form `username`, `password`) → `Authorization: Bearer <token>` →
`POST /documents` → `POST /query`.

## 7.7 Docker (not verified)

```powershell
docker compose -f deploy/docker-compose.yml up --build
```

## 7.8 Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PRIVRAG_STORE` | `A` | `plaintext`, `A`, `B`, `B_improved` |
| `PRIVRAG_DB_PATH` | `data/privrag_<store>.db` | SQLite file |
| `PRIVRAG_EMBEDDER` | `auto` | `minilm`, `hash`, `auto` |
| `PRIVRAG_LLM` | `ollama` | `ollama` or `echo` |
| `OLLAMA_MODEL` | `llama3:instruct` | model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_TIMEOUT` | `300` | seconds to wait for one answer (long "repeat the context" answers are slow on a CPU) |
| `OLLAMA_NUM_PREDICT` | unset | cap on generated tokens; unset = the model's default |
| `PRIVRAG_PUBLIC_KB` | `data/public_kb.jsonl` | public KB file |
| `PRIVRAG_SESSION_TTL` | `1800` | seconds |
| `PRIVRAG_SESSION_SECRET` | unset | set a long random value to persist sessions across restarts (AES-GCM file); unset keeps keys in memory only |
| `PRIVRAG_SESSION_FILE` | `data/sessions.enc` | where persisted sessions are written |
| `PRIVRAG_DEMO_DIR` | `.demo` | where `build_kb.py` and the UI keep the demo databases |

## 7.9 Troubleshooting

| Symptom | Cause / fix |
|---|---|
| UI says the knowledge base has not been built | run `python build_kb.py` (also needed after upgrading, to create the Method B databases) |
| `build_kb.py` fails with `PermissionError` on `.demo/*.db` | the UI is still running; stop Streamlit first |
| `ModuleNotFoundError: privrag` | run from the repository root |
| UI shows a red "Ollama not reachable" | start `ollama serve`, or set `PRIVRAG_LLM=echo` |
| Model not pulled | `ollama pull <model>` or set `OLLAMA_MODEL` |
| `ReadTimeout` from Ollama (also in `run_all`) | the answer took longer than the timeout: raise `OLLAMA_TIMEOUT` (or `run_all --llm-timeout 600`), cap length with `--llm-max-tokens 600`, make sure no other model is using the CPU, then `--resume` |
| API `/query` → 503 | Ollama down or model missing |
| API `/query` → generic 500 after manual DB edits | integrity check failed (by design) |
| `make_public_kb.py` says Hugging Face datasets failed | `pip install datasets` and check internet access to huggingface.co; the BEIR zip and then a 40-sentence fallback are tried next |
| Login returns 429 | 5 failed attempts for that user id from your address; wait 60 s |
| `selfcheck` crashes printing a symbol | `$env:PYTHONIOENCODING = "utf-8"` |
