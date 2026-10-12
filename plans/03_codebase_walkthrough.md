# 3. Codebase walkthrough (file by file)

Paths are relative to the repository root.

```
privrag/              the library
  crypto.py           cryptographic building blocks
  store/              storage back-ends behind one interface (plaintext, Method A, Method B)
  rag/                chunking, embedding, retrieval, prompt, LLM, pipeline
  sessions.py         token sessions, TTL, lockout
  api.py              FastAPI service
data/                 dataset generators + generated data
evaluation/           self-check, leakage, quality, key exposure, performance (+ results/)
demo/                 Streamlit UIs
deploy/               Dockerfile, compose, smoke test
tests/                147 tests
plans/                this documentation
build_kb.py           one-shot build: data, embeddings, encrypted stores, manifest
```

## 3.1 `privrag/crypto.py`

Thin, tested wrapper over `cryptography`. The paper's ⊥ is modelled as `CryptoError`.

| Function | Purpose | Paper |
|---|---|---|
| `generate_key`, `generate_iv`, `random_bytes` | `os.urandom`; keys 32 B (λ = 256), IV 16 B | Eq. 5-7 |
| `pad`, `unpad` | PKCS#7; `unpad` raises unless the last `k` bytes equal `k` | Eq. 9 |
| `aes_cbc_encrypt`, `aes_cbc_decrypt` | AES-256-CBC with padding | Eq. 7, 8, 14, 19 |
| `hmac_sha256`, `hmac_verify` | HMAC over length-prefixed parts; constant-time compare | Eq. 10, 13, 27 |
| `scrypt_derive`, `hkdf` | password stretching; key derivation / expansion | key wrapping, Eq. 16, 22 |
| `sha256`, `xor_bytes` | `H(.)` and the trapdoor XOR | Eq. 3, 17 |
| `aes_gcm_encrypt`, `aes_gcm_decrypt` | authenticated encryption for Method B improved | beyond paper |

## 3.2 `privrag/store/`

| File | Contents |
|---|---|
| `__init__.py` | the contract: `ChunkRecord`, `DecryptedChunk`, `UserSession`, `PrivateStore` protocol, exceptions `AuthenticationError`, `IntegrityError`, `IsolationViolation` |
| `plaintext.py` | `PlaintextStore`: everything in clear, ownership by an `owner` column; also `dump_all()` for the naive baseline |
| `method_a.py` | `MethodAStore(db_path, kdf_n)`: Method A |
| `method_b.py` | `MethodBStore(db_path, mode="improved"\|"base", kdf_n)`: Method B |
| `factory.py` | `make_store(kind)` for `plaintext`, `A`, `B`, `B_improved`; unknown names raise (no silent fallback to plaintext) |

### `MethodAStore`
| Member | Behaviour |
|---|---|
| `register_user` | samples `K_i`, `K_mac`, wraps them under a scrypt key, creates an empty encrypted PKlist; duplicate → `IntegrityError` |
| `unlock` | unwraps the keys; wrong password → `AuthenticationError` (unknown users cost one scrypt call) |
| `add_chunks` | one transaction: verify PKlist, insert records, rewrite PKlist |
| `load_user_data` | PKlist → rows (batches of 500) → verify tag → decrypt; any failure → `IntegrityError` |
| `delete_chunk` | own PK deletes; foreign existing PK → `IsolationViolation`; unknown → no-op |
| `raw_records`, `record_count` | what a DBA sees |

### `MethodBStore`
| Member | Behaviour |
|---|---|
| `register_user` | scrypt → masterkey, keysalt; `K_1`; writes the trapdoor (head = Null) |
| `unlock` | un-masks the trapdoor and runs `VerifyID`; session holds `master` and `keysalt` |
| `add_chunks` | walks to the tail, appends nodes (key chain), patches the tail pointer or the trapdoor |
| `load_user_data` | Algorithm 1; checks `H(K_i)` per node; cycles, missing nodes and bad cells → `IntegrityError` |
| `delete_chunk` | unlinks a node (head / middle / tail / only node) |
| `raw_records`, `record_count` | forensics |
| `read_forward_with_key(user_id, key)` | attacker model for the key-exposure experiment: scan for the node that key opens, follow the chain forward |

## 3.3 `privrag/rag/`

| File | Contents |
|---|---|
| `ingest.py` | `DocumentLoader` (txt/md/pdf), `TextChunker`, `Embedder` (MiniLM, 384-d, **no on-disk cache** for private text), `HashEmbedder` (offline fallback), `get_embedder` |
| `retriever.py` | `Retriever(public_embs, public_meta).retrieve(query, user_id, private_chunks, top_k)`: firewall on `metadata["owner"]`, pool = public ∪ private, cosine, top-k with deterministic ties |
| `generator.py` | `build_prompt`, `OllamaLLM` (default, `llama3:instruct`), `EchoLLM` (deliberately leaky test double), `get_llm` |
| `pipeline.py` | `RagService(store, embedder, retriever, llm)`: `add_texts`, `ingest_document`, `retrieve`, `prompt_for`, `answer`; `NaiveSharedRAG` baseline |
| `public_kb.py` | `load_public_kb`: reads `data/public_kb.jsonl`, embeds once, caches the `.npy` |
| `seed.py` | `load_jsonl`, `DEMO_PASSWORD`, `seed_store(service, corpus)` |

## 3.4 `privrag/sessions.py`, `privrag/api.py`

`SessionManager`: token (stored only as its SHA-256) -> `(UserSession, expiry)`, TTL 1800 s; expired or revoked
sessions drop their keys. Login throttling is per **(user id, client address)** (5 failures -> 60 s) plus a per-client
cap of 20 failures across all user ids, so a stranger cannot lock a user out from another address. Sessions are
in memory only by default; setting `PRIVRAG_SESSION_SECRET` stores them in one AES-GCM encrypted file
(`PRIVRAG_SESSION_FILE`, default `data/sessions.enc`) so they survive a restart.

`create_app(store, embedder, llm, settings)`: endpoints `/health`, `/register`, `/login`, `/logout`, `/documents`
(POST/GET), `/documents/upload`, `/documents/{id}` (DELETE), `/query`. The store comes from `PRIVRAG_STORE`
(`plaintext`, `A`, `B`, `B_improved`). Hardening: 2 MB `Content-Length` limit, chunked requests without a length refused (411), uploads read in
64 KiB chunks and stopped at the limit, log filter dropping canaries,
generic 500 for integrity failures, 401/403/429/503 mapping.

## 3.5 `data/`

| File | Purpose |
|---|---|
| `make_dataset.py` | Faker (seed 42, fixed 2025 date range) → `private_corpus.jsonl` (300 chunks) and `private_queries.jsonl` (30 queries, mixed types; the first per user is an appointment query because `selfcheck` needs a date) |
| `make_public_kb.py` | BEIR SciFact corpus, first `--limit` (default 500) abstracts: tries Hugging Face `datasets` (`BeIR/scifact`), then the BEIR zip, then 40 built-in medical sentences |
| `*.jsonl`, `public_kb.*.npy` | generated data; the `.npy` embedding caches are git-ignored |

## 3.6 `evaluation/`

| File | Purpose |
|---|---|
| `selfcheck.py` | end-to-end checks for Method A (9) and Method B (10 with LLM; Method B adds a leaked-key check); `--store A\|B\|B_improved`, `--skip-llm`; also the UI "Proof" tab |
| `leakage.py` | injection attack on every user for naive / isolated plaintext / Method A / Method B; options `--users`, `--prompts`, `--llm` |
| `quality.py` | Recall/MRR/nDCG@10 for plaintext, A, B, B improved; asserts identical rankings |
| `run_all.py` | the full evaluation of all systems in one command, written to CSV (see [05](05_testing_and_evaluation.md) §5.8) |
| `key_exposure.py` | one leaked key: how many records A and B expose |
| `perf.py` | ingest throughput, bytes/chunk, p50/p95 latency vs corpus size for all four stores |
| `results/` | CSVs and PNGs written by the scripts |

## 3.7 `demo/`

| File | Purpose |
|---|---|
| `app.py` | main UI: sign in, **Ask** (naive vs A vs B, three columns, each calling Ollama), **My documents** (read-only), **Public knowledge base**, **Key leak & tampering** (key-exposure slider and tamper demo), **Proof** (self-check). Needs `python build_kb.py` first. |
| `simple_app.py` | small 5-step Method A tester (disk view, login, isolation, tamper) on an in-memory store |
| `results_dashboard.py` | Streamlit dashboard for the CSV folder written by `evaluation/run_all.py`: leakage, key-leak blast radius, tamper matrix, retrieval quality, QA by record type, latency / storage, summary; a table view under every chart |

## 3.8 `build_kb.py`, `deploy/`

`build_kb.py` generates data, builds the public KB, embeds it, embeds and encrypts the private corpus into
`.demo/method_a.db`, `.demo/method_b_base.db`, `.demo/method_b_improved.db`, loads the plaintext baseline, and
writes `.demo/manifest.json` (about 20 s).

`deploy/`: `Dockerfile`, `docker-compose.yml`, `smoke_test.py` (health → register → login → add → query). Not built.

## 3.9 `tests/` (147 tests)

| File | Tests | Covers |
|---|---|---|
| `test_crypto_primitives.py` | 14 | AES-CBC vectors, padding, HMAC, HKDF vector, scrypt |
| `test_store_contract.py` | 40 | 10 behaviours × {plaintext, A, B, B improved} |
| `test_method_a_security.py` | 25 | at-rest, tamper, swap, isolation, persistence |
| `test_method_b_security.py` | 29 | at-rest, chain edits, tamper, truncation, trapdoor, key exposure |
| `test_api_method_a.py` | 8 | API flow with injection, auth failures, upload, tamper |
| `test_ollama_llm.py` | 6 | Ollama client (HTTP mocked), 503 when LLM down |
| `test_sessions.py` | 9 | TTL, revoke, per-client lockout, opt-in encrypted persistence |
| `test_ingest.py`, `test_retriever.py` | 3 + 2 | chunker, firewall and ranking |
| `test_run_all.py` | 2 | the full-evaluation script: CSVs written, key invariants hold, `--resume` adds no duplicates |
| `test_api_hardening.py` | 7 | log filter, size limit, chunked request refused, oversized upload, lockout from another address |
| `conftest.py` | — | environment pinning (hash embedder, echo LLM, temp DB), `make_record` |
