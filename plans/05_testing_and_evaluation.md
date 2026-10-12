# 5. Testing and evaluation

## 5.1 Automated tests: `python -m pytest tests` → 147 passed

`tests/conftest.py` pins the environment (hash embedder, `PRIVRAG_LLM=echo`, temp DB, no public KB), so no test
needs Ollama or the internet. Per-file coverage is in [03_codebase_walkthrough.md](03_codebase_walkthrough.md) §3.9.

The shared contract (`test_store_contract.py`) runs the same 10 behaviours on plaintext, Method A, Method B-paper
and Method B-hardened: register/unlock, duplicate user, wrong password, unknown user, add + load round trip
(text, metadata, `owner`, float32 embedding equality), survives re-login, cross-user isolation, delete own
(idempotent), cross-user delete → `IsolationViolation`, embedding required.

## 5.2 End-to-end self-check: `python -m evaluation.selfcheck [--store A|B_paper|B_hardened]` (also the UI "Proof" tab)

Runs through embedder → store → retriever → real LLM for Method A or Method B. With Ollama `llama3:instruct`, Method A: **9/9 passed**. Method B has the same checks (check 7 adapts to the mode: AES-GCM tag for hardened, `H(K_i)` field for paper) plus one more: *a leaked node key cannot open older records*. Without the LLM (`--skip-llm`): A 7/7, B-paper 8/8, B-hardened 8/8.

| # | Check | Observed |
|---|---|---|
| 1 | Wrong password rejected | `AuthenticationError` |
| 2 | DB holds no plaintext | 300 records scanned, none found |
| 3 | Decrypt round trip | 30 chunks equal originals, 384-d embeddings |
| 4 | Retrieval finds the right appointment | gold chunk rank 1 |
| 5 | LLM context never has foreign data | 4 injection prompts, 0 foreign items |
| 6 | Cannot delete another user's chunk | `IsolationViolation` |
| 7 | Tampered ciphertext rejected | 1-bit flip → `IntegrityError` |
| 8 | Injection through the real LLM leaks nothing foreign | 0 foreign canaries |
| 9 | LLM answers from the decrypted chunk | "…appointment with Dr. Rebecca Henderson is on 2025-01-25." |

On Windows set `PYTHONIOENCODING=utf-8` if a symbol fails to print.

## 5.3 Leakage harness: `python -m evaluation.leakage [--llm ollama|echo] [--users N] [--prompts N]`

For each user × injection prompt, counts **other** users' canaries in the LLM prompt and in its answer, for the
naive shared KB, the isolated plaintext store, Method A and Method B (hardened).

| Run | naive | plaintext-isolated | Method A | Method B |
|---|---|---|---|---|
| real Ollama, 1 user × 2 prompts (prompt / answer) | 16 / 16 | 0 / 0 | 0 / 0 | 0 / 0 |

The full-run results (all 10 users × 4 prompts at the prompt level; 3 users × 2 prompts through a real LLM) are in
[09_evaluation_report.md](09_evaluation_report.md); use `--users` and `--prompts` to scale this harness.

## 5.4 Retrieval parity: `python -m evaluation.quality`

30 queries (3 per user, mixed types), MiniLM, pool = user's 30 chunks + 500 SciFact abstracts.

| Store | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|
| plaintext | 0.9333 | 0.6848 | 0.7449 |
| method_a | 0.9333 | 0.6848 | 0.7449 |
| method_b_paper | 0.9333 | 0.6848 | 0.7449 |
| method_b_hardened | 0.9333 | 0.6848 | 0.7449 |

The script asserts identical ranked lists across stores. The absolute numbers are a property of the dataset and
embedder (MiniLM separates similar record templates poorly), not of encryption. With the earlier 40-sentence fallback KB the same queries scored MRR@10 0.634 / nDCG@10 0.707: those hand-written sentences about appointments competed with the gold records, which the SciFact abstracts do not.

Attack check on this pool (MiniLM, all 10 users × 4 prompts, no LLM): the naive shared KB put foreign secrets into 39 of 40 prompts (198 secrets); plaintext-isolated, Method A and Method B put none.

## 5.5 Key exposure: `python -m evaluation.key_exposure --position 20`

`user_000` owns 30 records and one key leaks at chain position 20 (together with the DB):

| | Records the attacker reads |
|---|---|
| Method A | 30 / 30 |
| Method B | 10 / 30 (records 20..29) |

## 5.6 Performance: `python -m evaluation.perf` (random embeddings, this laptop)

| Store | Ingest chunks/s | Bytes/chunk on disk | p50 ms @ 100 / 500 / 1000 / 2000 | p95 ms @ 2000 |
|---|---|---|---|---|
| Plaintext | 40 435 | 2 114 | 0.6 / 3.1 / 6.8 / 14.5 | 22.0 |
| Method A | 10 931 | 2 140 | 5.8 / 28.7 / 59.0 / 113.2 | 124.1 |
| Method B (paper) | 9 666 | 4 168 | 7.5 / 37.0 / 74.1 / 148.8 | 163.0 |
| Method B (hardened) | 21 370 | 2 114 | 2.5 / 12.5 / 25.3 / 53.6 | 70.5 |

Notes:
- Latency is `load_user_data` (verify + decrypt everything) plus retrieval; it excludes embedding the query and the
  LLM call, which dominate real latency.
- The B(paper) file size is about 2× larger only because of SQLite page packing: its rows are slightly over half of
  the 4 096-byte page (payload is 2 112 B/chunk vs 2 015 B for hardened). It is not extra data.
- B(hardened) is faster than A most likely because it decrypts one AES-GCM message per node, while A and B-paper do
  three or five separate CBC decryptions per record (not profiled). Run-to-run noise is large: an earlier run of the
  same script on a quieter machine gave A = 61 ms and plaintext = 6.8 ms at 2 000 chunks.
- Files: `evaluation/results/perf_results.csv`, `latency_scaling.png`, `storage_overhead.png`.

## 5.7 Other verification

| Check | Result |
|---|---|
| Method B with real Ollama, both modes, 1 question | correct answer, gold chunk ranked first |
| Streamlit `AppTest` (sign in, Ask with 3 columns, key-leak and tamper demos), echo LLM | no exceptions |
| Live uvicorn + `deploy/smoke_test.py` (Method A) | passed (earlier run) |
| Streamlit UI in a real browser against live Ollama | not done |
| Docker image build / compose up | not done |

## 5.8 Full evaluation to CSV: `python -m evaluation.run_all`

One command evaluates **plaintext, Method A, Method B (paper) and Method B (hardened)** on identical data and
identical, pre-computed embeddings, and writes CSV files for analysis to
`evaluation/results/full_run_<timestamp>/`. The vulnerable shared knowledge base (`naive_shared`) is added as a
reference row in the leakage results.

| File | One row per | Notes |
|---|---|---|
| `leakage.csv` | system × user × injection prompt | foreign secrets in the LLM **prompt** for all users and prompts (no LLM needed, deterministic); foreign secrets in the **answer** for a sample (`--llm-users`, `--llm-prompts`) that calls the model |
| `qa.csv` | system × query | gold rank, hit@1/3/5/10, retrieval ms, LLM answer, `answer_correct` (fact match per record type) |
| `retrieval.csv` | system | Recall@10, MRR@10, nDCG@10, hit@k, ranking parity and tie-proof score parity with plaintext |
| `key_exposure.csv` | system × user × leaked-key position | records readable out of N |
| `tamper.csv` | system × attack | `flip_ciphertext`, `flip_iv`, `swap_rows`, `delete_row`, `rollback`: detected, silent modification, silent data loss or silent rollback |
| `perf.csv` | system × corpus size | p50/p95 latency, ingest rate, bytes per chunk |
| `summary.csv` | system | one aggregated row each |
| `run_info.json` | run | arguments, model, embedder, counts, timings, platform |

Options: `--quick` (smoke run, a few minutes), `--skip-llm` (no model calls), `--llm echo`, `--users`, `--prompts`,
`--llm-users`, `--llm-prompts`, `--qa-queries`, `--perf-queries`, `--skip-perf`, `--reuse-llm` (answer identical
prompts once), `--llm-timeout S`, `--llm-max-tokens N`, `--llm-retries N`, `--public-kb PATH`, `--out DIR`,
`--resume DIR`. A model call that still fails after its retries is recorded in the `llm_error` column and left out
of the answer metrics instead of ending the run. `summary.csv` counts them (`llm_attacks_failed`, `qa_llm_failed`) and the dashboard
shows a banner with the totals and a per-system breakdown; the charts ignore them. Rows are appended as they are produced, so an interrupted run keeps its
results and `--resume` continues it.

Default size and expected time on a CPU with `llama3:instruct` (about 30 s for a long injection answer, a few seconds
for a short QA answer): 5 systems × 3 users × 2 prompts = 30 attack calls plus 4 systems × 30 queries = 120 QA calls,
so roughly 30 to 40 minutes in total; without the LLM it takes a few minutes.

View the results with `streamlit run demo/results_dashboard.py` (pick the run in the sidebar; partial runs work).

Interpretation notes:
- `flip_ciphertext` on Method B (paper) is reported as *detected* only because the garbled block fails to decode or
  pad; it is not authenticated, and the IV flip next to it shows the silent modification.
- `rollback` is silently accepted by every system (known issue H1).
- With the hash embedder (tests) many scores tie and tied chunks can swap places between stores, so
  `score_parity_with_plaintext` is the tie-proof measure; with MiniLM both parities are 1.0.
