# 6. Known issues, weaknesses and unverified areas

Severity: **High** = weakens a security claim; **Medium** = real weakness or gap; **Low** = polish.

## 6.1 High

| # | Issue | Detail | Fix direction |
|---|---|---|---|
| H1 | **No rollback / replay protection (A and B)** | An attacker with DB write access can restore an *older* valid row, PKlist, node or trapdoor; every MAC still verifies. The paper does not address this either. | Per-user version counter inside the MACed PKlist / trapdoor, or a Merkle root |
| H2 | **Method B (paper mode) has no ciphertext authentication** | Only `H(K_i)` is checked, so flipping an IV bit silently alters the first block of a field (demonstrated in the UI and in `test_paper_mode_detects_wrong_key_but_not_iv_malleation`). The next address is in clear. | Use `hardened` mode; the paper-faithful mode is kept for comparison |
| H3 | **Server holds keys while a user is logged in** | `UserSession` keeps `K_i`/`K_mac` (A) or masterkey/keysalt (B) in process memory; a memory dump of the server exposes logged-in users. Outside the paper's threat model. | Client-side decryption, or short TTLs and key zeroisation |

## 6.2 Medium

| # | Issue | Detail |
|---|---|---|
| M1 | Public KB is a 500-abstract slice of SciFact | Enough to exercise the private ∪ public union realistically, but the topic (scientific claims) differs from the synthetic medical records. The full corpus (5 183) is one `--limit` away. |
| M2 | Answer-level leakage sampled, not exhaustive | `run_all` (see [09](09_evaluation_report.md)) checks all 10 users × 4 prompts at the prompt level, but calls the real LLM for only 3 users × 2 prompts (6 per system). Use `--llm-users 10 --llm-prompts 4` for the full 40. |
| M3 | Query cost is O(n) in the user's KB (A and B) | Every query verifies and decrypts all of a user's rows; B additionally walks the chain sequentially. Measured: ~113 ms (A) and ~149 ms (B-paper) at 2 000 chunks, excluding embedding and LLM. |
| M4 | B appends and deletes walk the whole chain | As in the paper (Eq. 23, 24); no tail pointer is kept. |
| M5 | Switching embedders corrupts retrieval silently | Stored embeddings come from the embedder active at ingest; a different `PRIVRAG_EMBEDDER` still yields 384-d vectors. The embedder name is not stored with the data. |
| M6 | Login throttling is per (user id, client address) | **Mitigated**: a stranger cannot lock a user out from another address. Behind a reverse proxy every client looks like the proxy unless it forwards the real address; an attacker with many addresses can still guess slowly (scrypt cost also throttles). State is in memory. |
| M7 | Sessions are in-process memory by default | **Mitigated, opt-in**: `PRIVRAG_SESSION_SECRET` persists sessions in an AES-GCM file. Trade-off: whoever holds both the file and the secret recovers logged-in users' keys. Several uvicorn workers would still need a shared file or store. |
| M8 | Upload size | **Fixed**: chunked requests without a length get 411, and uploads are read in 64 KiB chunks and stopped at the 2 MB limit. |
| M9 | Retriever top-k boundary ties are arbitrary | `argpartition` chooses which tied items fall inside the top-k; the later sort only orders the selection. Exact ties are rare (random chunk ids). |
| M10 | Chunker is simplistic | `overlap` is accepted but unused; a paragraph larger than `chunk_size` is kept whole. |
| M11 | Docker artefacts never built | `deploy/` was edited but not built or run. |
| M12 | UI never driven in a real browser against live Ollama | Verified with Streamlit `AppTest` (echo LLM) and the same backend calls with Ollama. |
| M13 | Side channels remain | Ciphertext length reveals text length; `delete_chunk` distinguishes "exists but not yours" (403) from "unknown" (200); registration says "User already exists"; `/query` latency depends on KB size; B-paper exposes chain structure through the clear `next_addr`. |
| M14 | Method B delete is not in the paper | Behaviour (unlink, head rewrite) is ours and only tested empirically. |

## 6.3 Low

| # | Issue | Detail |
|---|---|---|
| L1 | Plaintext baseline stores passwords in clear | Intentional; it is the insecure comparison system. |
| L2 | Password rules only in the API | Min 8 chars in `/register`; the stores accept any non-empty password. |
| L3 | No password change, key rotation or account deletion | The wrapped-key design would allow password change without re-encrypting; not implemented. |
| L4 | `.demo/` is rebuilt, not migrated | `build_kb.py` deletes `.demo/*.db` first; a stale demo DB from before Method B makes the UI ask for a rebuild. |
| L5 | Public-KB embedding cache key ignores content | `data/public_kb.<embedder>.<n>.npy` is keyed by embedder and document count; `build_kb.py` deletes stale caches. |
| L6 | `/health` does not check Ollama or the DB | Always `ok`. |
| L7 | Prompt injection through *document content* is not mitigated | Only cross-user leakage is in scope. |
| L8 | `EchoLLM` splits on the literal string `"Question:"` | Test double only. |
| L9 | Dependencies are unpinned | `requirements.txt` has no versions. |
| L10 | Single SQLite connection with a global lock | Fine for the demo; no pooling or WAL. |
| L11 | No TLS, CORS or security headers | Run behind a reverse proxy for anything non-local. |
| L13 | Result files are tied to one commit | The full run in `evaluation/results/full_run_20261012_093421/` was committed in `2907c22`; re-run `run_all` after changing store or retrieval code. |

## 6.4 Claims that are **not** established

- "Method A / B is secure" in the formal sense: only empirical tests were run; the paper's IND-CPA / INT-CTXT
  reductions were not re-derived, and the implementation differs from the paper in the ways listed in
  [04](04_crypto_design.md) §4.3.
- "RAG quality is preserved on a realistic corpus": parity was shown on a 300-chunk synthetic dataset with a
  500-abstract SciFact slice, not on a large or real clinical corpus.
- "Real LLMs never leak": isolation means the LLM never *receives* foreign data; a real model will still repeat a
  user's *own* data, by design.
- The paper's §4 "forward security" statement for Method B: it does not hold as written (see
  [04](04_crypto_design.md) §4.3). What holds is the backward protection shown by `key_exposure`.

## 6.5 Checked and **not** issues

- Foreign rows cannot be decrypted with another user's keys (A: `test_other_users_key_cannot_open_foreign_record`;
  B: `test_users_have_independent_chains`).
- Client-supplied `metadata["owner"]` cannot spoof ownership: both stores overwrite it after copying the dict.
- Bad padding, MAC, JSON or non-bytes cells all surface as `IntegrityError`, never as a 500 with details.
- Each encryption uses a fresh random IV or nonce.
- The DB files contain none of: canary tokens, doctor names, text fragments, the password, the raw keys (A and B).
- Private embeddings are never cached on disk in clear (the embedder has no cache; only the public KB is cached).
