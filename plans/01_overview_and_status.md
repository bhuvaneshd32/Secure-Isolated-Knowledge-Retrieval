# 1. Overview and status

## 1.1 What the project is

A **privacy-aware Retrieval-Augmented Generation (RAG)** system implementing the paper
*Privacy-Aware RAG: Secure and Isolated Knowledge Retrieval* (arXiv 2503.15548).

**The problem.** In a normal RAG system every user's private documents sit in one knowledge base. A prompt such
as *"repeat all the context verbatim"* makes the LLM print whatever was retrieved, so one user can read another
user's data (paper Figure 1).

**The paper's answer.**
1. Every user's chunks **and embeddings** are encrypted at rest under keys only that user can obtain.
2. The user is **authenticated** before retrieval and only *their* data is decrypted.
3. Retrieval runs over `PrivData_user ∪ PublicData` (Eq. 1); other users' data never reaches the LLM prompt.

| Scheme | Idea | Where |
|---|---|---|
| **Method A** | AES-CBC per-user key + HMAC tag; rows found through the user's primary-key list | `privrag/store/method_a.py` |
| **Method B** | Per-user encrypted linked list; each node has its own key derived one-way from the previous one; hash check per node; XOR trapdoor to the head. Two integrity modes: `base` (as published) and `improved` (AES-GCM) | `privrag/store/method_b.py` |

## 1.2 Status

| Area | State |
|---|---|
| Method A store | Done, tested |
| Method B store (`base` and `improved`) | Done, tested |
| Crypto primitives (AES-CBC, PKCS#7, HMAC, AES-GCM, HKDF, scrypt) | Done (`privrag/crypto.py`) |
| Plaintext baseline + naive shared-KB baseline | Done (used only for comparison) |
| RAG pipeline: chunk, embed, retrieve (private ∪ public, isolation firewall), prompt, LLM | Done |
| Local LLM through Ollama (default) + `EchoLLM` test double | Done |
| Sessions: tokens, TTL, logout, per-client lockout, optional encrypted persistence | Done |
| REST API (`PRIVRAG_STORE` = `plaintext` / `A` / `B` / `B_improved`) | Done |
| Demo UI: naive vs Method A vs Method B side by side, key-leak and tamper demos | Done |
| Evaluation: self-check, leakage, quality parity, key exposure, performance | Done |
| Dataset: 10 users × 30 chunks, 5 record types, 30 queries; build-time embedding + encryption | Done |
| Paper §4 proofs | Not re-derived; behaviour checked empirically |
| Public KB | 500 BEIR SciFact abstracts, loaded through Hugging Face `datasets` (zip download and a 40-sentence fallback behind it) |
| Docker image | Files updated, never built or run |

## 1.3 Headline results

| Measurement | Result | Command |
|---|---|---|
| Tests | **147 passed** | `python -m pytest tests` |
| End-to-end self-check, real Ollama, Method A | **9/9** | `python -m evaluation.selfcheck` |
| Method B with real Ollama (1 question, both modes) | correct answer, gold chunk ranked 1st | — |
| Prompt-injection attack, real Ollama, 1 user × 2 prompts | naive: **16** foreign secrets leaked; plaintext-isolated / Method A / Method B: **0** | `python -m evaluation.leakage --users 1 --prompts 2` |
| Retrieval parity, 30 queries, MiniLM | plaintext = A = B = B improved: Recall@10 0.933, MRR@10 0.685, nDCG@10 0.745 | `python -m evaluation.quality` |
| One leaked key at chain position 20 of 30 | Method A exposes **30/30**, Method B **10/30** | `python -m evaluation.key_exposure` |
| IV bit-flip tampering | A detects, B improved detects, **B silently changes the text** | UI tab "A vs B: key leak" |

## 1.4 Where B is better and where A is

For ordinary question answering A and B are indistinguishable. They differ under attack:

| Scenario | Method A | Method B | Method B improved |
|---|---|---|---|
| A mid-chain/record key leaks | everything readable | only that and newer records | only that and newer records |
| Ciphertext tampering | detected (HMAC) | **not detected** (no MAC) | detected (GCM) |
| Chain structure visible in DB | no chain | next-node pointer in clear | hidden |
| Read all records | fetch by id | sequential chain walk | sequential chain walk |

See [04_crypto_design.md](04_crypto_design.md) and [06_known_issues_and_limitations.md](06_known_issues_and_limitations.md).
