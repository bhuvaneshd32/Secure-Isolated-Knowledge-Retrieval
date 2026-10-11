# 8. Roadmap: what remains

Both methods of the paper are implemented; what is left is verification breadth and hardening.

## 8.1 Verification

| # | Task | Done when |
|---|---|---|
| V2 | Full real-LLM leakage benchmark (10 users × 4 prompts × 4 systems, run in the background) | table added to [05](05_testing_and_evaluation.md) |
| V3 | Click through the UI in a browser against live Ollama | all tabs work with the real model |
| V4 | Build the Docker image and run `smoke_test.py` against it | container healthy, smoke test passes |
| V5 | Pytest wrapper for the Streamlit `AppTest` script | UI regressions are caught in CI |

## 8.2 Hardening

| # | Task | Notes |
|---|---|---|
| R1 | Rollback protection: version counter inside the MACed PKlist / trapdoor (H1) | closes a gap the paper leaves open |
| R2 | Store embedder name and dimension with the data and refuse mismatches (M5) | prevents silent garbage retrieval |
| R4 | Real overlap in the chunker (M10) | retrieval quality |
| R5 | Tail pointer in Method B so appends do not walk the whole chain (M4) | deviates from the paper; keep the walking version as the reference |
| R6 | Pin `requirements.txt`; commit the work in logical commits | hygiene |

## 8.3 Nice to have

- Streaming answers from Ollama in the UI.
- Password change (re-wrap only), account deletion, data export.
- Pagination for `GET /documents`; batched embedding for large uploads.
- A harder private corpus (free-text clinical notes) for a more realistic retrieval evaluation.
- Per-record keys for Method A as an ablation, to compare blast radius with Method B.
