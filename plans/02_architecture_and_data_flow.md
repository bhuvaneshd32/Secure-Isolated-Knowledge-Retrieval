# 2. Architecture and data flow

## 2.1 Layers

```
 ┌──────────────────────── clients ─────────────────────────┐
 │  Streamlit demo (demo/app.py)      HTTP clients / curl    │
 └───────────────┬──────────────────────────┬────────────────┘
                 │ in-process calls         │ REST
                 ▼                          ▼
        evaluation/*, demo/app.py      privrag/api.py   (FastAPI: size limit,
        use RagService directly        create_app()      log redaction, error mapping)
                  └──────────────┬─────────────┘
                                 ▼
                   privrag/rag/pipeline.py   RagService  /  NaiveSharedRAG
             embed ┌─────────────┼──────────────┐ generate
                   ▼             ▼ retrieve     ▼
           rag/ingest.py   store/*          rag/generator.py
           (chunk, embed)  (encrypt)        (OllamaLLM / EchoLLM)
                              rag/retriever.py (pool, cosine, top-k, firewall)
                       │
                       ▼
        privrag/crypto.py  (AES-CBC, AES-GCM, HMAC, HKDF, scrypt, SHA-256)
                       │
                       ▼
                 SQLite database file
```

Stores behind one interface (`privrag/store/__init__.py: PrivateStore`), chosen by `privrag/store/factory.py`:
`plaintext` (baseline), `A` (`method_a.py`), `B` and `B_improved` (`method_b.py`).

## 2.2 The two knowledge bases

| | Private KB | Public KB |
|---|---|---|
| Content | 10 users × 30 synthetic medical records, 5 types (appointment, prescription, lab result, visit note, billing); `data/private_corpus.jsonl`, `data/make_dataset.py` | `data/public_kb.jsonl`: the first 500 BEIR SciFact abstracts (`python data/make_public_kb.py`) |
| Secret? | Yes: each user's rows are encrypted under that user's keys | No: plain text, embeddings cached in `data/public_kb.<embedder>.<n>.npy` |
| Canaries | Every private chunk ends with a unique `CANARY-<user>-<chunk>`; a foreign canary in a prompt or answer proves a leak | none |

Everything is embedded and encrypted **once, by `build_kb.py`**. The UI and the evaluations only embed the question.

## 2.3 Method A flows (`MethodAStore`)

```
register_user(user_id, password)
  K_i, K_mac ← 32 random bytes each                                   (Eq. 5, 6)
  KEK ← scrypt(password, salt) → KEK_enc, KEK_mac (HKDF)
  wrapped_keys = IV ‖ AES-CBC(KEK_enc, K_i‖K_mac) ‖ HMAC(KEK_mac, …)
  PKlist = [] encrypted with K_i, tagged with K_mac

unlock → derive KEK, verify wrapped keys → UserSession{"enc","mac"}  (wrong password ⇒ AuthenticationError)

add_chunks, per chunk, one transaction:
  PK ← uuid4;  3 fresh IVs
  c_emb, c_chunk, c_meta ← AES-CBC(K_i, …)
  tag ← HMAC(K_mac, PK ‖ IVs ‖ ciphertexts)                           (Eq. 10)
  PKlist ← PKlist ∪ PK                                                (Eq. 11)

load_user_data: PKlist → fetch rows → verify tag (else ⊥) → decrypt   (Eq. 12–14)
```

## 2.4 Method B flows (`MethodBStore`)

```
register_user(user_id, password)
  root ← scrypt(password, salt);  masterkey, keysalt ← HKDF(root, …)
  K_1 ← HKDF(masterkey, salt=user_id, "InitKey")                      (Eq. 16)
  trapdoor ← H(user_id‖keysalt)-expanded pad  XOR  (H(ID) ‖ K_1 ‖ addr=Null)   (Eq. 17)

unlock → recompute pad, un-mask trapdoor, VerifyID(H(ID))             (Eq. 20, 21)
         wrong password ⇒ AuthenticationError

add_chunks: walk to the tail (key of next node is stored in the tail);
  new node i+1 encrypted under K_{i+1}; contains K_{i+2} = HKDF(K_{i+1}, "NextKey")   (Eq. 22, 23)
  patch the tail's next pointer; if the chain was empty, rewrite the trapdoor        (Eq. 24)

load_user_data (Algorithm 1): head key+address from trapdoor →
  loop: decrypt node under K_i, check H(K_i), collect (emb, chunk), K_{i+1}, addr → stop at Null
```

Node layouts:

| Mode | Stored | Next address |
|---|---|---|
| `base` | 5 CBC fields (emb, chunk, meta, next key, H(K_i)), each with its own IV; no MAC | in clear (field 5, Eq. 19) |
| `improved` | one AES-GCM message under K_i, AAD = user id ‖ node address; trapdoor carries an HMAC | inside the ciphertext |

## 2.5 Answering a question (`RagService.answer`, same for A and B)

```
1. store.load_user_data(session)        only the caller's rows are decrypted
2. embedder.embed_one(question)
3. Retriever.retrieve(...)              firewall: metadata["owner"] must equal the caller
                                        pool = public KB ∪ the user's decrypted chunks    (Eq. 1)
                                        score = embedding · query (cosine, unit vectors)  (Eq. 25)
                                        top-k, ties broken by chunk id                    (Eq. 26)
4. build_prompt(question, retrieved)    "answer using only the provided context" + <context> blocks
5. llm.generate(prompt)                 Ollama by default
```

Because step 1 only ever decrypts the caller's rows, foreign data cannot reach step 4. The firewall in step 3 is an
independent second check.

## 2.6 Naive baseline

`NaiveSharedRAG` reproduces the vulnerable system of Figure 1: every user's plaintext in one pool, no ownership
check. It is built from `PlaintextStore.dump_all()` and used only by the UI and the leakage harness.

## 2.7 Schemas

```
Method A   ma_users(user_id PK, kdf_salt, kdf_n, wrapped_keys, pk_iv, pk_cipher, pk_tag)
           ma_records(pk PK, iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta, tag)      -- no owner column

Method B   mb_meta(k, v)                           -- records the mode; a DB cannot be reopened in the other mode
           mb_users(user_id PK, kdf_salt, kdf_n)
           mb_trapdoors(handle PK, trapdoor, tag)  -- tag only in improved mode
           mb_nodes(addr PK, iv_*/c_* x5, next_addr)         -- paper
           mb_nodes(addr PK, nonce, blob)                    -- improved

Plaintext  users(user_id, password), chunks(chunk_id, owner, text, embedding, metadata)    -- all in clear
```

## 2.8 Trust boundaries

| Attacker | Gets | Protected? |
|---|---|---|
| Someone with the raw DB file | ciphertext, IVs, tags / node blobs | Yes. A: wrapped keys need the password. B: the trapdoor needs the password-derived keysalt. |
| Another authenticated user | the API | Yes: they only decrypt their own list / chain. |
| Prompt-injection attacker | LLM output | Yes: nothing foreign is in the prompt. They can still extract their *own* data. |
| Someone who modifies DB rows | modified ciphertext | A and B improved detect it. **B does not detect IV malleation.** |
| One leaked key + DB | A: all of that user's records. B: that record and newer ones | see [04](04_crypto_design.md) |
| Malicious server process / memory dump | keys of logged-in users | **No.** Keys live in `UserSession` while logged in. |
| LLM provider | decrypted chunks in the prompt | Local Ollama keeps them on the machine. |
