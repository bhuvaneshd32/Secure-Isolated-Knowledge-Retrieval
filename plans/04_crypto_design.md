# 4. Crypto design: Method A and Method B

Source: `privrag/store/method_a.py`, `privrag/store/method_b.py`, `privrag/crypto.py`.
Paper: §3.2 (A), §3.3 (B), §3.4 (RAG search), §4 (proofs, not re-derived here).

## 4.1 Method A, equation to code

| Paper | Requirement | Implementation |
|---|---|---|
| Eq. 1 | `PrivData_user ∪ PublicData` | `Retriever.retrieve` |
| Eq. 5, 6 | `K_i, K_mac ←$ {0,1}^λ` | `register_user` → `generate_key()` ×2 (λ = 256) |
| Eq. 7 | `c = AES-CBC.Enc(K_i, IV, m)`, fresh IV | `_insert_record`, 3 new IVs per record |
| Eq. 8, 9 | decrypt, padding valid else ⊥ | `aes_cbc_decrypt` → `unpad` → `CryptoError` → `IntegrityError` |
| Eq. 10 | record + HMAC tag | `_insert_record`, `_tag_parts` |
| Eq. 11 | `PKlist ← PKlist ∪ PK_new` | `add_chunks` → `_write_pklist` |
| Eq. 12 | `Select(PK = PK_target)` | `load_user_data` |
| Eq. 13, 27 | verify HMAC else ⊥ | `_open_record` (before any decryption) |
| Eq. 14 | decrypt | `_open_record` |

**`ma_records` row** (BLOBs except `pk`):

```
pk        32 hex (uuid4)
iv_emb    16 B     c_emb    AES-CBC(K_i, iv_emb,   float32-LE, PKCS7)    1536 B -> 1552 B
iv_chunk  16 B     c_chunk  AES-CBC(K_i, iv_chunk, UTF-8 text)
iv_meta   16 B     c_meta   AES-CBC(K_i, iv_meta,  JSON metadata incl. "owner")
tag       32 B     HMAC-SHA256(K_mac, "rec"‖pk‖iv_emb‖iv_chunk‖iv_meta‖c_emb‖c_chunk‖c_meta), length-prefixed
```

**`ma_users` row**: `kdf_salt`, `kdf_n`, `wrapped_keys = IV ‖ AES-CBC(KEK_enc, K_i‖K_mac) ‖ HMAC(KEK_mac, …)` with
`KEK = scrypt(password)`, and the PKlist as `pk_iv`, `pk_cipher`, `pk_tag` (AES-CBC under `K_i`, HMAC under `K_mac`).

## 4.2 Method B, equation to code

| Paper | Requirement | Implementation |
|---|---|---|
| Eq. 15 | per-user `keysalt` | `_derive_secrets` (scrypt-derived, see deviation 1) |
| Eq. 16 | `K_1 = HKDF(masterkey, salt=ID, "InitKey")` | `_initial_key` |
| Eq. 17 | `T = H(ID‖keysalt) ⊕ (ID‖K_1‖addr(node_1))` | `_pad`, `_write_trapdoor` |
| Eq. 18 | `node_i = emb‖chunk‖K_{i+1}‖H(K_i)‖addr(node_{i+1})` | `_Node`, `_store_node` |
| Eq. 19 | fields 1-4 AES-CBC under `K_i`, field 5 in clear | `paper` mode |
| Eq. 20, 21 | un-mask trapdoor, `VerifyID` | `unlock`, `_open_trapdoor` |
| Algorithm 1 | decrypt, check `H(K_i)`, collect, follow address, stop at Null | `_follow`, `load_user_data` |
| Eq. 22 | `K_{i+2} = HKDF(K_i, salt=ID, "NextKey")` | `_next_key` |
| Eq. 23, 24 | build new node, patch the tail's pointer | `add_chunks` |
| Eq. 28 | decrypt each field | `_load_node` |

**Paper-mode node row** (`mb_nodes`): `addr`, then `(iv_x, c_x)` for `x` in `emb`, `chunk`, `meta`, `key`
(`K_{i+1}`), `h` (`H(K_i)`), then `next_addr` in clear.

**Hardened-mode node row**: `addr`, `nonce` (12 B), `blob = AES-GCM(K_i, nonce, len-prefixed(emb, chunk, meta,
K_{i+1}, H(K_i), next_addr), AAD = "mb-node|" ‖ user_id ‖ "|" ‖ addr)`.

**Trapdoor** (80 B): `(H(ID) 32 B ‖ K_head 32 B ‖ addr_head 16 B) ⊕ HKDF(H(user_id ‖ keysalt), 80 B)`; hardened mode
adds `HMAC(HKDF(masterkey), trapdoor)`.

Operations not in the paper: **delete** unlinks a node (predecessor takes over the deleted node's next key and
next address; deleting the head rewrites the trapdoor) and works for head, middle, tail and last remaining node.

## 4.3 Deviations from the paper

| # | Paper | Implementation | Why |
|---|---|---|---|
| 1 | `keysalt` sampled at random and held by the user; masterkey unspecified | Both derived from the password with scrypt + HKDF | A server-side app cannot rely on the user holding them; a wrong password then fails `VerifyID` |
| 2 | Eq. 17 XORs a λ-bit hash with the longer `ID‖K_1‖addr` | Pad is HKDF-expanded from `H(ID‖keysalt)` to 80 B; ID field is `H(ID)` (fixed width) | The equation does not type-check as written |
| 3 | Eq. 19 stores the next address in clear; Eq. 28 decrypts it | `paper` follows Eq. 19, `hardened` encrypts it | The paper is inconsistent; both readings are provided |
| 4 | Eq. 22 writes `K_{i+2} ← HKDF(K_i)` | Consistent chain: `K_{i+1} = HKDF(K_i)` | Index inconsistency in the paper |
| 5 | One IV for the fields of a record/node | One IV per field | Eq. 7 requires a fresh IV for every encryption |
| 6 | Eq. 10 tags `IV‖c_emb‖c_chunk` (13/27 differ) | A: tag over PK, all IVs, all ciphertexts, length-prefixed, verified first | Boundary-shifting and PK-swap protection |
| 7 | Metadata not mentioned | Encrypted as an extra field in both methods | The UI/API need `source`, `date`, `owner` |
| 8 | Keys "kept by the user" (A) | Keys wrapped under a password-derived key, in memory only during a session | Server-side operation |
| 9 | PKlist "held by the user" (A) | Stored as an encrypted, MACed blob; no owner column in `ma_records` | Persistence without a plaintext ownership map |
| 10 | B has no MAC (only `H(K_i)`) | `paper` mode keeps that; `hardened` adds AES-GCM, bound address, MACed trapdoor | `H(K_i)` is not bound to the ciphertext, so CBC is malleable |
| 11 | ⊥ | `AuthenticationError`, `IntegrityError`, `IsolationViolation`; API returns one generic 500 for integrity failures | No padding or MAC oracle for the client |

Also: the paper's §4 "forward security" claim for B does not hold as stated (the key of node *i* decrypts node *i*,
which contains `K_{i+1}`, so it opens every later node). What the chain really gives is **backward** protection:
a key from the middle cannot open *earlier* nodes, because HKDF is one-way and pointers only go forward.

## 4.4 Security properties and how they are checked

| Property | Argument | Evidence |
|---|---|---|
| Confidentiality at rest | per-field random IVs / nonces; keys never stored in clear | A: `test_method_a_security.py`; B: `test_no_plaintext_on_disk`, `test_same_plaintext_gives_different_ciphertext` |
| Integrity | A: encrypt-then-MAC. B-hardened: AES-GCM + bound address + MACed trapdoor | A: `test_any_bit_flip_is_detected`; B: `test_hardened_detects_any_flipped_byte`, `…truncation_and_swapping`, `…trapdoor_is_authenticated` |
| Known weakness of B-paper | no ciphertext authentication | `test_paper_mode_detects_wrong_key_but_not_iv_malleation` documents it |
| User isolation | A: foreign rows fail the MAC; B: foreign chains need other keys; neither has an owner column | contract tests, `test_attacker_injecting_foreign_pk…`, `test_users_have_independent_chains` |
| Chain semantics (B) | append, delete head / middle / tail / all | `test_delete_head_middle_tail_then_append`, `test_delete_everything_then_reuse` |
| Key-exposure blast radius | B: a node key opens that node and later ones only | `test_leaked_node_key_opens_only_that_node_and_later_ones`, `evaluation/key_exposure.py` |
| Primitive correctness | NIST SP 800-38A CBC vector, RFC 5869 HKDF vector | `test_crypto_primitives.py` |

These are **empirical** checks of the properties §4 proves on paper; no formal reduction was re-derived.

## 4.5 Cost

Both methods decrypt the user's whole private KB per query (A: by id, B: by walking the chain), so latency grows
linearly. Measured numbers are in [05_testing_and_evaluation.md](05_testing_and_evaluation.md).
