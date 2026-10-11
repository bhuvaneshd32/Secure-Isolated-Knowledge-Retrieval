"""
Method A — AES-CBC based encryption (paper §3.2, Eq. 2, 5-14, 27).

Paper element                              Where implemented
-----------------------------------------  ---------------------------------
K_i  <-$ {0,1}^lambda   (Eq. 5)            register_user()
K_mac <-$ {0,1}^lambda  (Eq. 6)            register_user()
c = AES-CBC.Enc(K_i, IV, m)  (Eq. 7)       _encrypt()
m_pad = AES-CBC.Dec(...); padding check    _decrypt()  (Eq. 8, 9 -> ⊥)
Record = (PK_new, IV, c_emb, c_chunk, tag) _insert_record()  (Eq. 10)
tag = HMAC(K_mac, IV || c_emb || c_chunk)  _tag()
PKlist <- PKlist ∪ PK_new  (Eq. 11)       add_chunks()
Select(PK = PK_target)  (Eq. 12)           load_user_data()
HMAC check then decrypt, else ⊥ (13, 14)   _open_record()

Design notes / deliberate refinements (details: plans/04_crypto_design.md §4.3):
  * Per-user keys are random (Eq. 5, 6) as in the paper. So that the *server*
    can hold them without storing them in clear, they are wrapped
    (encrypt-then-MAC) under a key-encryption-key derived from the user's
    password with scrypt. Correct password == wrapped keys authenticate.
  * Eq. 7 requires a fresh IV for every encryption, and the chunk and the
    embedding are two separate encryptions, so each gets its own IV (plus one
    for the metadata field).
  * The tag covers PK, every IV and every ciphertext, with length prefixes
    (encrypt-then-MAC, verified before any decryption, constant-time).
  * The "PKlist held by the user" is persisted as an encrypted + MACed blob so
    that no plaintext owner column ties a row to a user.
  * The shared `records` table has no owner column at all: isolation is
    cryptographic. A foreign PK either is not in the caller's PKlist or fails
    its MAC under the caller's K_mac.
"""

import json
import sqlite3
import threading
import uuid

import numpy as np

from privrag import crypto as cp
from privrag.store import (
    AuthenticationError,
    ChunkRecord,
    DecryptedChunk,
    IntegrityError,
    IsolationViolation,
    PrivateStore,
    UserSession,
)

_SQLITE_IN_LIMIT = 500


class MethodAStore(PrivateStore):
    def __init__(self, db_path: str = ":memory:", kdf_n: int = 2**14):
        self.db_path = db_path
        self.kdf_n = kdf_n
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------ db
    def _init_db(self) -> None:
        with self._lock, self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ma_users (
                    user_id      TEXT PRIMARY KEY,
                    kdf_salt     BLOB NOT NULL,
                    kdf_n        INTEGER NOT NULL,
                    wrapped_keys BLOB NOT NULL,   -- IV || AES-CBC(K_i||K_mac) || tag
                    pk_iv        BLOB NOT NULL,   -- PKlist, encrypted with K_i
                    pk_cipher    BLOB NOT NULL,
                    pk_tag       BLOB NOT NULL
                )""")
            # Knowledge base: only primary keys + ciphertext. No owner column.
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ma_records (
                    pk       TEXT PRIMARY KEY,
                    iv_emb   BLOB NOT NULL,
                    iv_chunk BLOB NOT NULL,
                    iv_meta  BLOB NOT NULL,
                    c_emb    BLOB NOT NULL,
                    c_chunk  BLOB NOT NULL,
                    c_meta   BLOB NOT NULL,
                    tag      BLOB NOT NULL
                )""")

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _tag_parts(pk: str, iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta):
        # Eq. 10: tag = HMAC(K_mac, IV || c_embedding || c_chunk), extended with
        # the PK, per-field IVs and the metadata ciphertext (all length-prefixed).
        return (b"rec", pk.encode(), iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta)

    @staticmethod
    def _wrap_keys(
        password: str, user_id: str, salt: bytes, n: int, k_enc: bytes, k_mac: bytes
    ) -> bytes:
        kek = cp.scrypt_derive(password, salt, n=n)
        kek_enc = cp.hkdf(kek, b"privrag/A/kek-enc")
        kek_mac = cp.hkdf(kek, b"privrag/A/kek-mac")
        iv = cp.generate_iv()
        ct = cp.aes_cbc_encrypt(kek_enc, iv, k_enc + k_mac)
        tag = cp.hmac_sha256(kek_mac, b"wrap", user_id.encode(), iv, ct)
        return iv + ct + tag

    @staticmethod
    def _unwrap_keys(password: str, user_id: str, salt: bytes, n: int, blob: bytes):
        kek = cp.scrypt_derive(password, salt, n=n)
        kek_enc = cp.hkdf(kek, b"privrag/A/kek-enc")
        kek_mac = cp.hkdf(kek, b"privrag/A/kek-mac")
        iv, ct, tag = blob[:16], blob[16:-32], blob[-32:]
        if not cp.hmac_verify(kek_mac, tag, b"wrap", user_id.encode(), iv, ct):
            raise AuthenticationError("Invalid credentials")
        try:
            keys = cp.aes_cbc_decrypt(kek_enc, iv, ct)
        except cp.CryptoError:
            raise AuthenticationError("Invalid credentials") from None
        if len(keys) != 2 * cp.KEY_BYTES:
            raise AuthenticationError("Invalid credentials")
        return keys[: cp.KEY_BYTES], keys[cp.KEY_BYTES :]

    # --- PKlist (Eq. 11) ---------------------------------------------------
    @staticmethod
    def _seal_pklist(user_id: str, k_enc: bytes, k_mac: bytes, pks: list[str]):
        iv = cp.generate_iv()
        ct = cp.aes_cbc_encrypt(k_enc, iv, json.dumps(pks).encode())
        tag = cp.hmac_sha256(k_mac, b"pklist", user_id.encode(), iv, ct)
        return iv, ct, tag

    @staticmethod
    def _open_pklist(
        user_id: str, k_enc: bytes, k_mac: bytes, iv: bytes, ct: bytes, tag: bytes
    ) -> list[str]:
        if not cp.hmac_verify(k_mac, tag, b"pklist", user_id.encode(), iv, ct):
            raise IntegrityError("PKlist integrity check failed")
        try:
            return json.loads(cp.aes_cbc_decrypt(k_enc, iv, ct).decode())
        except (cp.CryptoError, ValueError):
            raise IntegrityError("PKlist integrity check failed") from None

    def _read_pklist(self, s: UserSession) -> list[str]:
        row = self.conn.execute(
            "SELECT pk_iv, pk_cipher, pk_tag FROM ma_users WHERE user_id = ?", (s.user_id,)
        ).fetchone()
        if row is None:
            raise AuthenticationError("Unknown user")
        return self._open_pklist(s.user_id, s.key("enc"), s.key("mac"), *row)

    def _write_pklist(self, s: UserSession, pks: list[str]) -> None:
        iv, ct, tag = self._seal_pklist(s.user_id, s.key("enc"), s.key("mac"), pks)
        self.conn.execute(
            "UPDATE ma_users SET pk_iv = ?, pk_cipher = ?, pk_tag = ? WHERE user_id = ?",
            (iv, ct, tag, s.user_id),
        )

    # --- record encryption (Eq. 7-10) -------------------------------------
    @staticmethod
    def _encode_embedding(emb: np.ndarray) -> bytes:
        arr = np.asarray(emb, dtype="<f4")
        if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("embedding must be a finite 1-D float32 vector")
        return arr.tobytes()

    def _insert_record(self, s: UserSession, chunk: ChunkRecord) -> str:
        if chunk.embedding is None:
            raise ValueError("ChunkRecord.embedding is required")
        k_enc, k_mac = s.key("enc"), s.key("mac")

        meta = dict(chunk.metadata)
        meta["owner"] = s.user_id  # checked again by the Retriever firewall

        pk = uuid.uuid4().hex
        iv_emb, iv_chunk, iv_meta = cp.generate_iv(), cp.generate_iv(), cp.generate_iv()
        c_emb = cp.aes_cbc_encrypt(k_enc, iv_emb, self._encode_embedding(chunk.embedding))
        c_chunk = cp.aes_cbc_encrypt(k_enc, iv_chunk, chunk.text.encode("utf-8"))
        c_meta = cp.aes_cbc_encrypt(k_enc, iv_meta, json.dumps(meta).encode("utf-8"))
        tag = cp.hmac_sha256(
            k_mac, *self._tag_parts(pk, iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta)
        )

        self.conn.execute(
            "INSERT INTO ma_records (pk, iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta, tag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pk, iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta, tag),
        )
        return pk

    def _open_record(self, s: UserSession, row) -> DecryptedChunk:
        pk, iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta, tag = row
        if not all(
            isinstance(x, bytes) for x in (iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta, tag)
        ):
            raise IntegrityError("Record integrity check failed")
        k_enc, k_mac = s.key("enc"), s.key("mac")
        # Eq. 13: verify the MAC first; only then decrypt (Eq. 14).
        if not cp.hmac_verify(
            k_mac, tag, *self._tag_parts(pk, iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta)
        ):
            raise IntegrityError("Record integrity check failed")
        try:
            emb_bytes = cp.aes_cbc_decrypt(k_enc, iv_emb, c_emb)
            text = cp.aes_cbc_decrypt(k_enc, iv_chunk, c_chunk).decode("utf-8")
            meta = json.loads(cp.aes_cbc_decrypt(k_enc, iv_meta, c_meta).decode("utf-8"))
        except (cp.CryptoError, ValueError):
            raise IntegrityError("Record integrity check failed") from None
        if len(emb_bytes) == 0 or len(emb_bytes) % 4 != 0:
            raise IntegrityError("Record integrity check failed")
        emb = np.frombuffer(emb_bytes, dtype="<f4").astype(np.float32)
        return DecryptedChunk(chunk_id=pk, text=text, embedding=emb, metadata=meta)

    # ------------------------------------------------------- PrivateStore
    def register_user(self, user_id: str, password: str) -> None:
        if not user_id or not password:
            raise ValueError("user_id and password are required")
        k_enc = cp.generate_key()  # Eq. 5
        k_mac = cp.generate_key()  # Eq. 6
        salt = cp.random_bytes(16)
        wrapped = self._wrap_keys(password, user_id, salt, self.kdf_n, k_enc, k_mac)
        iv, ct, tag = self._seal_pklist(user_id, k_enc, k_mac, [])
        with self._lock:
            try:
                with self.conn:
                    self.conn.execute(
                        "INSERT INTO ma_users "
                        "(user_id, kdf_salt, kdf_n, wrapped_keys, pk_iv, pk_cipher, pk_tag) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (user_id, salt, self.kdf_n, wrapped, iv, ct, tag),
                    )
            except sqlite3.IntegrityError:
                raise IntegrityError("User already exists") from None

    def unlock(self, user_id: str, password: str) -> UserSession:
        with self._lock:
            row = self.conn.execute(
                "SELECT kdf_salt, kdf_n, wrapped_keys FROM ma_users WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            # Burn comparable CPU so unknown users are not distinguishable by timing.
            cp.scrypt_derive(password, b"\x00" * 16, n=self.kdf_n)
            raise AuthenticationError("Invalid credentials")
        salt, n, wrapped = row
        k_enc, k_mac = self._unwrap_keys(password, user_id, salt, n, wrapped)
        return UserSession(user_id, {"enc": k_enc, "mac": k_mac})

    def add_chunks(self, s: UserSession, chunks: list[ChunkRecord]) -> list[str]:
        with self._lock, self.conn:  # one transaction: records + PKlist
            pklist = self._read_pklist(s)
            new_pks = [self._insert_record(s, c) for c in chunks]
            self._write_pklist(s, pklist + new_pks)  # Eq. 11
        return new_pks

    def load_user_data(self, s: UserSession) -> list[DecryptedChunk]:
        with self._lock:
            pklist = self._read_pklist(s)
            rows = {}
            for i in range(0, len(pklist), _SQLITE_IN_LIMIT):
                part = pklist[i : i + _SQLITE_IN_LIMIT]
                q = ",".join("?" * len(part))
                try:
                    fetched = self.conn.execute(
                        "SELECT pk, iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta, tag "
                        f"FROM ma_records WHERE pk IN ({q})",
                        part,
                    ).fetchall()
                except sqlite3.Error:  # corrupted / mistyped cells => ⊥
                    raise IntegrityError("Record integrity check failed") from None
                for r in fetched:
                    rows[r[0]] = r
        out: list[DecryptedChunk] = []
        for pk in pklist:  # Eq. 12: Select(PK = PK_target)
            row = rows.get(pk)
            if row is None:  # PK held by the user but record gone => ⊥
                raise IntegrityError("Record missing")
            out.append(self._open_record(s, row))
        return out

    def delete_chunk(self, s: UserSession, chunk_id: str) -> None:
        with self._lock, self.conn:
            pklist = self._read_pklist(s)
            if chunk_id not in pklist:
                exists = self.conn.execute(
                    "SELECT 1 FROM ma_records WHERE pk = ?", (chunk_id,)
                ).fetchone()
                if exists:
                    raise IsolationViolation("Attempted to delete a chunk owned by another user")
                return  # idempotent
            self.conn.execute("DELETE FROM ma_records WHERE pk = ?", (chunk_id,))
            self._write_pklist(s, [p for p in pklist if p != chunk_id])

    # ----------------------------------------------------- demo / forensics
    def raw_records(self, limit: int | None = 20) -> list[dict]:
        """What someone with raw database access (e.g. a DBA) can see."""
        q = "SELECT pk, iv_emb, iv_chunk, c_emb, c_chunk, c_meta, tag FROM ma_records"
        if limit:
            q += f" LIMIT {int(limit)}"
        keys = ["pk", "iv_emb", "iv_chunk", "c_emb", "c_chunk", "c_meta", "tag"]
        return [
            {k: (v if k == "pk" else v.hex()) for k, v in zip(keys, r, strict=True)}
            for r in self.conn.execute(q).fetchall()
        ]

    def record_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM ma_records").fetchone()[0]
