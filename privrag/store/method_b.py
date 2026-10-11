"""
Method B: chained dynamic key derivation (paper §3.3, Eq. 3, 15-24, Algorithm 1).

Each user owns a singly linked chain of nodes. Node i is encrypted under its own
key K_i, and contains the key K_{i+1} for the next node, so a reader walks the
chain from the head; keys are never reused between nodes.

Paper element                                   Where implemented
----------------------------------------------  ---------------------------
K_1 = HKDF(masterkey, salt=ID, "InitKey")       _initial_key()        (Eq. 15, 16)
T = H(ID||keysalt) xor (ID||K_1||addr(node_1))  _seal_trapdoor()      (Eq. 17, 20, 21)
VerifyID, recompute H(ID||keysalt), xor T       unlock(), _open_trapdoor()
node = emb||chunk||K_{i+1}||H(K_i)||addr(next)  _Node + _encode()      (Eq. 3, 18, 19)
walk chain, check H(K_i), stop at Null          _walk()               (Algorithm 1)
K_{i+2} = HKDF(K_i, salt=ID, "NextKey")         _next_key()           (Eq. 22)
append: patch last node's pointer               add_chunks()          (Eq. 23, 24)

Two integrity modes, selected by `mode`:

  "paper"     faithful to the paper. Fields are AES-256-CBC encrypted under K_i
              with a fresh IV each, the next address is stored in clear, and the
              only integrity check is H(K_i) inside the node. There is NO
              ciphertext authentication (the paper's Method B has none), so
              CBC bit-flipping on an IV can change a field undetected.

  "hardened"  the same chain, but each node is one AES-256-GCM message whose AAD
              binds it to (user id, node address). The next address is inside
              the ciphertext, node addresses are random, and the trapdoor
              carries an HMAC. Any modification is detected.

Deliberate choices where the paper is silent (details: plans/04_crypto_design.md §4.3):
  * D2: the trapdoor payload (80 bytes) is longer than SHA-256, so the pad is
    HKDF-expanded from H(ID||keysalt).
  * D3: the trapdoor is looked up by the public user id.
  * D4: masterkey and keysalt are derived from the password with scrypt, so the
    server stores nothing from which they can be recovered. A wrong password
    yields a wrong pad, VerifyID fails, and unlock() raises AuthenticationError.
  * ID inside the trapdoor is H(user id), a fixed-width stand-in for the id.
  * The head key and head address live in the trapdoor, so deleting the first
    node just rewrites the trapdoor.
"""

import json
import sqlite3
import struct
import threading
import uuid
from dataclasses import dataclass

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

MODES = ("paper", "hardened")
ADDR_BYTES = 16
ID_BYTES = 32
TRAPDOOR_BYTES = ID_BYTES + cp.KEY_BYTES + ADDR_BYTES  # H(ID) || K_head || addr(head)
NULL_ADDR = b"\x00" * ADDR_BYTES
FIELDS = ("emb", "chunk", "meta", "key", "h")  # the five encrypted fields in paper mode


@dataclass
class _Node:
    addr: str  # hex of 16 random bytes; also the public chunk id
    emb: bytes  # float32 little-endian
    text: str
    meta: dict
    next_key: bytes  # K_{i+1}
    next_addr: str | None  # None == Null


class _BadTrapdoor(Exception):
    pass


def _pack(*parts: bytes) -> bytes:
    return b"".join(struct.pack(">I", len(p)) + p for p in parts)


def _unpack(buf: bytes, n: int) -> list[bytes]:
    out, i = [], 0
    for _ in range(n):
        if i + 4 > len(buf):
            raise ValueError("truncated")
        (ln,) = struct.unpack(">I", buf[i : i + 4])
        i += 4
        if i + ln > len(buf):
            raise ValueError("truncated")
        out.append(buf[i : i + ln])
        i += ln
    if i != len(buf):
        raise ValueError("trailing bytes")
    return out


class MethodBStore(PrivateStore):
    def __init__(self, db_path: str = ":memory:", mode: str = "hardened", kdf_n: int = 2**14):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.db_path = db_path
        self.kdf_n = kdf_n
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------ db
    def _init_db(self) -> None:
        with self._lock, self.conn:
            self.conn.execute("CREATE TABLE IF NOT EXISTS mb_meta (k TEXT PRIMARY KEY, v TEXT)")
            row = self.conn.execute("SELECT v FROM mb_meta WHERE k = 'mode'").fetchone()
            if row is None:
                self.conn.execute("INSERT INTO mb_meta VALUES ('mode', ?)", (self.mode,))
            elif row[0] != self.mode:
                raise ValueError(f"database was created in mode {row[0]!r}, not {self.mode!r}")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS mb_users (
                    user_id  TEXT PRIMARY KEY,
                    kdf_salt BLOB NOT NULL,
                    kdf_n    INTEGER NOT NULL
                )""")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS mb_trapdoors (
                    handle   TEXT PRIMARY KEY,   -- public user id (D3)
                    trapdoor BLOB NOT NULL,
                    tag      BLOB                -- hardened mode only
                )""")
            if self.mode == "paper":
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS mb_nodes (
                        addr TEXT PRIMARY KEY,
                        iv_emb BLOB, c_emb BLOB, iv_chunk BLOB, c_chunk BLOB,
                        iv_meta BLOB, c_meta BLOB, iv_key BLOB, c_key BLOB,
                        iv_h BLOB, c_h BLOB,
                        next_addr TEXT           -- in clear, as in the paper (field 5)
                    )""")
            else:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS mb_nodes (
                        addr  TEXT PRIMARY KEY,
                        nonce BLOB NOT NULL,
                        blob  BLOB NOT NULL      -- AES-GCM(node), pointer inside
                    )""")

    # ----------------------------------------------------- key derivation
    @staticmethod
    def _initial_key(master: bytes, user_id: str) -> bytes:
        return cp.hkdf(master, b"InitKey", salt=user_id.encode())  # Eq. 15, 16

    @staticmethod
    def _next_key(key: bytes, user_id: str) -> bytes:
        return cp.hkdf(key, b"NextKey", salt=user_id.encode())  # Eq. 22

    @staticmethod
    def _derive_secrets(password: str, salt: bytes, n: int) -> tuple[bytes, bytes]:
        root = cp.scrypt_derive(password, salt, n=n)
        master = cp.hkdf(root, b"privrag/B/masterkey")
        keysalt = cp.hkdf(root, b"privrag/B/keysalt")
        return master, keysalt

    # ----------------------------------------------------------- trapdoor
    @staticmethod
    def _id_hash(user_id: str) -> bytes:
        return cp.sha256(b"privrag/B/id", user_id.encode())

    @staticmethod
    def _pad(user_id: str, keysalt: bytes) -> bytes:
        seed = cp.sha256(user_id.encode(), keysalt)  # H(ID || keysalt)
        return cp.hkdf(seed, b"privrag/B/trapdoor-pad", length=TRAPDOOR_BYTES)  # D2

    def _tag(self, master: bytes, user_id: str, trapdoor: bytes) -> bytes:
        key = cp.hkdf(master, b"privrag/B/trapdoor-tag")
        return cp.hmac_sha256(key, b"trapdoor", user_id.encode(), trapdoor)

    def _tag_ok(self, master: bytes, user_id: str, trapdoor: bytes, tag: bytes) -> bool:
        return cp.hmac_verify(
            cp.hkdf(master, b"privrag/B/trapdoor-tag"), tag, b"trapdoor", user_id.encode(), trapdoor
        )

    def _write_trapdoor(
        self, user_id: str, master: bytes, keysalt: bytes, k_head: bytes, addr: bytes
    ) -> None:
        payload = self._id_hash(user_id) + k_head + addr
        t = cp.xor_bytes(payload, self._pad(user_id, keysalt))  # Eq. 17
        tag = self._tag(master, user_id, t) if self.mode == "hardened" else None
        self.conn.execute(
            "INSERT OR REPLACE INTO mb_trapdoors (handle, trapdoor, tag) VALUES (?, ?, ?)",
            (user_id, t, tag),
        )

    def _open_trapdoor(self, user_id: str, master: bytes, keysalt: bytes) -> tuple[bytes, bytes]:
        row = self.conn.execute(
            "SELECT trapdoor, tag FROM mb_trapdoors WHERE handle = ?", (user_id,)
        ).fetchone()
        if row is None:
            raise _BadTrapdoor
        t, tag = row
        if not isinstance(t, bytes) or len(t) != TRAPDOOR_BYTES:
            raise _BadTrapdoor
        if self.mode == "hardened" and not (
            isinstance(tag, bytes) and self._tag_ok(master, user_id, t, tag)
        ):
            raise _BadTrapdoor
        payload = cp.xor_bytes(t, self._pad(user_id, keysalt))
        idh, k_head, addr = (
            payload[:ID_BYTES],
            payload[ID_BYTES : ID_BYTES + cp.KEY_BYTES],
            payload[ID_BYTES + cp.KEY_BYTES :],
        )
        if idh != self._id_hash(user_id):  # VerifyID
            raise _BadTrapdoor
        return k_head, addr

    def _head(self, s: UserSession) -> tuple[bytes, bytes]:
        try:
            return self._open_trapdoor(s.user_id, s.key("master"), s.key("keysalt"))
        except _BadTrapdoor:
            raise IntegrityError("Trapdoor integrity check failed") from None

    # --------------------------------------------------------------- nodes
    @staticmethod
    def _encode_embedding(emb: np.ndarray) -> bytes:
        arr = np.asarray(emb, dtype="<f4")
        if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("embedding must be a finite 1-D float32 vector")
        return arr.tobytes()

    def _aad(self, user_id: str, addr: str) -> bytes:
        return b"mb-node|" + user_id.encode() + b"|" + addr.encode()

    def _store_node(self, node: _Node, key: bytes, user_id: str) -> None:
        h = cp.sha256(key)  # H(K_i)
        meta = json.dumps(node.meta).encode("utf-8")
        text = node.text.encode("utf-8")
        if self.mode == "paper":
            vals = {"emb": node.emb, "chunk": text, "meta": meta, "key": node.next_key, "h": h}
            cols: list = [node.addr]
            for f in FIELDS:  # fresh IV per field (Eq. 7)
                iv = cp.generate_iv()
                cols += [iv, cp.aes_cbc_encrypt(key, iv, vals[f])]
            cols.append(node.next_addr)
            self.conn.execute(
                "INSERT OR REPLACE INTO mb_nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", cols
            )
        else:
            body = _pack(
                node.emb, text, meta, node.next_key, h, (node.next_addr or "").encode("ascii")
            )
            nonce = cp.random_bytes(cp.NONCE_BYTES)
            blob = cp.aes_gcm_encrypt(key, nonce, body, self._aad(user_id, node.addr))
            self.conn.execute(
                "INSERT OR REPLACE INTO mb_nodes (addr, nonce, blob) VALUES (?, ?, ?)",
                (node.addr, nonce, blob),
            )

    def _load_node(self, row, key: bytes, user_id: str) -> _Node:
        """Decrypt one node row under `key`; raises IntegrityError (the paper's bottom)."""
        try:
            if self.mode == "paper":
                addr = row[0]
                parts = {}
                for i, f in enumerate(FIELDS):
                    iv, c = row[1 + 2 * i], row[2 + 2 * i]
                    if not isinstance(iv, bytes) or not isinstance(c, bytes):
                        raise cp.CryptoError("bad cell")
                    parts[f] = cp.aes_cbc_decrypt(key, iv, c)
                next_addr = row[11]
                emb, text, meta = parts["emb"], parts["chunk"], parts["meta"]
                next_key, h = parts["key"], parts["h"]
            else:
                addr, nonce, blob = row
                if not isinstance(nonce, bytes) or not isinstance(blob, bytes):
                    raise cp.CryptoError("bad cell")
                body = cp.aes_gcm_decrypt(key, nonce, blob, self._aad(user_id, addr))
                emb, text, meta, next_key, h, na = _unpack(body, 6)
                next_addr = na.decode("ascii") or None
            if h != cp.sha256(key):  # check H(K_i) (Algorithm 1)
                raise cp.CryptoError("key hash mismatch")
            if len(next_key) != cp.KEY_BYTES or len(emb) == 0 or len(emb) % 4:
                raise cp.CryptoError("malformed node")
            if next_addr is not None and (
                not isinstance(next_addr, str) or len(next_addr) != 2 * ADDR_BYTES
            ):
                raise cp.CryptoError("malformed pointer")
            return _Node(
                addr=addr,
                emb=emb,
                text=text.decode("utf-8"),
                meta=json.loads(meta.decode("utf-8")),
                next_key=next_key,
                next_addr=next_addr,
            )
        except (cp.CryptoError, ValueError, TypeError, KeyError, IndexError):
            raise IntegrityError("Record integrity check failed") from None

    def _fetch(self, addr: str):
        return self.conn.execute("SELECT * FROM mb_nodes WHERE addr = ?", (addr,)).fetchone()

    def _follow(self, user_id: str, key: bytes, addr: str | None) -> list[tuple[_Node, bytes]]:
        """Algorithm 1: decrypt, check H(K_i), collect, follow the pointer until Null."""
        chain: list[tuple[_Node, bytes]] = []
        seen: set[str] = set()
        while addr is not None:
            if addr in seen:
                raise IntegrityError("Chain contains a cycle")
            seen.add(addr)
            try:
                row = self._fetch(addr)
            except sqlite3.Error:
                raise IntegrityError("Record integrity check failed") from None
            if row is None:
                raise IntegrityError("Record missing")
            node = self._load_node(row, key, user_id)
            chain.append((node, key))
            key, addr = node.next_key, node.next_addr
        return chain

    def _walk(self, s: UserSession) -> tuple[list[tuple[_Node, bytes]], bytes]:
        k_head, addr = self._head(s)
        a = None if addr == NULL_ADDR else addr.hex()
        return self._follow(s.user_id, k_head, a), k_head

    @staticmethod
    def _to_chunk(node: _Node, user_id: str) -> DecryptedChunk:
        meta = dict(node.meta)
        meta["owner"] = user_id
        emb = np.frombuffer(node.emb, dtype="<f4").astype(np.float32)
        return DecryptedChunk(chunk_id=node.addr, text=node.text, embedding=emb, metadata=meta)

    # ------------------------------------------------------- PrivateStore
    def register_user(self, user_id: str, password: str) -> None:
        if not user_id or not password:
            raise ValueError("user_id and password are required")
        salt = cp.random_bytes(16)
        master, keysalt = self._derive_secrets(password, salt, self.kdf_n)
        with self._lock:
            try:
                with self.conn:
                    self.conn.execute(
                        "INSERT INTO mb_users (user_id, kdf_salt, kdf_n) VALUES (?, ?, ?)",
                        (user_id, salt, self.kdf_n),
                    )
                    self._write_trapdoor(
                        user_id,
                        master,
                        keysalt,
                        self._initial_key(master, user_id),
                        NULL_ADDR,
                    )
            except sqlite3.IntegrityError:
                raise IntegrityError("User already exists") from None

    def unlock(self, user_id: str, password: str) -> UserSession:
        with self._lock:
            row = self.conn.execute(
                "SELECT kdf_salt, kdf_n FROM mb_users WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            cp.scrypt_derive(password, b"\x00" * 16, n=self.kdf_n)  # equalise timing
            raise AuthenticationError("Invalid credentials")
        master, keysalt = self._derive_secrets(password, row[0], row[1])  # outside the lock
        with self._lock:
            try:
                self._open_trapdoor(user_id, master, keysalt)  # VerifyID
            except _BadTrapdoor:
                raise AuthenticationError("Invalid credentials") from None
        return UserSession(user_id, {"master": master, "keysalt": keysalt})

    def add_chunks(self, s: UserSession, chunks: list[ChunkRecord]) -> list[str]:
        for c in chunks:
            if c.embedding is None:
                raise ValueError("ChunkRecord.embedding is required")
        embs = [self._encode_embedding(c.embedding) for c in chunks]
        if not chunks:
            return []
        with self._lock, self.conn:
            chain, k_head = self._walk(s)
            key = chain[-1][0].next_key if chain else k_head  # key for the new node

            new: list[tuple[_Node, bytes]] = []
            for c, emb in zip(chunks, embs, strict=True):
                meta = dict(c.metadata)
                meta["owner"] = s.user_id
                node = _Node(
                    addr=uuid.uuid4().bytes.hex(),
                    emb=emb,
                    text=c.text,
                    meta=meta,
                    next_key=self._next_key(key, s.user_id),
                    next_addr=None,
                )
                new.append((node, key))
                key = node.next_key
            for (a, _), (b, _) in zip(new, new[1:], strict=False):
                a.next_addr = b.addr
            for node, k in new:
                self._store_node(node, k, s.user_id)

            first = new[0][0].addr
            if chain:  # Eq. 23, 24: patch the previous tail's pointer
                tail, tail_key = chain[-1]
                tail.next_addr = first
                self._store_node(tail, tail_key, s.user_id)
            else:  # first node: the trapdoor now points at it
                self._write_trapdoor(
                    s.user_id, s.key("master"), s.key("keysalt"), k_head, bytes.fromhex(first)
                )
        return [n.addr for n, _ in new]

    def load_user_data(self, s: UserSession) -> list[DecryptedChunk]:
        with self._lock:
            chain, _ = self._walk(s)
        return [self._to_chunk(n, s.user_id) for n, _ in chain]

    def delete_chunk(self, s: UserSession, chunk_id: str) -> None:
        with self._lock, self.conn:
            chain, k_head = self._walk(s)
            idx = next((i for i, (n, _) in enumerate(chain) if n.addr == chunk_id), None)
            if idx is None:
                exists = self.conn.execute(
                    "SELECT 1 FROM mb_nodes WHERE addr = ?", (chunk_id,)
                ).fetchone()
                if exists:
                    raise IsolationViolation("Attempted to delete a chunk owned by another user")
                return  # idempotent
            node, _ = chain[idx]
            self.conn.execute("DELETE FROM mb_nodes WHERE addr = ?", (chunk_id,))
            if idx > 0:  # unlink: predecessor now points at our successor
                pred, pred_key = chain[idx - 1]
                pred.next_addr = node.next_addr
                pred.next_key = node.next_key
                self._store_node(pred, pred_key, s.user_id)
            else:  # we were the head: the trapdoor carries the new head
                addr = bytes.fromhex(node.next_addr) if node.next_addr else NULL_ADDR
                self._write_trapdoor(
                    s.user_id, s.key("master"), s.key("keysalt"), node.next_key, addr
                )

    # ----------------------------------------------------- demo / forensics
    def raw_records(self, limit: int | None = 20) -> list[dict]:
        """What someone with raw database access (e.g. a DBA) can see."""
        q = "SELECT * FROM mb_nodes" + (f" LIMIT {int(limit)}" if limit else "")
        cols = [d[0] for d in self.conn.execute("SELECT * FROM mb_nodes LIMIT 0").description]
        out = []
        for r in self.conn.execute(q).fetchall():
            out.append(
                {c: (v.hex() if isinstance(v, bytes) else v) for c, v in zip(cols, r, strict=True)}
            )
        return out

    def record_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM mb_nodes").fetchone()[0]

    def read_forward_with_key(self, user_id: str, key: bytes) -> list[DecryptedChunk]:
        """
        Attacker model for key-exposure experiments: someone holding ONE node key
        (and the database) scans for the node that key opens, then follows the
        chain forward. Earlier nodes are unreachable: K_{i-1} cannot be derived
        from K_i (HKDF is one-way and the chain only points forward).
        """
        for row in self.conn.execute("SELECT * FROM mb_nodes").fetchall():
            try:
                node = self._load_node(row, key, user_id)
            except IntegrityError:
                continue
            chain = self._follow(user_id, key, node.addr)
            return [self._to_chunk(n, user_id) for n, _ in chain]
        return []
