"""
Session management: the server-side half of "user authentication before the
retrieval phase" (paper §3.4).

A successful store.unlock() yields a UserSession holding the user's derived
keys. This service keeps it under an opaque random bearer token with a TTL and
drops the keys on logout/expiry.

By default sessions live in process memory only (keys are never written to
disk, but they are lost on restart). Persistence is opt-in: with a server
secret the sessions are stored in one AES-GCM encrypted file so a restart does
not log everyone out. That trades some security for convenience: anyone who has
both the file and the secret can recover the keys of logged-in users.

Failed logins are throttled per (user id, client) so that an attacker cannot lock
a victim out from another address, plus a per-client cap across all user ids.
"""

import hashlib
import json
import os
import secrets
import threading
import time

from privrag import crypto as cp
from privrag.store import AuthenticationError, UserSession

_FILE_MAGIC = b"PRS1"
_PRUNE_AT = 10_000  # failure-table size that triggers removal of expired entries


class SessionManager:
    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_failures: int = 5,
        lockout_seconds: int = 60,
        max_client_failures: int = 20,
        clock=time.time,
        persist_path: str | None = None,
        secret: str | None = None,
    ):
        self.ttl = ttl_seconds
        self.max_failures = max_failures
        self.max_client_failures = max_client_failures
        self.lockout = lockout_seconds
        self._clock = clock
        # token hash -> (session, expiry); raw tokens are never stored
        self._sessions: dict[str, tuple[UserSession, float]] = {}
        self._failures: dict[tuple[str, str], tuple[int, float]] = {}
        self._client_failures: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

        self._path = persist_path if (persist_path and secret) else None
        self._file_key = (
            # memory-hard: the file holds live user keys, so a weak secret must be costly to guess
            cp.scrypt_derive(secret, b"privrag/sessions/file-key/v2")
            if secret
            else b""
        )
        if self._path:
            self._load()

    # -- login throttling --------------------------------------------------
    def _locked(self, entry: tuple[int, float] | None, limit: int) -> bool:
        if entry is None:
            return False
        n, until = entry
        return n >= limit and self._clock() < until

    def check_not_locked(self, user_id: str, client: str = "") -> None:
        with self._lock:
            if self._locked(self._failures.get((user_id, client)), self.max_failures) or (
                client and self._locked(self._client_failures.get(client), self.max_client_failures)
            ):
                raise AuthenticationError("Too many failed attempts")

    def record_failure(self, user_id: str, client: str = "") -> None:
        with self._lock:
            now = self._clock()
            until = now + self.lockout
            n, prev = self._failures.get((user_id, client), (0, 0.0))
            if prev <= now:  # the previous window has lapsed: start counting afresh
                n = 0
            self._failures[(user_id, client)] = (n + 1, until)
            if client:
                c, prev_c = self._client_failures.get(client, (0, 0.0))
                if prev_c <= now:
                    c = 0
                self._client_failures[client] = (c + 1, until)
            if len(self._failures) > _PRUNE_AT or len(self._client_failures) > _PRUNE_AT:
                now = self._clock()
                self._failures = {k: v for k, v in self._failures.items() if v[1] > now}
                self._client_failures = {
                    k: v for k, v in self._client_failures.items() if v[1] > now
                }

    def record_success(self, user_id: str, client: str = "") -> None:
        with self._lock:
            self._failures.pop((user_id, client), None)

    # -- sessions ----------------------------------------------------------
    @staticmethod
    def _h(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create(self, session: UserSession) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[self._h(token)] = (session, self._clock() + self.ttl)
            self._persist()
        return token

    def get(self, token: str) -> UserSession | None:
        key = self._h(token)
        with self._lock:
            entry = self._sessions.get(key)
            if entry is None:
                return None
            session, expires = entry
            if self._clock() >= expires:
                session.close()
                del self._sessions[key]
                self._persist()
                return None
            return session

    def revoke(self, token: str) -> bool:
        with self._lock:
            entry = self._sessions.pop(self._h(token), None)
            if entry:
                self._persist()
        if entry:
            entry[0].close()
            return True
        return False

    # -- optional encrypted persistence ------------------------------------
    def _persist(self) -> None:
        """Rewrite the encrypted session file (caller holds the lock)."""
        if not self._path:
            return
        now = self._clock()
        payload = {
            h: {
                "uid": s.user_id,
                "keys": {name: k.hex() for name, k in s.export().items()},
                "exp": exp,
            }
            for h, (s, exp) in self._sessions.items()
            if exp > now
        }
        nonce = cp.random_bytes(cp.NONCE_BYTES)
        blob = cp.aes_gcm_encrypt(
            self._file_key, nonce, json.dumps(payload).encode(), b"privrag-sessions"
        )
        tmp = self._path + ".tmp"
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(_FILE_MAGIC + nonce + blob)
        os.replace(tmp, self._path)

    def _load(self) -> None:
        """Restore sessions after a restart. A missing, foreign or corrupt file means none."""
        try:
            with open(self._path, "rb") as f:
                raw = f.read()
            if raw[:4] != _FILE_MAGIC:
                return
            nonce, blob = raw[4 : 4 + cp.NONCE_BYTES], raw[4 + cp.NONCE_BYTES :]
            data = json.loads(
                cp.aes_gcm_decrypt(self._file_key, nonce, blob, b"privrag-sessions").decode()
            )
            now = self._clock()
            for h, rec in data.items():
                if rec["exp"] > now:
                    keys = {name: bytes.fromhex(v) for name, v in rec["keys"].items()}
                    self._sessions[h] = (UserSession(rec["uid"], keys), rec["exp"])
        except (OSError, ValueError, KeyError, TypeError, cp.CryptoError):
            self._sessions.clear()
