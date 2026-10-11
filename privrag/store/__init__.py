from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class ChunkRecord:
    text: str
    metadata: dict
    # float32 (d,), L2-normalised. Required by every store: the paper encrypts
    # the embedding together with the chunk (Method A, Eq. 10).
    embedding: np.ndarray | None = None


@dataclass
class DecryptedChunk:
    chunk_id: str
    text: str
    embedding: np.ndarray  # float32 (d,), L2-normalised
    metadata: dict


class UserSession:
    """
    Result of a successful unlock().

    Holds the user's id and (for encrypted stores) the derived key material.
    Keys live only in process memory; they are never persisted or logged.
    """

    def __init__(self, user_id: str | None = None, keys: dict | None = None):
        self.user_id = user_id
        self._keys = dict(keys) if keys else {}

    def key(self, name: str) -> bytes:
        try:
            return self._keys[name]
        except KeyError:
            raise AuthenticationError("Session holds no key material") from None

    def export(self) -> dict[str, bytes]:
        """Key material for the opt-in encrypted session file (privrag/sessions.py only)."""
        return dict(self._keys)

    def close(self) -> None:
        """Drop key references (logout / expiry)."""
        self._keys.clear()

    def __repr__(self) -> str:  # never print key material
        return f"UserSession(user_id={self.user_id!r})"


class PrivateStore(Protocol):
    def register_user(self, user_id: str, password: str) -> None: ...

    def unlock(self, user_id: str, password: str) -> UserSession: ...  # Raises AuthenticationError

    def add_chunks(
        self, s: UserSession, chunks: list[ChunkRecord]
    ) -> list[str]: ...  # Raises IntegrityError

    def load_user_data(self, s: UserSession) -> list[DecryptedChunk]: ...  # Raises IntegrityError

    def delete_chunk(self, s: UserSession, chunk_id: str) -> None: ...  # Raises IsolationViolation


class AuthenticationError(Exception):
    pass


class IntegrityError(Exception):
    pass


class IsolationViolation(Exception):
    pass
