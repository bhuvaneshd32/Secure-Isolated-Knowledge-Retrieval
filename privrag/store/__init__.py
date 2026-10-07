from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass
class ChunkRecord:
    text: str
    metadata: dict

@dataclass
class DecryptedChunk:
    chunk_id: str
    text: str
    embedding: np.ndarray # float32 (d,), L2-normalised
    metadata: dict

class UserSession:
    pass # Person 1 will implement this in auth/service.py

class PrivateStore(Protocol):
    def register_user(self, user_id: str, password: str) -> None:
        ...
        
    def unlock(self, user_id: str, password: str) -> UserSession:
        ... # Raises AuthenticationError
        
    def add_chunks(self, s: UserSession, chunks: list[ChunkRecord]) -> list[str]:
        ... # Raises IntegrityError
        
    def load_user_data(self, s: UserSession) -> list[DecryptedChunk]:
        ... # Raises IntegrityError
        
    def delete_chunk(self, s: UserSession, chunk_id: str) -> None:
        ... # Raises IsolationViolation

class AuthenticationError(Exception): pass
class IntegrityError(Exception): pass
class IsolationViolation(Exception): pass