import sqlite3
import json
import uuid
import numpy as np
from typing import List
from privrag.store import (
    PrivateStore, ChunkRecord, DecryptedChunk, UserSession,
    AuthenticationError, IntegrityError, IsolationViolation
)

class PlaintextStore(PrivateStore):
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    password TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    owner TEXT,
                    text TEXT,
                    embedding BLOB,
                    metadata TEXT,
                    FOREIGN KEY(owner) REFERENCES users(user_id)
                )
            """)

    def register_user(self, user_id: str, password: str) -> None:
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO users (user_id, password) VALUES (?, ?)", 
                    (user_id, password)
                )
        except sqlite3.IntegrityError:
            raise IntegrityError("User already exists")

    def unlock(self, user_id: str, password: str) -> UserSession:
        cur = self.conn.execute("SELECT password FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if not row or row[0] != password:
            raise AuthenticationError("Invalid credentials")
        
        session = UserSession()
        session.user_id = user_id
        return session

    def add_chunks(self, s: UserSession, chunks: List[ChunkRecord]) -> List[str]:
        inserted_ids = []
        with self.conn:
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                dummy_embedding = np.zeros(384, dtype=np.float32).tobytes()
                
                # ADD THIS LINE: Explicitly tag the metadata with the owner
                chunk.metadata["owner"] = s.user_id 
                
                meta_json = json.dumps(chunk.metadata)
                
                self.conn.execute(
                    "INSERT INTO chunks (chunk_id, owner, text, embedding, metadata) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, s.user_id, chunk.text, dummy_embedding, meta_json)
                )
                inserted_ids.append(chunk_id)
        return inserted_ids

    
    def load_user_data(self, s: UserSession) -> List[DecryptedChunk]:
        cur = self.conn.execute(
            "SELECT chunk_id, text, embedding, metadata FROM chunks WHERE owner = ?", 
            (s.user_id,)
        )
        results = []
        for row in cur.fetchall():
            emb_array = np.frombuffer(row[2], dtype=np.float32)
            results.append(DecryptedChunk(
                chunk_id=row[0],
                text=row[1],
                embedding=emb_array,
                metadata=json.loads(row[3])
            ))
        return results

    def delete_chunk(self, s: UserSession, chunk_id: str) -> None:
        # First verify the chunk belongs to the user to enforce cross-user isolation
        cur = self.conn.execute("SELECT owner FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        if not row:
            return # Idempotent delete
        if row[0] != s.user_id:
            raise IsolationViolation("Attempted to delete a chunk owned by another user")
            
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))