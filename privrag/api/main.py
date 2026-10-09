from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import numpy as np

from privrag.store import ChunkRecord, AuthenticationError, IntegrityError, IsolationViolation
from privrag.store.plaintext import PlaintextStore
from privrag.rag.retriever import Retriever
from privrag.rag.generator import EchoLLM, build_prompt

# 1. Scaffolding & Shared State
app = FastAPI(title="Secure and Isolated Knowledge Retrieval")
store = PlaintextStore("plaintext_rag.db")
retriever = Retriever()
llm = EchoLLM()

# Person 1 Auth Stub: Replace this map with auth/service.py when Person 1 finishes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
FAKE_TOKEN_MAP = {}

def get_current_session(token: str = Depends(oauth2_scheme)):
    if token not in FAKE_TOKEN_MAP:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return FAKE_TOKEN_MAP[token]

# 2. Pydantic Schemas
class UserAuth(BaseModel):
    user_id: str
    password: str

class DocumentInput(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}

class QueryRequest(BaseModel):
    question: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# 3. Endpoints
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserAuth):
    try:
        store.register_user(user.user_id, user.password)
        return {"msg": "User registered successfully"}
    except IntegrityError:
        raise HTTPException(status_code=400, detail="User already exists")

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        # FastAPI's OAuth2 form uses 'username' by default
        session = store.unlock(form_data.username, form_data.password)
        
        # Stub logic: generate a fake token and hold the session in memory
        fake_token = str(uuid.uuid4())
        FAKE_TOKEN_MAP[fake_token] = session
        
        return {"access_token": fake_token, "token_type": "bearer"}
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/documents")
def add_documents(docs: List[DocumentInput], session = Depends(get_current_session)):
    try:
        records = [ChunkRecord(text=d.text, metadata=d.metadata) for d in docs]
        inserted_ids = store.add_chunks(session, records)
        return {"inserted_ids": inserted_ids}
    except IntegrityError as e:
        raise HTTPException(status_code=500, detail="Database integrity error")

@app.get("/documents")
def list_documents(session = Depends(get_current_session)):
    chunks = store.load_user_data(session)
    return [{"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks]

@app.delete("/documents/{chunk_id}")
def delete_document(chunk_id: str, session = Depends(get_current_session)):
    try:
        store.delete_chunk(session, chunk_id)
        return {"msg": f"Deleted {chunk_id}"}
    except IsolationViolation:
        raise HTTPException(status_code=403, detail="Isolation violation: Cannot delete another user's chunk")

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest, session = Depends(get_current_session)):
    try:
        # 1. Load user's private data
        private_chunks = store.load_user_data(session)
        
        # 2. In a real system, embed the question here. We use a dummy vector for API testing.
        dummy_query_emb = np.zeros(384, dtype=np.float32)
        
        # 3. Retrieve and isolate
        retrieved = retriever.retrieve(
            query_emb=dummy_query_emb, 
            user_id=session.user_id, 
            private_chunks=private_chunks, 
            top_k=req.k
        )
        
        # 4. Generate answer with EchoLLM
        prompt = build_prompt(req.question, retrieved)
        answer = llm.generate(prompt)
        
        sources = [{"chunk_id": r.chunk_id, "score": r.score, "source": r.source} for r in retrieved]
        return QueryResponse(answer=answer, sources=sources)
        
    except IsolationViolation:
        raise HTTPException(status_code=403, detail="Security protocol triggered: Isolation violation detected.")