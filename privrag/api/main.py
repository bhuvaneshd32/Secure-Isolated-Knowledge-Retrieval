import logging
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Dict, Any
import uuid
import numpy as np

from privrag.store import ChunkRecord, AuthenticationError, IntegrityError, IsolationViolation
from privrag.store.plaintext import PlaintextStore
from privrag.rag.retriever import Retriever
from privrag.rag.generator import EchoLLM, build_prompt


# 1. Pydantic Settings for Store Switching
class Settings(BaseSettings):
    privrag_store: str = "plaintext"

settings = Settings()

def get_store(store_type: str):
    if store_type == "plaintext":
        return PlaintextStore("plaintext_rag.db")
    # Person 1 will plug in their encrypted stores here tomorrow:
    # elif store_type == "A": return MethodAStore()
    # elif store_type == "B_paper": return MethodBStore(integrity_mode='paper')
    # elif store_type == "B_hardened": return MethodBStore(integrity_mode='hardened')
    return PlaintextStore("plaintext_rag.db")

app = FastAPI(title="Secure and Isolated Knowledge Retrieval")
store = get_store(settings.privrag_store)
retriever = Retriever()
llm = EchoLLM()

# 2. Log Redaction Filter
class RedactingFilter(logging.Filter):
    def filter(self, record):
        if not isinstance(record.msg, str):
            return True
        # Drops any log containing a canary token to prevent secrets leaking to server logs
        if "CANARY" in record.msg:
            return False
        return True

# Apply filter to uvicorn's access logger
logging.getLogger("uvicorn.access").addFilter(RedactingFilter())

# 3. Request Size Limit Middleware
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        # Enforce a strict 2MB upload limit
        if content_length and int(content_length) > 2 * 1024 * 1024:
            return JSONResponse(status_code=413, content={"detail": "Payload Too Large"})
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware)

# 4. Error Mapping
@app.exception_handler(AuthenticationError)
async def auth_error_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    # Generic 500 with no details to prevent oracle padding attacks during decryption
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.exception_handler(IsolationViolation)
async def isolation_error_handler(request: Request, exc: IsolationViolation):
    return JSONResponse(status_code=403, content={"detail": "Security protocol triggered"})

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