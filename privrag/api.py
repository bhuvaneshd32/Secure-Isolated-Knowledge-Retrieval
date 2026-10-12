import logging
import os
import threading
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from starlette.middleware.base import BaseHTTPMiddleware

from privrag.rag.generator import LLMError, get_llm
from privrag.rag.ingest import DocumentLoader, get_embedder
from privrag.rag.pipeline import RagService
from privrag.rag.public_kb import load_public_kb
from privrag.rag.retriever import Retriever
from privrag.sessions import SessionManager
from privrag.store import AuthenticationError, IntegrityError, IsolationViolation
from privrag.store.factory import make_store


# 1. Settings (env: PRIVRAG_STORE, PRIVRAG_DB_PATH, PRIVRAG_EMBEDDER, ...)
class Settings(BaseSettings):
    privrag_store: str = "A"  # plaintext | A | B | B_improved
    privrag_db_path: str | None = None
    privrag_embedder: str = "auto"  # auto | minilm | hash
    privrag_public_kb: str = "data/public_kb.jsonl"
    privrag_llm: str = "ollama"  # ollama | echo (test double)
    privrag_session_ttl: int = 1800
    # Optional: keep sessions across restarts in an AES-GCM encrypted file. Leave the
    # secret unset (default) to keep keys in memory only.
    privrag_session_secret: str | None = None
    privrag_session_file: str = "data/sessions.enc"


# 2. Log redaction: drop any log record that mentions a canary token
class RedactingFilter(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        return "CANARY" not in msg


def install_log_redaction() -> None:
    """
    Logger-level filters do not apply to records from child loggers, so the filter is
    attached to the handlers (root and uvicorn's), which see every record.
    """
    flt = RedactingFilter()
    for name in ("", "uvicorn", "uvicorn.access", "uvicorn.error", "privrag"):
        lg = logging.getLogger(name)
        lg.addFilter(flt)
        for h in lg.handlers:
            if not any(isinstance(f, RedactingFilter) for f in h.filters):
                h.addFilter(flt)


# 3. Request size limit
MAX_BODY = 2 * 1024 * 1024
_READ_CHUNK = 64 * 1024


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is None and "chunked" in request.headers.get("transfer-encoding", "").lower():
            # Without a declared length the size cannot be checked before reading the body.
            return JSONResponse(status_code=411, content={"detail": "Content-Length required"})
        try:
            too_big = cl is not None and int(cl) > MAX_BODY
        except ValueError:
            too_big = True
        if too_big:
            return JSONResponse(status_code=413, content={"detail": "Payload Too Large"})
        return await call_next(request)


# 4. Schemas
class UserAuth(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=8, max_length=256)


class DocumentInput(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    metadata: dict[str, Any] = {}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=5, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


def create_app(store=None, embedder=None, llm=None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    install_log_redaction()
    app = FastAPI(title="Secure and Isolated Knowledge Retrieval")
    app.add_middleware(RequestSizeLimitMiddleware)

    store = (
        store if store is not None else make_store(settings.privrag_store, settings.privrag_db_path)
    )
    embedder = embedder if embedder is not None else get_embedder(settings.privrag_embedder)
    sessions = SessionManager(
        ttl_seconds=settings.privrag_session_ttl,
        persist_path=settings.privrag_session_file,
        secret=settings.privrag_session_secret,
    )
    _svc_lock = threading.Lock()
    _svc: dict[str, RagService] = {}

    def service() -> RagService:
        # Built lazily so importing the app never loads a model.
        with _svc_lock:
            if "s" not in _svc:
                embs, meta = load_public_kb(embedder, settings.privrag_public_kb)
                _svc["s"] = RagService(
                    store, embedder, Retriever(embs, meta), llm or get_llm(settings.privrag_llm)
                )
            return _svc["s"]

    app.state.store = store
    app.state.sessions = sessions
    app.state.service = service

    # Error mapping ---------------------------------------------------------
    @app.exception_handler(AuthenticationError)
    async def _auth(request: Request, exc: AuthenticationError):
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    @app.exception_handler(IntegrityError)
    async def _integrity(request: Request, exc: IntegrityError):
        # Generic message: no decryption/padding/MAC oracle for the client.
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.exception_handler(IsolationViolation)
    async def _isolation(request: Request, exc: IsolationViolation):
        return JSONResponse(status_code=403, content={"detail": "Security protocol triggered"})

    @app.exception_handler(LLMError)
    async def _llm(request: Request, exc: LLMError):
        return JSONResponse(status_code=503, content={"detail": "Language model unavailable"})

    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

    def get_current_session(token: str = Depends(oauth2_scheme)):
        session = sessions.get(token)
        if session is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return session

    # Endpoints -------------------------------------------------------------
    @app.get("/health")
    def health_check():
        return {"status": "ok", "store": settings.privrag_store, "llm": settings.privrag_llm}

    @app.post("/register", status_code=status.HTTP_201_CREATED)
    def register(user: UserAuth):
        try:
            store.register_user(user.user_id, user.password)
        except IntegrityError:
            raise HTTPException(status_code=400, detail="User already exists") from None
        return {"msg": "User registered successfully"}

    @app.post("/login")
    def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
        # Throttle per (user id, client address): a stranger cannot lock a user out of
        # their own account from another address. Behind a reverse proxy the client
        # address is the proxy's unless it is configured to pass the real one on.
        uid = form_data.username
        client = request.client.host if request.client else ""
        try:
            sessions.check_not_locked(uid, client)
        except AuthenticationError:
            raise HTTPException(status_code=429, detail="Too many failed attempts") from None
        try:
            session = store.unlock(uid, form_data.password)
        except AuthenticationError:
            sessions.record_failure(uid, client)
            raise HTTPException(status_code=401, detail="Invalid credentials") from None
        sessions.record_success(uid, client)
        return {"access_token": sessions.create(session), "token_type": "bearer"}

    @app.post("/logout")
    def logout(token: str = Depends(oauth2_scheme)):
        sessions.revoke(token)
        return {"msg": "Logged out"}

    @app.post("/documents")
    def add_documents(docs: list[DocumentInput], session=Depends(get_current_session)):
        ids = service().add_texts(session, [d.text for d in docs], [d.metadata for d in docs])
        return {"inserted_ids": ids}

    @app.post("/documents/upload")
    async def upload_document(file: UploadFile = File(...), session=Depends(get_current_session)):
        buf = bytearray()
        while chunk := await file.read(_READ_CHUNK):  # stop as soon as the limit is passed
            buf.extend(chunk)
            if len(buf) > MAX_BODY:
                raise HTTPException(status_code=413, detail="Payload Too Large")
        suffix = os.path.splitext(file.filename or "")[1].lower() or ".txt"
        if suffix not in (".txt", ".md", ".pdf"):
            raise HTTPException(status_code=415, detail="Only .txt, .md and .pdf are supported")
        # Parsed in memory: the user's plaintext is never written to a temp file.
        try:
            text = DocumentLoader.load_bytes(bytes(buf), suffix)
        except Exception:
            raise HTTPException(status_code=422, detail="Could not read the document") from None
        if not text.strip():
            raise HTTPException(status_code=422, detail="Document contains no text")
        ids = service().ingest_document(
            session, text, {"source": os.path.basename(file.filename or "upload")}
        )
        return {"inserted_ids": ids}

    @app.get("/documents")
    def list_documents(session=Depends(get_current_session)):
        chunks = store.load_user_data(session)
        return [{"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks]

    @app.delete("/documents/{chunk_id}")
    def delete_document(chunk_id: str, session=Depends(get_current_session)):
        store.delete_chunk(session, chunk_id)  # IsolationViolation -> 403 via handler
        return {"msg": f"Deleted {chunk_id}"}

    @app.post("/query", response_model=QueryResponse)
    def query_rag(req: QueryRequest, session=Depends(get_current_session)):
        answer, retrieved, _ = service().answer(session, req.question, req.k)
        sources = [
            {"chunk_id": r.chunk_id, "score": r.score, "source": r.source} for r in retrieved
        ]
        return QueryResponse(answer=answer, sources=sources)

    return app


def __getattr__(name: str):
    """`uvicorn privrag.api:app` builds the app on first access, not at import time."""
    if name == "app":
        global _app
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)


_app: FastAPI | None = None
