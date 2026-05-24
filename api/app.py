"""
api/app.py  —  FastAPI REST API
================================
Endpoints:
  POST /api/v1/query          — main RAG query
  GET  /api/v1/health         — health check + index stats
  GET  /api/v1/users/{id}     — user profile
  GET  /api/v1/audit          — audit log (admin only)
  POST /api/v1/documents      — ingest new document (admin only)
"""
from __future__ import annotations
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import time

from src.rag_engine import EnterpriseRAGPipeline, Document, User
from dataset import DOCUMENTS, USERS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag.api")

# ── app setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG Intelligence API",
    description="Production-grade RAG with RBAC, hybrid retrieval, and citation support.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── global pipeline ────────────────────────────────────────────────────────────

pipeline = EnterpriseRAGPipeline()

@app.on_event("startup")
async def startup():
    log.info("Starting Enterprise RAG API — loading index…")
    for doc in DOCUMENTS:
        pipeline.add_document(doc)
    stats = pipeline.build_index()
    for user in USERS:
        pipeline.register_user(user)
    log.info(f"Ready: {stats['documents']} docs, {stats['chunks']} chunks, "
             f"{stats['vocab_size']} vocab terms, {len(USERS)} users")

# ── request middleware (latency logging) ───────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"{request.method} {request.url.path} → {response.status_code} [{ms}ms]")
    return response

# ── auth helper ────────────────────────────────────────────────────────────────

def get_user(x_user_id: str = Header(..., description="User ID (e.g. U001)")):
    user = pipeline.users.get(x_user_id)
    if not user:
        raise HTTPException(status_code=401, detail=f"Unknown user: {x_user_id}")
    return user

def require_admin(user: User = Depends(get_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# ── schemas ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          example="What are the salary bands for Level 3 engineers?")
    top_k: int = Field(default=5, ge=1, le=10)

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    confidence: float
    hallucination_risk: str
    retrieved_count: int
    denied_count: int
    routed_to: Optional[str]
    latency_ms: float
    timestamp: str

class IngestRequest(BaseModel):
    doc_id: str
    title: str
    content: str
    source_type: str = Field(..., pattern="^(pdf|csv|json|text|sql)$")
    department: str
    classification: str = Field(..., pattern="^(public|internal|confidential|restricted)$")
    allowed_roles: list[str] = []
    metadata: dict = {}

# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", tags=["System"])
def health():
    """Health check with index statistics."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "index": {
            "documents": len(pipeline.store.documents) if pipeline.store else 0,
            "chunks":    len(pipeline.store.chunks)    if pipeline.store else 0,
            "vocab":     len(pipeline.embedder.vocab),
            "users":     len(pipeline.users),
        },
        "query_stats": pipeline.stats(),
    }

@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
def query(req: QueryRequest, user: User = Depends(get_user)):
    """
    Main RAG query endpoint.
    - Routes query to the best department
    - Retrieves with hybrid semantic+keyword search + MMR diversity
    - Enforces RBAC (clearance + role whitelist)
    - Returns cited, grounded answer with confidence score
    """
    resp = pipeline.query(user.user_id, req.question, top_k=req.top_k)
    return resp.to_dict()

@app.get("/api/v1/users/{user_id}", tags=["Users"])
def get_user_profile(user_id: str, caller: User = Depends(get_user)):
    """Get user profile. Users can view their own; admins can view anyone."""
    if caller.user_id != user_id and caller.role != "admin":
        raise HTTPException(status_code=403, detail="Cannot view other users' profiles")
    user = pipeline.users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user.user_id, "name": user.name, "role": user.role,
            "department": user.department, "clearance_level": user.clearance_level}

@app.get("/api/v1/audit", tags=["Admin"])
def get_audit_log(admin: User = Depends(require_admin),
                  limit: int = 50, offset: int = 0):
    """Full audit log — admin only."""
    log_slice = pipeline.audit_log[offset: offset + limit]
    return {
        "total": len(pipeline.audit_log),
        "offset": offset,
        "limit": limit,
        "entries": log_slice,
        "stats": pipeline.stats(),
    }

@app.post("/api/v1/documents", tags=["Admin"], status_code=201)
def ingest_document(req: IngestRequest, admin: User = Depends(require_admin)):
    """Ingest a new document into the index — admin only."""
    if req.doc_id in (pipeline.store.documents if pipeline.store else {}):
        raise HTTPException(status_code=409, detail=f"Document {req.doc_id} already exists")
    doc = Document(
        doc_id=req.doc_id, title=req.title, content=req.content,
        source_type=req.source_type, department=req.department,
        classification=req.classification, allowed_roles=req.allowed_roles,
        metadata=req.metadata,
    )
    pipeline.add_document(doc)
    pipeline.store.add(doc)
    return {"status": "indexed", "doc_id": doc.doc_id,
            "chunks_added": sum(1 for c in pipeline.store.chunks if c.doc_id == doc.doc_id)}

@app.get("/api/v1/documents", tags=["Admin"])
def list_documents(admin: User = Depends(require_admin)):
    """List all indexed documents — admin only."""
    if not pipeline.store:
        return {"documents": []}
    return {"documents": [
        {"doc_id": d.doc_id, "title": d.title, "type": d.source_type,
         "department": d.department, "classification": d.classification}
        for d in pipeline.store.documents.values()
    ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
