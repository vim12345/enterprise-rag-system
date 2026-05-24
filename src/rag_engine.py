"""
src/rag_engine.py  —  Enterprise RAG Core
==========================================
Features:
  • TF-IDF + cosine hybrid retrieval (NumPy batch matrix ops)
  • Maximal Marginal Relevance (MMR) diversity reranking
  • Hallucination risk assessment
  • Two-layer RBAC (clearance hierarchy + role whitelist)
  • Claude API answer generation (extractive fallback)
  • Latency tracking + structured audit log
"""
from __future__ import annotations
import os, json, math, hashlib, logging, re, time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

log = logging.getLogger("rag.engine")

# ── constants ──────────────────────────────────────────────────────────────────

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","that","this","it","its",
    "by","from","as","into","not","no","so","if","all","any","both","each",
    "he","she","we","you","i","our","your","his","her","my","me","us","up",
    "out","about","also","per","there","their","they","them","then","when",
    "where","who","which","how","what","these","those","just","than","too",
    "very","only","same","such","through","during","before","after","above",
}

CLEARANCE_RANK = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}

DEPT_SIGNALS: dict[str, list[str]] = {
    "hr":          ["salary","payroll","compensation","bonus","leave","employee",
                    "headcount","attrition","hiring","onboarding","benefits",
                    "performance","promotion","termination","workforce"],
    "finance":     ["revenue","budget","expense","invoice","financial","profit",
                    "ebitda","arr","cash","vendor","quarter","fiscal","q1","q2",
                    "margin","cost","income","payment","billing","accounts"],
    "engineering": ["deploy","api","server","incident","outage","architecture",
                    "database","redis","kubernetes","latency","slo","bug","code",
                    "docker","microservice","uptime","pipeline","devops"],
    "legal":       ["contract","compliance","soc2","audit","regulation","policy",
                    "gdpr","ccpa","breach","privacy","legal","retention","law",
                    "clause","agreement","violation","liability"],
}

LLM_SYSTEM_PROMPT = """You are a secure enterprise AI assistant. Rules:
1. Answer ONLY from the provided [Source N] context blocks. Never use prior knowledge.
2. Cite every factual claim inline as [Source N].
3. If context is insufficient say: "I cannot find sufficient information in the available data."
4. Never reveal documents the user cannot access.
5. Be concise, factual, professional. Use bullet points for lists."""

# ── models ─────────────────────────────────────────────────────────────────────

@dataclass
class Document:
    doc_id: str; title: str; content: str
    source_type: str; department: str; classification: str
    allowed_roles: list[str]
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class User:
    user_id: str; name: str; role: str; department: str; clearance_level: str

@dataclass
class Chunk:
    chunk_id: str; doc_id: str; title: str; content: str
    source_type: str; department: str; classification: str
    chunk_index: int; word_count: int; embedding: np.ndarray

@dataclass
class RetrievedChunk:
    chunk_id: str; doc_id: str; title: str; content: str
    score: float; source_type: str; department: str
    classification: str; chunk_index: int

@dataclass
class RAGResponse:
    query: str; answer: str; sources: list[dict]
    confidence: float; hallucination_risk: str
    user_id: str; role: str
    retrieved_count: int; denied_count: int
    routed_to: Optional[str]; latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

# ── vocab embedder ─────────────────────────────────────────────────────────────

class VocabEmbedder:
    """
    Corpus TF-IDF embedder with precomputed IDF array.
    Production swap: replace transform() with sentence-transformers / OpenAI ada-002.
    """
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.doc_freq: dict[str, int] = {}
        self.n_docs = 0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    def fit(self, documents: list[Document]):
        self.n_docs = len(documents)
        for doc in documents:
            for t in set(self.tokenize(doc.content)):
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        for t in self.doc_freq:
            self.vocab[t] = len(self.vocab)
        V = len(self.vocab)
        self.idf = np.zeros(V, dtype=np.float32)
        for t, idx in self.vocab.items():
            self.idf[idx] = math.log((self.n_docs + 1) / (self.doc_freq[t] + 1)) + 1.0
        log.info(f"VocabEmbedder: {V} terms across {self.n_docs} docs")

    def transform(self, text: str) -> np.ndarray:
        tokens = self.tokenize(text)
        V = len(self.vocab)
        if not tokens or V == 0:
            return np.zeros(max(V, 1), dtype=np.float32)
        vec = np.zeros(V, dtype=np.float32)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t, cnt in tf.items():
            if t in self.vocab:
                vec[self.vocab[t]] = (cnt / len(tokens)) * self.idf[self.vocab[t]]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

# ── document store ─────────────────────────────────────────────────────────────

class DocumentStore:
    def __init__(self, embedder: VocabEmbedder, chunk_size=200, overlap=40):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self._matrix: Optional[np.ndarray] = None

    def _split(self, text: str) -> list[str]:
        words = text.split()
        step = max(1, self.chunk_size - self.overlap)
        return [" ".join(words[i:i+self.chunk_size])
                for i in range(0, len(words), step)
                if words[i:i+self.chunk_size]]

    def add(self, doc: Document):
        self.documents[doc.doc_id] = doc
        for idx, text in enumerate(self._split(doc.content)):
            cid = hashlib.md5(f"{doc.doc_id}:{idx}".encode()).hexdigest()[:10]
            self.chunks.append(Chunk(
                chunk_id=cid, doc_id=doc.doc_id, title=doc.title,
                content=text, source_type=doc.source_type,
                department=doc.department, classification=doc.classification,
                chunk_index=idx, word_count=len(text.split()),
                embedding=self.embedder.transform(text),
            ))
        self._matrix = None

    def matrix(self) -> np.ndarray:
        if self._matrix is None or self._matrix.shape[0] != len(self.chunks):
            self._matrix = np.vstack([c.embedding for c in self.chunks])
        return self._matrix

# ── RBAC ───────────────────────────────────────────────────────────────────────

class RBACEngine:
    def can_access(self, user: User, doc: Document) -> bool:
        if CLEARANCE_RANK.get(user.clearance_level, 0) < CLEARANCE_RANK.get(doc.classification, 99):
            return False
        if doc.allowed_roles and user.role not in doc.allowed_roles and user.role != "admin":
            return False
        return True

    def filter(self, user: User, chunks: list[RetrievedChunk],
               docs: dict[str, Document]) -> tuple[list[RetrievedChunk], int]:
        allowed, denied = [], 0
        for ch in chunks:
            doc = docs.get(ch.doc_id)
            if doc and self.can_access(user, doc):
                allowed.append(ch)
            else:
                denied += 1
                log.warning(f"RBAC DENY user={user.user_id} doc={ch.doc_id} class={ch.classification}")
        return allowed, denied

# ── hybrid retriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Stage 1: Hybrid score = 0.6×cosine + 0.4×BM25-style keyword overlap
    Stage 2: MMR reranking for result diversity (λ=0.7)
    """
    def __init__(self, store: DocumentStore, embedder: VocabEmbedder):
        self.store = store
        self.embedder = embedder

    def route(self, query: str) -> Optional[str]:
        tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        scores = {d: sum(1 for kw in kws if kw in tokens) for d, kws in DEPT_SIGNALS.items()}
        best, score = max(scores.items(), key=lambda x: x[1])
        return best if score >= 2 else None

    @staticmethod
    def _kw(q_tokens: set[str], chunk: Chunk) -> float:
        c_tokens = set(re.findall(r"[a-zA-Z0-9]+", chunk.content.lower()))
        return len(q_tokens & c_tokens) / (math.sqrt(len(q_tokens) * max(len(c_tokens), 1)) + 1e-9)

    @staticmethod
    def _mmr(candidates: list[tuple[float, Chunk]], top_k: int, lam=0.7) -> list[tuple[float, Chunk]]:
        """Maximal Marginal Relevance — balances relevance with diversity."""
        if not candidates:
            return []
        selected: list[tuple[float, Chunk]] = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda x: x[0])
                selected.append(best); remaining.remove(best); continue
            sel_embs = np.vstack([c.embedding for _, c in selected])
            scored = []
            for score, chunk in remaining:
                q_n = chunk.embedding / (np.linalg.norm(chunk.embedding) + 1e-9)
                sim = float(np.max(sel_embs @ q_n))
                scored.append((lam * score - (1 - lam) * sim, score, chunk))
            best = max(scored, key=lambda x: x[0])
            selected.append((best[1], best[2]))
            remaining = [(s, c) for s, c in remaining if c.chunk_id != best[2].chunk_id]
        return selected

    def retrieve(self, query: str, top_k=8,
                 dept_filter: Optional[str] = None,
                 use_mmr=True) -> list[RetrievedChunk]:
        q_vec = self.embedder.transform(query)
        q_tokens = set(VocabEmbedder.tokenize(query))
        mat = self.store.matrix()
        if mat.size == 0:
            return []

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        safe = mat / np.where(norms > 0, norms, 1.0)
        qn = np.linalg.norm(q_vec)
        cos = safe @ (q_vec / qn if qn > 0 else q_vec)

        candidates = []
        for i, chunk in enumerate(self.store.chunks):
            if dept_filter and chunk.department != dept_filter:
                continue
            h = 0.6 * float(cos[i]) + 0.4 * self._kw(q_tokens, chunk)
            if h > 0.005:
                candidates.append((h, chunk))

        candidates.sort(key=lambda x: x[0], reverse=True)
        final = self._mmr(candidates[:top_k*4], top_k) if use_mmr else candidates[:top_k]

        return [RetrievedChunk(
            chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title,
            content=c.content, score=round(s, 5), source_type=c.source_type,
            department=c.department, classification=c.classification,
            chunk_index=c.chunk_index,
        ) for s, c in final]

# ── hallucination guard ────────────────────────────────────────────────────────

class HallucinationGuard:
    def assess(self, confidence: float, n_sources: int, query: str, answer: str) -> str:
        q_tok = set(VocabEmbedder.tokenize(query))
        a_tok = set(VocabEmbedder.tokenize(answer))
        overlap = len(q_tok & a_tok) / (len(q_tok) + 1)
        risk = 0
        if confidence < 0.3: risk += 2
        elif confidence < 0.6: risk += 1
        if n_sources == 0: risk += 3
        elif n_sources < 2: risk += 1
        if overlap < 0.1: risk += 1
        return "HIGH" if risk >= 3 else "MEDIUM" if risk >= 1 else "LOW"

# ── answer generator ───────────────────────────────────────────────────────────

class AnswerGenerator:
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = "claude-sonnet-4-20250514"
        self.guard = HallucinationGuard()
        mode = "Claude API" if self.api_key else "extractive fallback"
        log.info(f"AnswerGenerator: {mode}")

    def _context(self, chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
        parts, sources = [], []
        for i, ch in enumerate(chunks[:6], 1):
            parts.append(f"[Source {i} | {ch.title} | {ch.source_type.upper()} "
                         f"| {ch.department} | {ch.classification}]\n{ch.content}")
            sources.append({"source_id": i, "doc_id": ch.doc_id, "title": ch.title,
                             "type": ch.source_type, "department": ch.department,
                             "classification": ch.classification,
                             "relevance_score": ch.score, "chunk_index": ch.chunk_index})
        return "\n\n---\n\n".join(parts), sources

    def _llm(self, query: str, context: str, user: User) -> Optional[str]:
        try:
            import urllib.request
            payload = json.dumps({
                "model": self.model, "max_tokens": 1024,
                "system": LLM_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content":
                    f"User: {user.name} | Role: {user.role} | Clearance: {user.clearance_level}\n\n"
                    f"CONTEXT:\n{context}\n\nQUESTION: {query}"}]
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=payload,
                headers={"Content-Type": "application/json",
                         "x-api-key": self.api_key,
                         "anthropic-version": "2023-06-01"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())["content"][0]["text"]
        except Exception as e:
            log.warning(f"LLM call failed: {e}")
            return None

    def _extractive(self, chunks: list[RetrievedChunk], user: User) -> str:
        top = chunks[0]
        lines = [f"**[Source 1 — {top.title}]** ({top.source_type.upper()}, "
                 f"{top.department}, {top.classification})\n"
                 f"{top.content[:500].rsplit(' ', 1)[0]}…"]
        if len(chunks) > 1:
            lines.append("\n**Supporting sources:**")
            for i, ch in enumerate(chunks[1:4], 2):
                lines.append(f"• [Source {i}] **{ch.title}** — "
                              + ch.content[:180].rsplit(" ", 1)[0] + "…")
        lines.append(f"\n_Role: **{user.role}** | Clearance: **{user.clearance_level}**_")
        return "\n".join(lines)

    def generate(self, query: str, chunks: list[RetrievedChunk],
                 user: User) -> tuple[str, float, str, list[dict]]:
        if not chunks:
            return ("⚠️ No accessible documents matched your query. "
                    "Contact your administrator if you need broader access.",
                    0.0, "HIGH", [])
        context, sources = self._context(chunks)
        answer = (self._llm(query, context, user) if self.api_key else None) \
                 or self._extractive(chunks, user)
        confidence = round(min(sum(c.score for c in chunks[:3]) / min(3, len(chunks)) * 5.0, 1.0), 2)
        risk = self.guard.assess(confidence, len(sources), query, answer)
        return answer, confidence, risk, sources

# ── pipeline ───────────────────────────────────────────────────────────────────

class EnterpriseRAGPipeline:
    def __init__(self):
        self.embedder = VocabEmbedder()
        self._raw_docs: list[Document] = []
        self.store: Optional[DocumentStore] = None
        self.retriever: Optional[HybridRetriever] = None
        self.rbac = RBACEngine()
        self.generator = AnswerGenerator()
        self.users: dict[str, User] = {}
        self.audit_log: list[dict] = []

    def add_document(self, doc: Document):
        self._raw_docs.append(doc)

    def build_index(self) -> dict:
        log.info(f"Building index: {len(self._raw_docs)} docs…")
        self.embedder.fit(self._raw_docs)
        self.store = DocumentStore(self.embedder)
        for doc in self._raw_docs:
            self.store.add(doc)
        self.retriever = HybridRetriever(self.store, self.embedder)
        return {"documents": len(self._raw_docs), "chunks": len(self.store.chunks),
                "vocab_size": len(self.embedder.vocab)}

    def register_user(self, user: User):
        self.users[user.user_id] = user

    def query(self, user_id: str, question: str, top_k=6) -> RAGResponse:
        t0 = time.perf_counter()
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"Unknown user: {user_id}")
        dept = self.retriever.route(question)
        cands = self.retriever.retrieve(question, top_k=top_k*3, dept_filter=dept)
        if len(cands) < 2 and dept:
            cands = self.retriever.retrieve(question, top_k=top_k*3)
        allowed, denied = self.rbac.filter(user, cands, self.store.documents)
        answer, conf, risk, sources = self.generator.generate(question, allowed[:top_k], user)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        resp = RAGResponse(query=question, answer=answer, sources=sources,
                           confidence=conf, hallucination_risk=risk,
                           user_id=user_id, role=user.role,
                           retrieved_count=len(allowed), denied_count=denied,
                           routed_to=dept, latency_ms=latency)
        self.audit_log.append({
            "ts": resp.timestamp, "user": user.name, "role": user.role,
            "query": question, "dept_hint": dept, "retrieved": len(allowed),
            "denied": denied, "confidence": conf, "risk": risk,
            "latency_ms": latency, "sources": [s["doc_id"] for s in sources],
        })
        return resp

    def stats(self) -> dict:
        if not self.audit_log:
            return {}
        n = len(self.audit_log)
        return {
            "total_queries": n,
            "avg_confidence": round(sum(e["confidence"] for e in self.audit_log) / n, 2),
            "avg_latency_ms": round(sum(e["latency_ms"] for e in self.audit_log) / n, 1),
            "total_denied": sum(e["denied"] for e in self.audit_log),
            "low_risk_pct": round(sum(1 for e in self.audit_log if e["risk"] == "LOW") / n * 100, 1),
        }

    def export_audit(self, path="audit_log.json"):
        with open(path, "w") as f:
            json.dump({"log": self.audit_log, "stats": self.stats()}, f, indent=2)
        log.info(f"Audit log → {path}")
