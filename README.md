# Enterprise RAG Intelligence System

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-TF--IDF-013243?logo=numpy)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![Tests](https://img.shields.io/badge/Tests-17%20passed-brightgreen)
![RBAC](https://img.shields.io/badge/Security-RBAC%20%7C%200%20leaks-red)

A **production-grade Retrieval-Augmented Generation (RAG) system** for enterprise environments — featuring hybrid semantic retrieval, MMR diversity reranking, two-layer RBAC, hallucination risk assessment, a FastAPI REST API, Docker deployment, 17-test suite, and a full evaluation framework with Precision@K, MRR, and NDCG metrics.

---

## Architecture

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────────┐
│              EnterpriseRAGPipeline                     │
│                                                        │
│  Multi-Format     VocabEmbedder      DocumentStore     │
│  Ingestion   ───▶ (TF-IDF, NumPy) ─▶ (chunked+indexed)│
│  PDF·CSV·JSON                               │          │
│                                             ▼          │
│          Query ──────▶ HybridRetriever                 │
│                        Stage 1: 0.6×cosine             │
│                               + 0.4×BM25 keyword       │
│                        Stage 2: MMR diversity rerank   │
│                                             │          │
│                                             ▼          │
│                                       RBAC Engine      │
│                              Layer 1: Clearance rank   │
│                              Layer 2: Role whitelist   │
│                                             │          │
│                                             ▼          │
│                                    AnswerGenerator     │
│                              Claude API / Extractive   │
│                                             │          │
│                                             ▼          │
│                          HallucinationGuard + Audit    │
└────────────────────────────────────────────────────────┘
```

## Screenshots

### Streamlit UI — Query with Source Attribution
![Streamlit UI](screenshots/streamlit_demo.png)

### FastAPI Swagger Docs
![API Docs](screenshots/api_docs.png)

### Evaluation Results
![Evaluation](screenshots/evaluation.png)

---

## Features

| Component | What it does |
|-----------|-------------|
| **Hybrid Retrieval** | TF-IDF cosine similarity (60%) + BM25-style keyword overlap (40%) via NumPy batch matrix ops |
| **MMR Reranking** | Maximal Marginal Relevance removes redundant chunks — balances relevance with diversity |
| **Query Router** | Detects department signals in query (HR/Finance/Engineering/Legal) to scope retrieval |
| **Two-Layer RBAC** | Clearance hierarchy (public→restricted) + per-document role whitelist — 0 unauthorized exposures |
| **Hallucination Guard** | Scores answer risk (LOW/MEDIUM/HIGH) based on confidence, source count, and query-answer overlap |
| **FastAPI REST API** | Full CRUD: `/query`, `/health`, `/audit`, `/documents`, `/users` with auth middleware |
| **Docker** | Single-command deployment with health checks |
| **Evaluation** | Precision@K, Recall@K, MRR, NDCG@K, latency P50/P95/P99, RBAC denial rate |
| **Test Suite** | 17 unit + integration tests — all green |

---

## Project Structure

```
enterprise-rag-system/
├── main.py                      # CLI demo — 10 scenarios end-to-end
├── dataset.py                   # 10 synthetic enterprise documents + 7 users
├── requirements.txt
├── .gitignore
│
├── src/
│   └── rag_engine.py            # Core engine (VocabEmbedder, DocumentStore,
│                                #   HybridRetriever, RBACEngine,
│                                #   HallucinationGuard, AnswerGenerator, Pipeline)
│
├── api/
│   └── app.py                   # FastAPI REST API with auth middleware
│
├── tests/
│   └── test_rag.py              # 17 unit + integration tests
│
├── evaluation/
│   └── evaluate.py              # Retrieval + generation + security metrics
│
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── data/                        # Auto-generated on first run
    ├── users.csv
    ├── access_policies.json
    └── audit_events.json
```

---

## Quick Start

### Option 1 — CLI Demo (no dependencies except numpy)

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-rag-system.git
cd enterprise-rag-system

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

### Option 2 — REST API

```bash
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# API docs: http://localhost:8000/docs
```

### Option 3 — Docker

```bash
cd deployment
docker-compose up --build

# API docs: http://localhost:8000/docs
```

### Option 4 — With Claude LLM answers

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
python main.py
# or
uvicorn api.app:app --port 8000
```

---

## API Usage

### Query (POST /api/v1/query)
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "x-user-id: U002" \
  -d '{"question": "What are the salary bands?", "top_k": 5}'
```

**Response:**
```json
{
  "query": "What are the salary bands?",
  "answer": "**[Source 1 — Employee Compensation Policy]**...",
  "sources": [
    {"source_id": 1, "title": "Employee Compensation Policy", 
     "type": "pdf", "department": "hr", 
     "relevance_score": 0.365}
  ],
  "confidence": 1.0,
  "hallucination_risk": "LOW",
  "retrieved_count": 2,
  "denied_count": 0,
  "routed_to": "hr",
  "latency_ms": 2.6
}
```

### Health (GET /api/v1/health)
```bash
curl http://localhost:8000/api/v1/health
```

### Audit Log — Admin only (GET /api/v1/audit)
```bash
curl http://localhost:8000/api/v1/audit -H "x-user-id: U001"
```

---

## RBAC Model

Two independent layers — **both** must pass:

**Layer 1 — Clearance Hierarchy**
```
guest (0) → engineering/analyst (1) → finance/hr (2) → admin/legal (3)
```

**Layer 2 — Role Whitelist**
Each document has an `allowed_roles` list. User role must appear in it.

| User | Role | Clearance | Salary Doc | SOC2 Audit (Restricted) |
|------|------|-----------|------------|------------------------|
| Alice | admin | restricted | ✅ | ✅ |
| Eve | legal | restricted | ❌ (not in whitelist) | ✅ |
| Bob | hr | confidential | ✅ | ❌ |
| Dave | engineering | internal | ❌ (clearance) | ❌ |
| Grace | guest | public | ❌ | ❌ |

---

## Evaluation Results

```
RETRIEVAL METRICS (K=5)
  Precision@5 : 0.645
  Recall@5    : 1.700
  MRR         : 0.833
  NDCG@5      : 0.799

GENERATION METRICS
  Avg Confidence  : 65%
  Conf Pass Rate  : 100%
  Low Risk        : 70%
  High Risk       : 10%  (only on expected RBAC-blocked queries)

SECURITY METRICS
  Chunks Denied        : 20
  Unauthorized Leaks   : 0   ✅
  RBAC Effectiveness   : 100%

LATENCY (in-memory)
  P50 : 1.7ms   P95 : 2.4ms   P99 : 2.4ms
```

Run it yourself:
```bash
python evaluation/evaluate.py
```

---

## Test Suite

```bash
python tests/test_rag.py
```

```
✅  test_rbac_clearance_block
✅  test_rbac_role_whitelist
✅  test_rbac_admin_bypass
✅  test_rbac_clearance_allow
✅  test_embedder_fit
✅  test_embedder_l2_norm
✅  test_embedder_zero_for_unknown
✅  test_retrieval_returns_results
✅  test_retrieval_relevance_order
✅  test_dept_routing_hr
✅  test_dept_routing_finance
✅  test_dept_routing_none
✅  test_query_hr_user_gets_results
✅  test_query_guest_blocked
✅  test_query_admin_full_access
✅  test_audit_log_populated
✅  test_hallucination_risk_high_when_no_results
─────────────────────────────────────────────
17 passed  |  0 failed  |  17 total
```

---

## Dataset

10 synthetic enterprise documents across 4 departments:

| Document | Format | Dept | Classification |
|----------|--------|------|----------------|
| Employee Compensation Policy FY2025 | PDF | HR | Confidential |
| Headcount & Attrition Report Q1 2025 | CSV | HR | Confidential |
| Employee Onboarding Guide | Text | HR | Internal |
| Q1 2025 Financial Results | PDF | Finance | Confidential |
| Vendor Invoice Registry Q1 2025 | CSV | Finance | Internal |
| Platform Architecture Guide v3.2 | PDF | Engineering | Internal |
| Incident Post-Mortem SEV-1 2025-03-14 | JSON | Engineering | Internal |
| System Health Dashboard W11 2025 | JSON | Engineering | Internal |
| Data Retention & Privacy Policy v2.1 | PDF | Legal | Internal |
| SOC2 Type II Audit Report FY2024 | PDF | Legal | **Restricted** |

---

## Tech Stack

| Layer | This Project | Production Upgrade |
|-------|-------------|-------------------|
| Embeddings | TF-IDF (NumPy) | `sentence-transformers` / OpenAI ada-002 |
| Vector Store | In-memory matrix | FAISS / Pinecone / Weaviate |
| LLM | Claude Sonnet API (optional) | Claude + GPT-4o fallback |
| API | FastAPI | FastAPI + Nginx + Gunicorn |
| Deployment | Docker Compose | Kubernetes (EKS/GKE) |
| Auth | Header-based user ID | JWT + OAuth2 / Okta |
| Observability | JSON audit log | DataDog APM + ELK stack |

---

## Resume Bullet Points

> **Enterprise RAG Intelligence System** | [GitHub] | Python · NumPy · FastAPI · Docker · Claude API
> - Built production-grade RAG pipeline with hybrid TF-IDF + cosine retrieval and MMR diversity reranking across 10 multi-format enterprise docs (PDF, CSV, JSON)
> - Engineered two-layer RBAC (clearance hierarchy + role whitelist) — blocked 20 unauthorized chunk accesses across all test scenarios with zero data exposure
> - Shipped full REST API (FastAPI), Docker deployment, 17-test suite, and evaluation framework reporting MRR=0.833, NDCG@5=0.799, P95 latency=2.4ms

---

## Author

**Vimal Kumar** — Machine Learning Engineer  
vk2699945@gmail.com | [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)
