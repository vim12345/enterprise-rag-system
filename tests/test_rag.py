"""
tests/test_rag.py  —  Unit & Integration Tests
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.rag_engine import (
    EnterpriseRAGPipeline, Document, User, RBACEngine,
    VocabEmbedder, HybridRetriever, DocumentStore,
)
from dataset import DOCUMENTS, USERS


def build_pipeline():
    p = EnterpriseRAGPipeline()
    for doc in DOCUMENTS:
        p.add_document(doc)
    p.build_index()
    for u in USERS:
        p.register_user(u)
    return p


# ── RBAC tests ─────────────────────────────────────────────────────────────────

def test_rbac_clearance_block():
    """Guest cannot access confidential documents."""
    rbac = RBACEngine()
    guest = User("G1","Guest","guest","ext","public")
    doc   = Document("D1","Salary","content","pdf","hr","confidential",[].__class__(),[])
    assert not rbac.can_access(guest, doc), "Guest should be blocked from confidential"

def test_rbac_role_whitelist():
    """Engineer cannot access legal-only doc even at same clearance level."""
    rbac = RBACEngine()
    eng  = User("E1","Eng","engineering","eng","internal")
    doc  = Document("D2","Arch","content","pdf","engineering","internal",["legal","admin"])
    assert not rbac.can_access(eng, doc), "Engineer not in whitelist should be blocked"

def test_rbac_admin_bypass():
    """Admin can always access any document."""
    rbac  = RBACEngine()
    admin = User("A1","Admin","admin","exec","restricted")
    doc   = Document("D3","Secret","content","pdf","legal","restricted",["legal"])
    assert rbac.can_access(admin, doc), "Admin should bypass whitelist"

def test_rbac_clearance_allow():
    """Confidential-clearance user can access confidential doc."""
    rbac    = RBACEngine()
    finance = User("F1","Finance","finance","fin","confidential")
    doc     = Document("D4","Rev","content","pdf","finance","confidential",[])
    assert rbac.can_access(finance, doc)


# ── Embedding tests ────────────────────────────────────────────────────────────

def test_embedder_fit():
    emb = VocabEmbedder()
    docs = [
        Document("d1","T1","the quick brown fox jumps","text","hr","public",[]),
        Document("d2","T2","the lazy dog sleeps quietly","text","hr","public",[]),
    ]
    emb.fit(docs)
    assert len(emb.vocab) > 0
    assert len(emb.idf) == len(emb.vocab)

def test_embedder_l2_norm():
    emb = VocabEmbedder()
    docs = [Document("d1","T1","salary bonus compensation employee","text","hr","public",[])]
    emb.fit(docs)
    vec = emb.transform("salary bonus")
    import numpy as np
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5 or norm == 0.0, f"Vector should be L2-normalized, got {norm}"

def test_embedder_zero_for_unknown():
    emb = VocabEmbedder()
    docs = [Document("d1","T","hello world","text","hr","public",[])]
    emb.fit(docs)
    vec = emb.transform("zzz unknown xyzabc")
    import numpy as np
    assert float(np.linalg.norm(vec)) == 0.0


# ── Retrieval tests ────────────────────────────────────────────────────────────

def test_retrieval_returns_results():
    p = build_pipeline()
    chunks = p.retriever.retrieve("salary bands bonus", top_k=5)
    assert len(chunks) > 0, "Should retrieve results for salary query"

def test_retrieval_relevance_order():
    p = build_pipeline()
    chunks = p.retriever.retrieve("salary compensation bonus", top_k=5)
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by score desc"

def test_dept_routing_hr():
    p = build_pipeline()
    dept = p.retriever.route("salary bonus compensation employee leave")
    assert dept == "hr", f"Expected hr routing, got {dept}"

def test_dept_routing_finance():
    p = build_pipeline()
    dept = p.retriever.route("revenue budget expense financial quarter")
    assert dept == "finance", f"Expected finance routing, got {dept}"

def test_dept_routing_none():
    p = build_pipeline()
    dept = p.retriever.route("hello world general question")
    assert dept is None, "Ambiguous query should not be routed"


# ── End-to-end tests ───────────────────────────────────────────────────────────

def test_query_hr_user_gets_results():
    p = build_pipeline()
    resp = p.query("U002", "What are the salary bands?")
    assert resp.retrieved_count > 0
    assert resp.confidence > 0
    assert len(resp.sources) > 0

def test_query_guest_blocked():
    p = build_pipeline()
    resp = p.query("U007", "What are the salary ranges and compensation?")
    assert resp.retrieved_count == 0, "Guest should get 0 accessible results"
    assert resp.denied_count > 0, "Guest should have chunks denied"

def test_query_admin_full_access():
    p = build_pipeline()
    resp = p.query("U001", "What did the SOC2 audit find?")
    assert resp.retrieved_count > 0, "Admin should access restricted SOC2 doc"

def test_audit_log_populated():
    p = build_pipeline()
    p.query("U002", "salary bands")
    p.query("U004", "system architecture")
    assert len(p.audit_log) == 2
    assert all("confidence" in e for e in p.audit_log)
    assert all("denied" in e for e in p.audit_log)

def test_hallucination_risk_high_when_no_results():
    p = build_pipeline()
    resp = p.query("U007", "confidential salary data")
    assert resp.hallucination_risk == "HIGH"


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_rbac_clearance_block, test_rbac_role_whitelist,
        test_rbac_admin_bypass, test_rbac_clearance_allow,
        test_embedder_fit, test_embedder_l2_norm, test_embedder_zero_for_unknown,
        test_retrieval_returns_results, test_retrieval_relevance_order,
        test_dept_routing_hr, test_dept_routing_finance, test_dept_routing_none,
        test_query_hr_user_gets_results, test_query_guest_blocked,
        test_query_admin_full_access, test_audit_log_populated,
        test_hallucination_risk_high_when_no_results,
    ]
    passed = failed = 0
    print("\n" + "═"*55)
    print("  TEST SUITE — Enterprise RAG System")
    print("═"*55)
    for t in tests:
        try:
            t()
            print(f"  ✅  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌  {t.__name__}  →  {e}")
            failed += 1
    print("─"*55)
    print(f"  {passed} passed  |  {failed} failed  |  {len(tests)} total")
    print("═"*55 + "\n")
    if failed:
        sys.exit(1)
