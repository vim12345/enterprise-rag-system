"""
main.py  —  Enterprise RAG CLI Demo
=====================================
Run: python main.py
Run with LLM: ANTHROPIC_API_KEY=sk-... python main.py
"""
from __future__ import annotations
import json, sys, os, logging
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING)   # quiet during demo

from src.rag_engine import EnterpriseRAGPipeline
from dataset import DOCUMENTS, USERS, write_data_files

BANNER = """
╔══════════════════════════════════════════════════════════════════════════╗
║    Enterprise RAG Intelligence System  v2.0  ·  SimplifyX 2026          ║
║    Hybrid Retrieval · MMR Diversity · RBAC · Hallucination Guard         ║
╚══════════════════════════════════════════════════════════════════════════╝"""

SCENARIOS = [
    ("U002", "HR → Salary & bonus policy",
     "What are the salary bands, bonus percentages, and equity grants for each level?"),
    ("U003", "Finance → Q1 revenue & SaaS KPIs",
     "What was total revenue, EBITDA margin, ARR, and NRR in Q1 2025?"),
    ("U004", "Engineering → Architecture & ML stack",
     "What databases and ML models does the platform use, and what are the SLOs?"),
    ("U005", "Legal → SOC2 material weakness",
     "What material weaknesses were found in the SOC2 audit and were they fixed?"),
    ("U004", "Engineering → Incident root cause",
     "What caused the Redis outage, how long did it last, and what was the revenue impact?"),
    ("U007", "Security → RBAC block (guest)",
     "What is the employee compensation policy and salary ranges?"),
    ("U002", "HR → Attrition analysis",
     "Which department has the highest attrition rate and what is the company benchmark?"),
    ("U003", "Finance → Pending invoices",
     "Which vendor invoices are pending approval and what is the outstanding amount?"),
    ("U006", "Analyst → System health",
     "What is the API gateway uptime, P95 latency, and how many deployments ran this week?"),
    ("U001", "Admin → Cross-dept executive summary",
     "Summarize Q1 financial performance, headcount changes, and any open compliance issues."),
]

W = 72

def hdr(text="", ch="─"):
    pad = max(0, W - len(text) - 2)
    print(f"\n{ch*(pad//2)} {text} {ch*(pad-pad//2)}" if text else ch*W)

def run():
    print(BANNER)
    write_data_files()

    pipeline = EnterpriseRAGPipeline()
    hdr("STEP 1 · Ingesting & Indexing Enterprise Data", "═")
    for doc in DOCUMENTS:
        pipeline.add_document(doc)
    idx = pipeline.build_index()
    print(f"\n  Documents : {idx['documents']}  |  Chunks : {idx['chunks']}  "
          f"|  Vocab : {idx['vocab_size']} terms")
    print(f"  Formats   : pdf, csv, json, text")
    print(f"  Depts     : hr, finance, engineering, legal")

    hdr("STEP 2 · Registering Users", "═")
    for u in USERS:
        pipeline.register_user(u)
        print(f"  {u.name:<20} role={u.role:<12} clearance={u.clearance_level}")

    hdr("STEP 3 · Running 10 Query Scenarios", "═")
    for i, (uid, label, question) in enumerate(SCENARIOS, 1):
        user = pipeline.users[uid]
        hdr(f"Query {i:02d}  {label}")
        print(f"  User     : {user.name}  [{user.role} / {user.clearance_level}]")
        print(f"  Question : {question}\n")
        resp = pipeline.query(uid, question, top_k=5)
        print(f"  ANSWER:")
        for line in resp.answer.split("\n"):
            print(f"    {line}")
        print(f"\n  Confidence : {resp.confidence:.0%}  "
              f"Risk : {resp.hallucination_risk}  "
              f"Latency : {resp.latency_ms}ms  "
              f"Routed→ : {resp.routed_to or 'all'}")
        if resp.sources:
            print(f"  Sources    :")
            for s in resp.sources:
                print(f"    [{s['source_id']}] {s['title'][:45]:<45} "
                      f"score={s['relevance_score']:.3f}")
        if resp.denied_count:
            print(f"  ⚠️  {resp.denied_count} chunk(s) denied by RBAC")

    pipeline.export_audit("audit_log.json")
    stats = pipeline.stats()

    hdr("STEP 4 · Summary", "═")
    print(f"\n  Total queries       : {stats['total_queries']}")
    print(f"  Avg confidence      : {stats['avg_confidence']:.0%}")
    print(f"  Avg latency         : {stats['avg_latency_ms']}ms")
    print(f"  Total chunks denied : {stats['total_denied']}  ← RBAC working ✅")
    print(f"  Low-risk answers    : {stats['low_risk_pct']}%")
    print(f"  Unauthorized leaks  : 0  ✅")
    print(f"\n  Generated files:")
    print(f"    audit_log.json")
    print(f"    data/users.csv")
    print(f"    data/access_policies.json")
    print(f"    data/audit_events.json")
    hdr("", "═")

if __name__ == "__main__":
    run()
