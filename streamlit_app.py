"""
streamlit_app.py  —  Interactive RAG Demo UI
=============================================
Run: streamlit run streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import time
from src.rag_engine import EnterpriseRAGPipeline
from dataset import DOCUMENTS, USERS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG Intelligence System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e; border-radius: 10px; padding: 16px;
    border: 1px solid #313244; margin: 4px 0;
}
.source-card {
    background: #181825; border-radius: 8px; padding: 12px;
    border-left: 3px solid #89b4fa; margin: 6px 0; font-size: 0.9em;
}
.badge-low    { background:#a6e3a1; color:#1e1e2e; padding:2px 8px; border-radius:12px; font-size:0.8em; }
.badge-medium { background:#f9e2af; color:#1e1e2e; padding:2px 8px; border-radius:12px; font-size:0.8em; }
.badge-high   { background:#f38ba8; color:#1e1e2e; padding:2px 8px; border-radius:12px; font-size:0.8em; }
.deny-badge   { background:#f38ba8; color:#1e1e2e; padding:2px 8px; border-radius:12px; font-size:0.8em; }
</style>
""", unsafe_allow_html=True)

# ── Pipeline (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Building RAG index…")
def load_pipeline():
    p = EnterpriseRAGPipeline()
    for doc in DOCUMENTS:
        p.add_document(doc)
    stats = p.build_index()
    for u in USERS:
        p.register_user(u)
    return p, stats

pipeline, idx_stats = load_pipeline()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏢 Enterprise RAG")
    st.caption("v2.0 · Hybrid Retrieval · RBAC · MMR")

    st.divider()
    st.subheader("👤 Select User")

    user_options = {
        f"{u.name}  [{u.role}]": u.user_id for u in USERS
    }
    selected_label = st.selectbox("Login as:", list(user_options.keys()))
    selected_uid   = user_options[selected_label]
    selected_user  = pipeline.users[selected_uid]

    st.markdown(f"""
    <div class="metric-card">
    <b>Role</b>: {selected_user.role}<br>
    <b>Dept</b>: {selected_user.department}<br>
    <b>Clearance</b>: <code>{selected_user.clearance_level}</code>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Results (top_k)", 1, 8, 5)

    st.divider()
    st.subheader("📊 Index Stats")
    st.metric("Documents", idx_stats["documents"])
    st.metric("Chunks",    idx_stats["chunks"])
    st.metric("Vocab",     idx_stats["vocab_size"])
    st.metric("Users",     len(pipeline.users))

    st.divider()
    st.subheader("🗄️ Documents")
    for doc in DOCUMENTS:
        color = {"public":"🟢","internal":"🔵","confidential":"🟡","restricted":"🔴"}.get(doc.classification,"⚪")
        st.caption(f"{color} [{doc.source_type.upper()}] {doc.title[:38]}")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🏢 Enterprise RAG Intelligence System")
st.caption("Hybrid Retrieval · MMR Diversity · Two-Layer RBAC · Hallucination Guard")

# Sample queries
SAMPLE_QUERIES = {
    "💰 Salary bands & bonuses (HR)":
        "What are the salary bands, bonus percentages, and equity grants for each level?",
    "📈 Q1 revenue & SaaS metrics (Finance)":
        "What was total revenue, EBITDA margin, ARR, and NRR in Q1 2025?",
    "🏗️ Platform architecture & ML stack (Engineering)":
        "What databases and ML models does the platform use, and what are the SLOs?",
    "⚖️ SOC2 audit findings (Legal)":
        "What material weaknesses were found in the SOC2 audit and were they remediated?",
    "🔴 Redis outage root cause (Engineering)":
        "What caused the production outage, how long did it last, and what was the revenue impact?",
    "🚫 RBAC test — guest tries salary data":
        "What is the employee compensation policy and salary ranges?",
    "📉 Attrition analysis (HR)":
        "Which department has the highest attrition rate and what is the company benchmark?",
    "💳 Pending vendor invoices (Finance)":
        "Which vendor invoices are pending approval and what is the total outstanding amount?",
}

st.subheader("❓ Ask a Question")

col1, col2 = st.columns([3, 1])
with col2:
    sample = st.selectbox("Or pick a sample:", ["— type your own —"] + list(SAMPLE_QUERIES.keys()))

with col1:
    default_q = SAMPLE_QUERIES.get(sample, "") if sample != "— type your own —" else ""
    question = st.text_input("Your question:", value=default_q, placeholder="Ask anything about the enterprise data…")

run_btn = st.button("🔍 Query", type="primary", use_container_width=False)

# ── Query execution ───────────────────────────────────────────────────────────
if run_btn and question.strip():
    with st.spinner("Retrieving and generating answer…"):
        t0   = time.perf_counter()
        resp = pipeline.query(selected_uid, question, top_k=top_k)
        wall = round((time.perf_counter() - t0) * 1000, 1)

    st.divider()

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Confidence",  f"{resp.confidence:.0%}")
    m2.metric("Sources Used", resp.retrieved_count)
    m3.metric("RBAC Denied",  resp.denied_count)
    m4.metric("Latency",      f"{resp.latency_ms}ms")
    m5.metric("Routed →",     resp.routed_to or "all depts")

    # Risk badge
    risk_html = {
        "LOW":    '<span class="badge-low">🟢 LOW RISK</span>',
        "MEDIUM": '<span class="badge-medium">🟡 MEDIUM RISK</span>',
        "HIGH":   '<span class="badge-high">🔴 HIGH RISK</span>',
    }.get(resp.hallucination_risk, "")
    st.markdown(f"Hallucination Risk: {risk_html}", unsafe_allow_html=True)

    # RBAC denial warning
    if resp.denied_count > 0:
        st.warning(f"🔐 **{resp.denied_count}** document chunk(s) were hidden — your role "
                   f"(`{selected_user.role}` / `{selected_user.clearance_level}`) "
                   f"does not have clearance to access them.")

    st.divider()
    st.subheader("💬 Answer")
    st.markdown(resp.answer)

    # Sources
    if resp.sources:
        st.divider()
        st.subheader("📚 Source Attribution")
        for s in resp.sources:
            cls_color = {"public":"🟢","internal":"🔵","confidential":"🟡","restricted":"🔴"}.get(s["classification"],"⚪")
            st.markdown(f"""
            <div class="source-card">
            <b>[Source {s['source_id']}]</b> {s['title']}<br>
            {cls_color} <code>{s['classification']}</code> &nbsp;·&nbsp;
            <code>{s['type'].upper()}</code> &nbsp;·&nbsp;
            dept: <b>{s['department']}</b> &nbsp;·&nbsp;
            relevance: <b>{s['relevance_score']:.3f}</b>
            </div>
            """, unsafe_allow_html=True)
    elif resp.retrieved_count == 0:
        st.error("🚫 No documents accessible for your role and clearance level.")

# ── Audit log tab ─────────────────────────────────────────────────────────────
st.divider()
if pipeline.audit_log:
    with st.expander(f"📋 Session Audit Log  ({len(pipeline.audit_log)} queries)", expanded=False):
        stats = pipeline.stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries",   stats["total_queries"])
        c2.metric("Avg Confidence",  f"{stats['avg_confidence']:.0%}")
        c3.metric("Total Denied",    stats["total_denied"])
        c4.metric("Low Risk %",      f"{stats['low_risk_pct']}%")

        st.dataframe(
            [{
                "User":       e["user"],
                "Role":       e["role"],
                "Query":      e["query"][:50] + "…",
                "Dept":       e["dept_hint"] or "all",
                "Retrieved":  e["retrieved"],
                "Denied":     e["denied"],
                "Confidence": f"{e['confidence']:.0%}",
                "Risk":       e["risk"],
                "Latency":    f"{e['latency_ms']}ms",
            } for e in pipeline.audit_log],
            use_container_width=True,
        )
