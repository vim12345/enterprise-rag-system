"""
Synthetic Enterprise Dataset
=============================
Generates realistic multi-format enterprise data:
PDF-style text · CSV tables · JSON logs · Technical reports · Compliance records
"""
from __future__ import annotations
import json, csv, os
from src.rag_engine import Document, User

# ═══════════════════════════════════════════════════════════════════════════════
#  DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

DOCUMENTS: list[Document] = [

    # ─── HR ───────────────────────────────────────────────────────────────────
    Document(
        doc_id="HR-POL-001", title="Employee Compensation & Benefits Policy FY2025",
        source_type="pdf", department="hr", classification="confidential",
        allowed_roles=["hr","admin","legal"],
        content="""
EMPLOYEE COMPENSATION & BENEFITS POLICY — FY 2025
Issued by: Human Resources | Effective: January 1, 2025 | Version: 4.2

━━ 1. SALARY BANDS ━━
Role Level  | Band         | Midpoint  | Equity (Options)
Level 1 IC  | $55–$75K     | $65K      | 2,500
Level 2 IC  | $76–$105K    | $90K      | 5,000
Level 3 IC  | $106–$145K   | $125K     | 10,000
Level 4 IC  | $146–$195K   | $170K     | 20,000
Level 5 IC  | $196–$260K   | $225K     | 40,000
Manager L1  | $120–$160K   | $140K     | 15,000
Director    | $180–$240K   | $210K     | 50,000
VP+         | $250K+       | $300K     | 100,000+

━━ 2. BONUS STRUCTURE ━━
Target bonus as % of base salary:
• Levels 1–2: 8–10%   • Levels 3–4: 12–15%   • Levels 5+: 18–25%
Outstanding performers (top 15%) receive 1.5× multiplier.
Bonus pools allocated quarterly; payout in February following fiscal year.
Employees must have 6 months tenure by Dec 31 to be eligible.

━━ 3. EQUITY ━━
4-year vesting with 1-year cliff. 10-year option exercise window post-grant.
Annual refresh grants for Level 3+ reviewed each April.
Equity acceleration: 12 months cliff on acquisition; double-trigger for VP+.

━━ 4. LEAVE POLICY ━━
• Annual Leave: 20 days (L1–L3); 25 days (L4+); unlimited for VP+
• Sick Leave: 12 days, fully paid, non-carry-over
• Parental Leave: 16 weeks primary caregiver; 8 weeks secondary
• Bereavement: 5 days immediate family; 3 days extended family
• Unpaid Leave: VP approval required; max 90 days/year

━━ 5. BENEFITS ━━
Health: Medical (Blue Cross PPO + HDHP options), dental, vision — 100% employee,
        80% dependent premiums covered.
401(k): 4% dollar-for-dollar match; immediate vesting.
Perks: $800/year remote stipend; $2,000/year L&D; $500/year wellness.
Life insurance: 2× annual salary. Short/long-term disability included.

CLASSIFICATION: CONFIDENTIAL — Unauthorized distribution is a policy violation.
"""),

    Document(
        doc_id="HR-HC-002", title="Headcount & Attrition Report Q1 2025",
        source_type="csv", department="hr", classification="confidential",
        allowed_roles=["hr","admin","finance"],
        content="""
HEADCOUNT REPORT — Q1 2025 (as of March 31, 2025)

Department      | HC_Start | HC_End | Hired | Departed | OpenReqs | Attrition%
----------------|----------|--------|-------|----------|----------|----------
Engineering     | 138      | 142    | 9     | 5        | 18       | 3.6%
Product         | 33       | 34     | 2     | 1        | 4        | 3.0%
Finance         | 27       | 28     | 2     | 1        | 2        | 3.7%
HR              | 15       | 15     | 1     | 1        | 1        | 6.7%
Legal           | 12       | 12     | 0     | 0        | 0        | 0.0%
Sales           | 62       | 67     | 11    | 6        | 9        | 9.7% ⚠️
Marketing       | 30       | 31     | 3     | 2        | 3        | 6.7%
Operations      | 42       | 44     | 4     | 2        | 5        | 4.8%
─────────────────────────────────────────────────────────────────────────────
TOTALS          | 359      | 373    | 32    | 18       | 42       | 5.0%

KEY INSIGHTS:
• Sales attrition (9.7%) exceeds company benchmark of 7.5% — root-cause review underway
• Engineering grew +4 net; 18 open reqs reflects aggressive AI/ML hiring push
• Overall Q1 attrition 5.0% vs 6.2% same quarter last year — improving trend
• 42 open requisitions all budget-approved; sourcing in progress

SOURCE: Workday HRIS extract — March 31, 2025
"""),

    Document(
        doc_id="HR-KB-003", title="Employee Onboarding & Tools Guide",
        source_type="text", department="hr", classification="internal",
        allowed_roles=[],
        content="""
EMPLOYEE ONBOARDING GUIDE — SimplifyX Inc.

Welcome! This guide covers your first 30 days.

WEEK 1 CHECKLIST
□ IT setup: laptop provisioning, SSO, GitHub, Jira, Slack
□ Sign: Code of Conduct, IP Assignment, Data Privacy Acknowledgement
□ Complete: Security Awareness Training (mandatory, 2 hrs)
□ Complete: Anti-Harassment & Diversity Training (mandatory, 1 hr)
□ Meet your onboarding buddy and schedule weekly 1:1s for 90 days
□ Enroll in benefits within 30 days of start date (hard deadline)

ESSENTIAL TOOLS
Tool              | URL / Handle              | Purpose
Slack             | simplifyx.slack.com       | Communication
Jira              | jira.simplifyx.internal   | Project tracking
GitHub Enterprise | github.simplifyx.internal | Code & docs
Notion            | simplifyx.notion.site     | Knowledge base
Workday           | hr.simplifyx.internal     | HR, payroll, time-off
Zoom              | simplifyx.zoom.us         | Video calls

KEY CONTACTS
• IT Help Desk : it-help@simplifyx.com | Slack: #it-support (response < 2 hrs)
• HR Questions : hr@simplifyx.com | Slack: #hr-team
• Security     : security@simplifyx.com | Slack: #security-alerts
• Payroll      : payroll@simplifyx.com (payroll queries only)

CORE HOURS: 10 AM – 3 PM local time for meetings. Async outside those hours.
REMOTE WORK: Up to 3 days/week for most roles. Check with your manager.
PERFORMANCE REVIEWS: Bi-annual — June (mid-year) and December (annual).
PROBATION: 90-day review with your manager and HR.

Questions? hr@simplifyx.com or drop by #hr-team on Slack.
"""),

    # ─── Finance ──────────────────────────────────────────────────────────────
    Document(
        doc_id="FIN-QR-001", title="Q1 2025 Financial Results & Analysis",
        source_type="pdf", department="finance", classification="confidential",
        allowed_roles=["finance","admin","legal"],
        content="""
Q1 2025 FINANCIAL RESULTS — CONFIDENTIAL
Prepared by: Finance | Date: April 12, 2025 | Distribution: Restricted

━━ INCOME STATEMENT SUMMARY ━━
                        Q1 2025     Q1 2024     YoY Change
Product Revenue       : $18.4M      $15.1M      +21.9%
Services Revenue      : $4.1M       $3.8M       +7.9%
────────────────────────────────────────────────────────
TOTAL REVENUE         : $22.5M      $18.9M      +19.0%

Cost of Revenue       : $6.8M       $5.7M       +19.3%
Gross Profit          : $15.7M      $13.2M      +18.9%
Gross Margin          : 69.8%       69.8%       flat

R&D                   : $5.2M       $4.1M       +26.8%
Sales & Marketing     : $4.7M       $4.0M       +17.5%
General & Admin       : $2.1M       $1.9M       +10.5%
────────────────────────────────────────────────────────
Total OpEx            : $18.8M      $15.7M      +19.7%
Operating Income      : $3.7M       $3.2M       +15.6%
Operating Margin      : 16.4%       16.9%       -50bps
Net Income            : $2.9M       $2.5M       +16.0%
EBITDA                : $4.4M       $3.8M       +15.8%
EBITDA Margin         : 19.6%       20.1%       -50bps

━━ BALANCE SHEET HIGHLIGHTS ━━
Cash & Equivalents    : $41.2M   (up from $38.7M at Dec 31)
Accounts Receivable   : $8.3M    (DSO: 33 days, target <40)
Deferred Revenue      : $12.1M
Total Debt            : $0       (debt-free)
Runway                : >18 months at current net burn

━━ KEY SaaS METRICS ━━
Annual Recurring Revenue (ARR)    : $87.6M   (+27% YoY)
Net Revenue Retention (NRR)       : 118%     (best quarter ever)
Customer Acquisition Cost (CAC)   : $3,400
Lifetime Value (LTV)              : $52,000
LTV:CAC                           : 15.3×
Churn Rate (gross)                : 0.8% monthly
Paying Customers                  : 1,847    (+340 net-new Q1)

━━ Q2 2025 GUIDANCE ━━
Revenue: $23.5M – $24.5M (+18–22% YoY)
EBITDA Margin: 18–20%
Headcount additions: ~30 (primarily Engineering and Sales)

MATERIAL NON-PUBLIC INFORMATION — Do not distribute outside approved list.
"""),

    Document(
        doc_id="FIN-INV-002", title="Vendor Invoice & AP Registry — Q1 2025",
        source_type="csv", department="finance", classification="internal",
        allowed_roles=["finance","admin"],
        content="""
ACCOUNTS PAYABLE — Q1 2025 VENDOR REGISTRY

Invoice ID    | Vendor                  | Amount    | Category            | Status     | Due Date   | PO Number
INV-2025-0301 | Amazon Web Services     | $48,320   | Infrastructure      | PAID       | 2025-03-15 | PO-2025-114
INV-2025-0302 | Salesforce Inc          | $12,400   | CRM Software        | PAID       | 2025-03-20 | PO-2025-089
INV-2025-0303 | Acme Strategy Consulting| $28,000   | Professional Svcs   | PENDING ⚠  | 2025-04-01 | PO-2025-203
INV-2025-0304 | Office Depot            | $1,240    | Facilities          | PAID       | 2025-03-10 | PO-2025-071
INV-2025-0305 | DataDog Inc             | $3,800    | Monitoring          | PAID       | 2025-03-18 | PO-2025-096
INV-2025-0306 | Legal Partners LLP      | $15,000   | Legal Fees          | PENDING ⚠  | 2025-04-05 | PO-2025-187
INV-2025-0307 | TalentSearch Agency     | $22,500   | Recruitment         | PROCESSING | 2025-04-10 | PO-2025-211
INV-2025-0308 | HubSpot                 | $6,750    | Marketing Platform  | PAID       | 2025-03-25 | PO-2025-102
INV-2025-0309 | Pinecone AI             | $4,200    | Vector DB           | PAID       | 2025-03-28 | PO-2025-198
INV-2025-0310 | Cohere Technologies     | $2,100    | AI APIs             | PAID       | 2025-03-30 | PO-2025-199
─────────────────────────────────────────────────────────────────────────────────────────────────
QUARTER TOTAL                          | $144,310

PENDING APPROVALS: INV-2025-0303, INV-2025-0306 require VP Finance sign-off
PROCESSING: INV-2025-0307 — waiting on recruiter deliverable confirmation
AP Contact: ap@simplifyx.com | Cut-off for weekly run: Fridays 3 PM PT
"""),

    # ─── Engineering ──────────────────────────────────────────────────────────
    Document(
        doc_id="ENG-ARCH-001", title="Platform Architecture & Infrastructure Guide v3.2",
        source_type="pdf", department="engineering", classification="internal",
        allowed_roles=["engineering","admin"],
        content="""
PLATFORM ARCHITECTURE GUIDE — v3.2
Team: Platform Engineering | Last Updated: March 2025

━━ HIGH-LEVEL OVERVIEW ━━
SimplifyX operates a cloud-native, microservices architecture on AWS (primary)
with GCP as the DR region. All services are containerized (Docker) and
orchestrated via Kubernetes (EKS v1.29).

━━ CORE SERVICES ━━
Service               | Language  | Framework     | Instances | P95 Latency
api-gateway           | Go 1.22   | Kong          | 6         | 12ms
user-service          | Python    | FastAPI       | 4         | 45ms
query-service         | Python    | FastAPI       | 8         | 1,840ms
data-ingestion-svc    | Python    | Celery+Kafka  | 6         | async
notification-svc      | Node.js   | Express       | 3         | 28ms
analytics-svc         | Python    | FastAPI       | 2         | 210ms

━━ DATA LAYER ━━
• PostgreSQL 15       : Primary relational store (RDS Multi-AZ, r6g.2xlarge)
• Redis 7 Cluster     : Session cache + rate limiting (6-node, 12GB → expanded to 20GB post-incident)
• Pinecone            : Vector DB for RAG embeddings (1536-dim, ada-002)
• S3 (versioned)      : Document blob storage (4.2 TB used)
• Snowflake           : Analytics data warehouse (synced hourly via dbt)
• Elasticsearch 8     : Full-text search, log indexing

━━ ML / AI STACK ━━
• Embedding model  : text-embedding-ada-002 (OpenAI)
• Primary LLM      : GPT-4o (generation, routing)
• Fallback LLM     : Claude 3.5 Sonnet (Anthropic)
• Reranker         : Cohere rerank-v3
• Vector store     : Pinecone (index: simplifyx-prod, 50M vectors)
• Guardrails       : Custom prompt injection filters + Lakera Guard

━━ SECURITY ━━
• mTLS between all internal services (Istio service mesh)
• Secrets: AWS Secrets Manager (zero hard-coded credentials policy)
• Auth: JWT (15-min TTL) + OAuth2 with PKCE (external); mTLS (internal)
• Network: VPC with private subnets; no public service IPs except ALB
• WAF: AWS WAF with OWASP Top 10 rule set
• SOC2 Type II certified; PEN test completed January 2025 (no critical findings)

━━ OBSERVABILITY ━━
• Metrics  : DataDog APM + infrastructure metrics
• Logs     : Structured JSON → ELK stack (7-day hot, 90-day warm)
• Traces   : OpenTelemetry → DataDog
• Alerts   : PagerDuty (P1/P2); Slack #alerts (P3/P4)

SLOs: 99.9% availability; P95 API latency < 500ms; P99 < 2s
Current SLO attainment (Q1 2025): 99.74% (impacted by Mar 14 incident)
"""),

    Document(
        doc_id="ENG-INC-002", title="Incident Post-Mortem: SEV-1 Outage 2025-03-14",
        source_type="json", department="engineering", classification="internal",
        allowed_roles=["engineering","admin"],
        content="""
INCIDENT POST-MORTEM — SEV-1 PRODUCTION OUTAGE
Incident ID : INC-2025-031
Date        : March 14, 2025
Duration    : 2 hours 18 minutes (09:42 – 12:00 UTC)
Severity    : SEV-1 (100% customer-facing impact)
IC          : Sarah K. (Platform Lead)
Status      : CLOSED

━━ TIMELINE ━━
09:42 UTC — DataDog alert: API error rate exceeds 50% threshold
09:44 UTC — PagerDuty page to on-call engineer
09:47 UTC — Incident bridge opened; IC assigned
10:05 UTC — Root cause identified: Redis cluster out of memory (OOM)
10:15 UTC — Attempted cache flush; blocked by key lock contention
10:35 UTC — Decision: scale Redis + restart with warm-up
11:20 UTC — Redis restart initiated; service degraded but partial traffic served
11:52 UTC — API error rate < 1%; services recovering
12:00 UTC — All-clear declared; customer notification sent

━━ ROOT CAUSE ━━
Redis maxmemory-policy was set to 'noeviction'. A nightly batch job (user
session pre-computation) pushed 14.2 GB of data in 3 minutes, exhausting
the 12 GB cluster. With noeviction, Redis began rejecting all writes,
causing authentication failures which cascaded to complete API unavailability.

━━ IMPACT ━━
• 100% of authenticated API calls failed for 2h 18min
• ~8,400 active users impacted
• 4 enterprise customers triggered SLA breach clauses
• Estimated direct revenue impact: $34,000
• No data corruption or loss confirmed

━━ REMEDIATION (all completed) ━━
[DONE] Changed Redis maxmemory-policy to 'allkeys-lru'
[DONE] Expanded Redis cluster from 12 GB → 20 GB
[DONE] Added DataDog alert: Redis memory > 70% triggers P2 page
[DONE] Batch job migrated to dedicated Redis namespace (redis-batch)
[DONE] Runbook updated: Redis OOM recovery procedure documented
[IN PROGRESS] Chaos engineering test: simulate Redis OOM in staging monthly

━━ LESSONS LEARNED ━━
1. Never use noeviction in production for non-critical cache data
2. Batch jobs must use isolated infrastructure to prevent blast radius
3. Memory alerts should be at 70%, not 90% (previous threshold)

Jira Epic: PLAT-4821 | Post-mortem Owner: Platform Team
"""),

    Document(
        doc_id="ENG-SYS-003", title="System Health & Performance Dashboard — W11 2025",
        source_type="json", department="engineering", classification="internal",
        allowed_roles=["engineering","admin","analyst"],
        content="""
SYSTEM HEALTH REPORT — Week 11 (March 10–16, 2025)
Generated: 2025-03-17 00:05 UTC | Source: DataDog + PagerDuty

━━ SERVICE AVAILABILITY (7-day) ━━
Service                | Uptime    | SLO     | Status
api-gateway            | 99.74%    | 99.9%   | ⚠️ MISS (SEV-1 outage Mar 14)
user-service           | 99.91%    | 99.9%   | ✅ MET
query-service          | 99.88%    | 99.9%   | ✅ MET (within margin)
data-ingestion-svc     | 99.95%    | 99.5%   | ✅ MET
notification-svc       | 99.99%    | 99.9%   | ✅ MET
analytics-svc          | 99.93%    | 99.5%   | ✅ MET

━━ LATENCY (P95, weekly average) ━━
api-gateway      : 312ms     (target <500ms)  ✅
query-service    : 1,840ms   (target <2000ms) ✅
embedding-svc    : 420ms     (target <600ms)  ✅
postgres-queries : 48ms      (target <100ms)  ✅
redis-ops        : 0.9ms     (target <5ms)    ✅

━━ ERROR RATES ━━
4xx Client Errors    : 0.8%   (normal)
5xx Server Errors    : 0.3%   (elevated Mar 14, normalized post-incident)
Timeout Rate         : 0.05%

━━ INFRASTRUCTURE ━━
CPU utilization (avg)  : 34%  (peak 71% during incident)
Memory utilization     : 61%  (Redis now at 52% after expansion)
DB connections (peak)  : 312  / 500 max pool
S3 storage             : 4.2 TB  (+0.3 TB vs prior week)
Kubernetes nodes       : 24 active / 2 standby

━━ DEPLOYMENT SUMMARY ━━
Successful deployments  : 7
Rollbacks               : 0
Change requests approved: 3
Change requests pending : 5

━━ INCIDENTS ━━
Open incidents     : 0
SEV-1 (closed)     : INC-2025-031 (March 14, 2025) — CLOSED
SEV-2 (closed)     : INC-2025-028 (March 9, 2025) — CLOSED
MTTDetect (avg)    : 4.2 min  | MTTResolve (avg): 47 min
"""),

    # ─── Legal / Compliance ───────────────────────────────────────────────────
    Document(
        doc_id="LEG-PRIV-001", title="Data Retention, Privacy & Compliance Policy v2.1",
        source_type="pdf", department="legal", classification="internal",
        allowed_roles=["legal","admin","hr","engineering"],
        content="""
DATA RETENTION, PRIVACY & COMPLIANCE POLICY — v2.1
Effective: January 1, 2025 | Approved by: Chief Legal Officer
Applicable Regulations: GDPR, CCPA, SOX, HIPAA (limited), PCI-DSS (scope)

━━ 1. SCOPE ━━
Applies to all personal data, operational data, and system logs processed by
SimplifyX Inc. including customer, employee, and partner data.

━━ 2. DATA RETENTION SCHEDULE ━━
Data Category           | Retention Period  | Legal Basis
Customer PII            | 7 years           | GDPR Art. 17; CCPA
Employee Records        | 7 years post-term | Employment law
Financial Records       | 10 years          | SOX compliance
System Access Logs      | 2 years           | Security audit
Application Logs        | 90 days rolling   | Operational need
Backup Snapshots        | 30-day rolling    | 1 annual kept 7 years
Marketing/Consent Data  | 3 years or revoke | Consent-based
Contractual Documents   | 10 years post-end | Contract law

━━ 3. DATA SUBJECT RIGHTS ━━
Right to Access    : Fulfilled within 30 days of valid request
Right to Erasure   : PII deleted within 14 business days
Right to Portability: Machine-readable export (JSON/CSV) on request
Right to Restrict  : Processing paused pending review
Marketing Opt-Out  : Immediate effect, no re-enrollment without consent

━━ 4. BREACH NOTIFICATION ━━
• Internal escalation: within 1 hour of discovery → Security + Legal
• Regulatory (GDPR Art. 33): notify within 72 hours if high-risk
• Customer notification: within 7 days if rights/freedoms at risk
• Incident log maintained in Legal's SharePoint (legal/incidents/)

━━ 5. THIRD-PARTY DATA PROCESSORS ━━
All vendors processing personal data must:
• Sign a Data Processing Agreement (DPA) before go-live
• Pass security questionnaire (SOC2 or ISO 27001 preferred)
• Appear on Approved Vendor List (maintained by Legal + Procurement)

Current approved processors: AWS, Google Workspace, Salesforce, Workday,
Zoom, Slack, GitHub Enterprise, Pinecone, OpenAI (DPA v4.1 on file).

━━ 6. VIOLATIONS & ENFORCEMENT ━━
Violations must be reported to legal@simplifyx.com within 24 hours.
Repeat violations: disciplinary action up to and including termination.
Intentional violations: referred to appropriate authorities.

Next review: January 2026 | Owner: Legal Team
"""),

    Document(
        doc_id="LEG-SOC2-002", title="SOC 2 Type II Audit Report — FY2024",
        source_type="pdf", department="legal", classification="restricted",
        allowed_roles=["legal","admin"],
        content="""
SOC 2 TYPE II COMPLIANCE AUDIT REPORT — FY 2024
Auditor        : Deloitte & Touche LLP
Audit Period   : January 1, 2024 – December 31, 2024
Report Date    : February 28, 2025
Trust Criteria : Security · Availability · Processing Integrity
                 Confidentiality · Privacy (all five TSCs)

━━ EXECUTIVE SUMMARY ━━
187 controls evaluated across five Trust Service Criteria.

Result Summary:
  Controls Passed      : 181  (96.8%)
  Minor Exceptions     : 5    (2.7%)
  Material Weaknesses  : 1    (0.5%)

━━ MATERIAL WEAKNESS — MW-2024-01 ━━
Control   : CC6.3 — Privileged Access Management
Finding   : 3 former employees retained active admin credentials for >30 days
            post-termination. One account showed 7 login events post-departure.
Risk      : HIGH — potential unauthorized data access
Mgmt Resp : Emergency IAM audit completed Dec 12, 2024. Automated
            offboarding workflow deployed (Okta + AWS IAM). Deprovisioning
            now occurs within 2 hours of Workday termination event.
Status    : REMEDIATED (verified by auditor Dec 20, 2024)

━━ MINOR EXCEPTIONS ━━
ME-01 (CC7.2): 2 emergency changes lacked post-implementation approval docs
ME-02 (CC9.1): Annual risk assessment delayed 6 weeks (completed Oct 2024)
ME-03 (A1.2) : 4 EC2 instances lacked memory utilization alerts (now resolved)
ME-04 (PI1.4): 1 batch job missing reconciliation check (added Dec 2024)
ME-05 (P6.5) : 1 product page missing GDPR cookie consent banner (fixed Oct 2024)

━━ AUDITOR OPINION ━━
With the noted material weakness (now remediated) and minor exceptions,
controls were operating effectively during the audit period for the remaining
181 controls. A clean unqualified opinion is expected for FY2025 if
remediation actions are maintained.

━━ NEXT STEPS ━━
• FY2025 audit engagement: October 2025 – January 2026 (Deloitte retained)
• Interim control testing: June 2025 (internal)
• Quarterly access reviews: Q2, Q3, Q4 2025 scheduled

DISTRIBUTION: Legal department, Executive team, Board Audit Committee ONLY.
This document is RESTRICTED. Unauthorized disclosure may violate NDA obligations.
"""),

]

# ═══════════════════════════════════════════════════════════════════════════════
#  USERS
# ═══════════════════════════════════════════════════════════════════════════════

USERS: list[User] = [
    User("U001","Alice Chen",    "admin",       "executive",   "restricted"),
    User("U002","Bob Martinez",  "hr",          "hr",          "confidential"),
    User("U003","Carol Singh",   "finance",     "finance",     "confidential"),
    User("U004","Dave Kim",      "engineering", "engineering", "internal"),
    User("U005","Eve Okonkwo",   "legal",       "legal",       "restricted"),
    User("U006","Frank Liu",     "analyst",     "operations",  "internal"),
    User("U007","Grace Patel",   "guest",       "external",    "public"),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORT SUPPORTING DATA FILES
# ═══════════════════════════════════════════════════════════════════════════════

def write_data_files(out: str = "data"):
    os.makedirs(out, exist_ok=True)

    # users.csv
    with open(f"{out}/users.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id","name","role","department","clearance_level"])
        for u in USERS:
            w.writerow([u.user_id,u.name,u.role,u.department,u.clearance_level])

    # access_policies.json
    with open(f"{out}/access_policies.json","w") as f:
        json.dump({
            "clearance_hierarchy": ["public","internal","confidential","restricted"],
            "role_clearance_map": {
                "admin":"restricted","legal":"restricted",
                "finance":"confidential","hr":"confidential",
                "engineering":"internal","analyst":"internal","guest":"public"
            },
            "default_policy": "deny",
            "note": "Two-layer RBAC: clearance rank + per-document role whitelist"
        }, f, indent=2)

    # audit_events.json — synthetic system log
    import random, hashlib
    events = []
    actions = ["LOGIN","DOC_VIEW","QUERY","EXPORT","API_CALL","SETTINGS_CHANGE",
               "PASSWORD_RESET","REPORT_DOWNLOAD","ADMIN_ACTION","DATA_EXPORT"]
    docs    = [d.doc_id for d in DOCUMENTS]
    for i in range(80):
        u = USERS[i % len(USERS)]
        events.append({
            "event_id":  f"EVT-{2000+i}",
            "timestamp": f"2025-03-{(i%28)+1:02d}T{(i%23)+1:02d}:{(i*3%60):02d}:00Z",
            "user_id":   u.user_id,
            "user_name": u.name,
            "role":      u.role,
            "action":    actions[i % len(actions)],
            "resource":  docs[i % len(docs)],
            "ip":        f"10.{i%5}.{(i*7)%255}.{(i*13)%255}",
            "status":    "FAILED" if i % 9 == 0 else "SUCCESS",
            "user_agent":"Mozilla/5.0 (corporate browser)",
        })
    with open(f"{out}/audit_events.json","w") as f:
        json.dump(events, f, indent=2)

    print(f"✅ Data files written to ./{out}/")

if __name__ == "__main__":
    write_data_files()
    print(f"Documents : {len(DOCUMENTS)}")
    print(f"Users     : {len(USERS)}")
    print(f"Formats   : {set(d.source_type for d in DOCUMENTS)}")
    print(f"Depts     : {set(d.department for d in DOCUMENTS)}")
