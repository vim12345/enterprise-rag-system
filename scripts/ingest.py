"""
scripts/ingest.py  —  Real File Ingestion
==========================================
Ingests actual PDF, CSV, and JSON files into the RAG pipeline.

Usage:
  python scripts/ingest.py --file report.pdf --dept finance --class confidential
  python scripts/ingest.py --file data.csv   --dept hr      --class internal
  python scripts/ingest.py --file logs.json  --dept engineering --class internal
  python scripts/ingest.py --dir ./my_docs/  --dept legal   --class restricted

Requirements (install as needed):
  pip install pdfplumber pandas     # for PDF + CSV
"""
from __future__ import annotations
import sys, os, json, argparse, csv, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.rag_engine import Document

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n\n".join(pages).strip()
        print(f"  PDF: extracted {len(text)} chars from {len(pages)} pages")
        return text
    except ImportError:
        print("  ⚠️  pdfplumber not installed. Run: pip install pdfplumber")
        print("  Falling back to filename as content.")
        return f"[PDF document: {os.path.basename(path)}]"
    except Exception as e:
        print(f"  ⚠️  PDF parse error: {e}")
        return ""

def parse_csv(path: str) -> str:
    """Convert CSV to readable text table."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
        lines = [f"CSV Dataset: {os.path.basename(path)}",
                 f"Rows: {len(df)} | Columns: {list(df.columns)}",
                 "", df.to_string(index=False)]
        print(f"  CSV: {len(df)} rows × {len(df.columns)} cols")
        return "\n".join(lines)
    except ImportError:
        # Fallback: stdlib csv
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        lines = [" | ".join(row) for row in rows]
        print(f"  CSV (stdlib): {len(rows)} rows")
        return "\n".join(lines)
    except Exception as e:
        print(f"  ⚠️  CSV parse error: {e}")
        return ""

def parse_json(path: str) -> str:
    """Flatten JSON/JSONL to readable text."""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        # Try JSONL first
        if content.startswith("{") and "\n" in content:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
            text = f"JSON Log: {os.path.basename(path)}\n{len(records)} records\n\n"
            text += "\n".join(json.dumps(r, indent=2) for r in records[:20])
        else:
            data = json.loads(content)
            text = f"JSON Document: {os.path.basename(path)}\n\n"
            text += json.dumps(data, indent=2)
        print(f"  JSON: {len(text)} chars")
        return text
    except Exception as e:
        print(f"  ⚠️  JSON parse error: {e}")
        return ""

def parse_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()

# ── Main ingestion ─────────────────────────────────────────────────────────────

PARSERS = {
    ".pdf":  ("pdf",  parse_pdf),
    ".csv":  ("csv",  parse_csv),
    ".json": ("json", parse_json),
    ".jsonl":("json", parse_json),
    ".txt":  ("text", parse_text),
    ".md":   ("text", parse_text),
}

def file_to_document(path: str, department: str, classification: str,
                      allowed_roles: list[str]) -> Document | None:
    ext = os.path.splitext(path)[1].lower()
    if ext not in PARSERS:
        print(f"  ⚠️  Unsupported format: {ext}")
        return None

    source_type, parser = PARSERS[ext]
    print(f"\n📄 Ingesting: {os.path.basename(path)} [{source_type.upper()}]")
    content = parser(path)

    if not content.strip():
        print(f"  ⚠️  No content extracted from {path}")
        return None

    doc_id = hashlib.md5(path.encode()).hexdigest()[:10].upper()
    title  = os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()

    doc = Document(
        doc_id        = f"ING-{doc_id}",
        title         = title,
        content       = content,
        source_type   = source_type,
        department    = department,
        classification= classification,
        allowed_roles = allowed_roles,
        metadata      = {"original_path": path, "file_size": os.path.getsize(path)},
    )
    print(f"  ✅ Created doc_id={doc.doc_id} | {len(content.split())} words")
    return doc

def ingest_files(paths: list[str], department: str, classification: str,
                 allowed_roles: list[str]) -> list[Document]:
    docs = []
    for path in paths:
        if os.path.isdir(path):
            for fname in os.listdir(path):
                fpath = os.path.join(path, fname)
                if os.path.isfile(fpath):
                    doc = file_to_document(fpath, department, classification, allowed_roles)
                    if doc:
                        docs.append(doc)
        else:
            doc = file_to_document(path, department, classification, allowed_roles)
            if doc:
                docs.append(doc)
    return docs

def save_ingested(docs: list[Document], out: str = "data/ingested_docs.json"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    records = [{
        "doc_id": d.doc_id, "title": d.title,
        "source_type": d.source_type, "department": d.department,
        "classification": d.classification, "allowed_roles": d.allowed_roles,
        "word_count": len(d.content.split()), "metadata": d.metadata,
    } for d in docs]
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n📋 Saved ingestion manifest → {out}")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest real files (PDF/CSV/JSON/TXT) into Enterprise RAG"
    )
    parser.add_argument("--file", help="Path to a single file")
    parser.add_argument("--dir",  help="Path to a directory of files")
    parser.add_argument("--dept", required=True,
                        choices=["hr","finance","engineering","legal","operations"],
                        help="Department this document belongs to")
    parser.add_argument("--class", dest="classification", required=True,
                        choices=["public","internal","confidential","restricted"],
                        help="Security classification level")
    parser.add_argument("--roles", default="",
                        help="Comma-separated allowed roles (empty = all authenticated users)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo ingestion on sample files")
    args = parser.parse_args()

    allowed_roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    if args.demo:
        print("\n🔧 Demo mode: creating sample files and ingesting them…")
        os.makedirs("demo_files", exist_ok=True)

        # Create demo files
        with open("demo_files/q2_report.txt", "w") as f:
            f.write("Q2 2025 Revenue: $26.1M (+16% YoY)\nEBITDA: $5.1M (19.5% margin)\nARR: $98.3M")
        with open("demo_files/team_data.csv", "w") as f:
            f.write("name,role,dept,start_date\nJohn Smith,Engineer,Engineering,2023-01-15\n"
                    "Jane Doe,Analyst,Finance,2022-06-01\nBob Lee,Manager,HR,2021-03-20\n")
        with open("demo_files/alerts.json", "w") as f:
            json.dump([{"id":"ALT-001","ts":"2025-04-01T10:00Z","type":"CPU_HIGH",
                        "service":"query-svc","value":"92%"},
                       {"id":"ALT-002","ts":"2025-04-01T10:05Z","type":"LATENCY",
                        "service":"api-gw","value":"820ms"}], f)

        paths = ["demo_files/q2_report.txt", "demo_files/team_data.csv", "demo_files/alerts.json"]
        docs  = ingest_files(paths, "engineering", "internal", [])
        save_ingested(docs, "data/ingested_docs.json")

        print(f"\n✅ Demo complete: {len(docs)} documents ingested")
        print("   Add them to the pipeline with: pipeline.add_document(doc)")
        sys.exit(0)

    paths = []
    if args.file: paths.append(args.file)
    if args.dir:  paths.append(args.dir)
    if not paths:
        parser.print_help()
        sys.exit(1)

    docs = ingest_files(paths, args.dept, args.classification, allowed_roles)
    save_ingested(docs)

    print(f"\n{'='*50}")
    print(f"  ✅ Ingested {len(docs)} document(s)")
    print(f"  Add them to your pipeline:")
    print(f"    from scripts.ingest import ingest_files")
    print(f"    docs = ingest_files(['{paths[0]}'], ...)")
    print(f"    for doc in docs: pipeline.add_document(doc)")
    print(f"    pipeline.build_index()")
    print(f"{'='*50}\n")
