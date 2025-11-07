#!/usr/bin/env python3
"""
Inspect indexed chunks for the most recently saved document/session.

Usage:
  python backend/scripts/inspect_chunks.py [--limit 200]

Prints chunk id, length, first 220 chars, and key metadata. Filters by the
latest saved lesson_session's session_id/doc_id when possible.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

# Ensure the backend directory (which contains the `app` package) is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)  # .../backend
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.core.database import db
from app.core.vectorstore import vectorstore


def get_latest_session() -> Dict[str, Any]:
    sessions = db.get_all_lesson_sessions()
    return sessions[0] if sessions else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    latest = get_latest_session()
    if not latest:
        print("No saved sessions found.")
        return

    session_id = latest.get("session_id", "")
    # session ids use convention session_{doc_id}; derive doc_id if present
    doc_id = session_id.replace("session_", "") if session_id.startswith("session_") else None

    print(f"Inspecting chunks for session: {session_id} (doc_id={doc_id})")

    data = vectorstore.collection.get(include=["documents", "metadatas", "ids"])
    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []
    ids = data.get("ids", []) or []

    # Filter by doc_id if present in metadata; otherwise show all
    rows = []
    for i in range(len(docs)):
        m = metas[i] or {}
        if doc_id and m.get("doc_id") != doc_id:
            continue
        rows.append((ids[i], docs[i] or "", m))

    if not rows and docs:
        # Fallback: no doc_id metadata found; show first N entries
        rows = list(zip(ids, docs, metas))

    rows = rows[: args.limit]

    print(f"Found {len(rows)} chunk(s) (showing up to {args.limit}).\n")
    for i, (cid, text, meta) in enumerate(rows, start=1):
        preview = text[:220].replace("\n", " ")
        print(f"[{i}] id={cid}")
        print(f"     len={len(text)}  meta={json.dumps(meta, ensure_ascii=False)}")
        print(f"     preview=\"{preview}\"\n")


if __name__ == "__main__":
    main()


