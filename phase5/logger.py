"""
Simple JSON-line audit logger for Streamlit app / pipeline runs.
Append-only file: phase5/audit.log
"""
import json
import time
from typing import Dict

AUDIT_LOG_PATH = "phase5/audit.log"

def audit_event(event: Dict):
    # extend event with timestamp
    ev = dict(event)
    ev.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    except Exception:
        # best-effort, do not crash the app
        pass
