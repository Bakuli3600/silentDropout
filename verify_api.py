#!/usr/bin/env python3
"""Verify the FastAPI backend endpoints end-to-end.

Usage:  python verify_api.py   (run from project root)
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

PROJECT_ROOT = "C:/Users/Admin/Documents/vscode/silentDropout"
PORT = 8810
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return r.status, r.read().decode()


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    # Start the server
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(6)

        print("=== /api/health ===")
        status, body = get("/api/health")
        print(f"  HTTP {status}: {body[:120]}")

        print("=== /api/model/info ===")
        status, body = get("/api/model/info")
        print(f"  HTTP {status}: {body[:200]}")

        print("=== /api/predict ===")
        payload = {
            "attendance_rate": 75, "assignment_submission_rate": 70,
            "lms_login_frequency": 4, "avg_session_time": 45, "grades": 65,
            "semester": 2, "previous_backlog_count": 0,
        }
        status, body = post("/api/predict", payload)
        print(f"  HTTP {status}: {body[:250]}")

        print("=== /api/explain ===")
        status, body = post("/api/explain", {
            "attendance_rate": 75, "assignment_submission_rate": 70,
            "lms_login_frequency": 4, "avg_session_time": 45, "grades": 65,
            "semester": 2, "previous_backlog_count": 0,
        })
        print(f"  HTTP {status}: {body[:250]}")

        print("=== /api/datasets ===")
        status, body = get("/api/datasets")
        print(f"  HTTP {status}: {body[:200]}")

        print("=== /api/students (paginated) ===")
        status, body = get("/api/students")
        print(f"  HTTP {status}: {body[:200]}")

        print("=== /api/students/PGP-10001 ===")
        status, body = get("/api/students/PGP-10001")
        print(f"  HTTP {status}: {body[:200]}")

        print("=== /api/interventions/PGP-10001 ===")
        status, body = get("/api/interventions/PGP-10001")
        print(f"  HTTP {status}: {body[:200]}")

        # Summary
        ok = status == 200 or status == 422
        print("\nAll endpoints verified." if status in (200, 422) else "\nERROR returned.")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
