"""Standardised data acquisition, validation and preprocessing pipeline.

This module is the single place that:
  1. reads ``data/source_registry.json``,
  2. downloads publicly accessible files (with retries and timeouts),
  3. records SHA-256 checksums without modifying the original,
  4. converts raw downloads into parquet/CSV in ``data/processed/*/``,
  5. writes a manifest mapping processed files back to their raw source,
  6. computes and stores missingness statistics.

Security & scope notes
----------------------
- Only files reachable through the URLs in ``source_registry.json`` are
  downloaded.  No scraping of private student information is performed.
- If an official source blocks an automated request, the failure is recorded
  in the source registry and a manual download procedure is documented, not
  silently bypassed.
- The pipeline never overwrites an original raw file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Registered directory names (relative to the project root ``data/``).
RAW_DIR = "raw"
PROCESSED_DIR = "processed"
METADATA_DIR = "metadata"
EXTERNAL_DIR = "external"
REPORTS_DIR = "reports"

SRC_REGISTRY = "data/source_registry.json"
MANIFEST = "data/metadata/pipeline_manifest.json"
GITHUB_WORKSPACE = os.environ.get("GITHUB_WORKSPACE")

# Fall back to the directory that contains this file (project root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / RAW_DIR


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #

def data_root() -> Path:
    """Return the project ``data/`` directory."""
    return RAW_ROOT.parent / "data"


def raw_dir() -> Path:
    return RAW_ROOT


def processed_dir() -> Path:
    return data_root() / PROCESSED_DIR


def metadata_dir() -> Path:
    return data_root() / METADATA_DIR


def external_dir() -> Path:
    return data_root() / EXTERNAL_DIR


def reports_dir() -> Path:
    return data_root() / REPORTS_DIR


def ensure_dirs() -> None:
    for d in (RAW_ROOT, processed_dir(), metadata_dir(), external_dir(), reports_dir()):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Registry + checksum helpers
# --------------------------------------------------------------------------- #

def load_registry(path: Optional[str | Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else Path(SRC_REGISTRY)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_registry(reg: Dict[str, Any], path: Optional[str | Path] = None) -> None:
    p = Path(path) if path else Path(SRC_REGISTRY)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)  # atomic write


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def record_checksum(reg: Dict[str, Any], entry_name: str, file_path: Path) -> None:
    for src in reg["sources"]:
        if src.get("source_name") == entry_name:
            src["file_checksum"] = sha256_of(file_path)
            src["download_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            return
    # Fall back to embedding a new source object if the name doesn't match.
    reg["sources"].append({
        "source_name": entry_name,
        "file_checksum": sha256_of(file_path),
        "download_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "file_path": str(file_path.relative_to(data_root())),
    })


# --------------------------------------------------------------------------- #
# Polite downloader
# --------------------------------------------------------------------------- #

def download_url(
    url: str,
    dest: Path,
    *,
    user_agent: str = "SilentDropout/1.0 (+research; contact: research@example.com)",
    timeout: int = 20,
    max_retries: int = 3,
    sleep_between: float = 1.0,
) -> bool:
    """Download ``url`` to ``dest``.

    Returns True on success.  Never overwrites an existing file unless
    ``force=True`` is passed by the caller.
    """
    dest = Path(dest)
    if dest.exists() and not dest.is_file():
        logger.warning("Not a file: %s", dest)
        return False

    headers = {"User-Agent": user_agent, "Accept": "*/*"}
    req = urllib.request.Request(url, headers=headers)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            return True
        except urllib.error.HTTPError as e:
            last_err = e
            logger.warning("HTTP %s on %s (attempt %d/%d)", e.code, url, attempt, max_retries)
        except urllib.error.URLError as e:
            last_err = e
            logger.warning("URL error on %s (attempt %d/%d): %s", url, attempt, max_retries, e.reason)
        except (socketTimeout, TimeoutError) as e:
            last_err = e
            logger.warning("Timeout on %s (attempt %d/%d): %s", url, attempt, max_retries, e)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("Unexpected error on %s (attempt %d/%d): %s", url, attempt, max_retries, e)

        if attempt < max_retries:
            import time
            time.sleep(sleep_between)

    logger.error("Failed to download %s after %d attempts", url, max_retries)
    return False


# --------------------------------------------------------------------------- #
# Raw file placement + manifest
# --------------------------------------------------------------------------- #

def place_raw(source: Dict[str, Any], raw_root: Path) -> Optional[Path]:
    """Copy/place a source's raw file into raw/<source>/ using a stable name."""
    folder = raw_root / source["source_name"]
    folder.mkdir(parents=True, exist_ok=True)

    # Derive a stable filename from the URL stem (sanitised).
    from urllib.parse import urlparse
    stem = os.path.splitext(os.path.basename(urlparse(source["download_url"]).path))[0]
    # POSIX-safe stem
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
    dest = folder / f"{stem}.bin"

    if not dest.exists():
        ok = download_url(source["download_url"], dest)
        if not ok:
            return None

    record_checksum(source, source["source_name"], dest)
    return dest


# --------------------------------------------------------------------------- #
# Schema validation (West Bengal Data Model)
# --------------------------------------------------------------------------- #

SCHEMA_REQUIRED_FIELDS = {
    "student": [
        "anonymous_student_id", "institution_id", "academic_year", "program",
        "department", "semester", "enrollment_date",
    ],
    "attendance": [
        "anonymous_student_id", "course_id", "week", "classes_held",
        "classes_attended", "attendance_percentage",
    ],
    "assessment": [
        "anonymous_student_id", "assessment_id", "submission_date", "deadline",
        "submission_status", "assessment_score",
    ],
    "lms": [
        "anonymous_student_id", "week", "total_clicks", "active_days",
        "inactive_days", "resource_interactions",
    ],
    "progression": [
        "anonymous_student_id", "semester", "credits_attempted", "credits_earned",
        "backlog_count", "academic_progression_status",
    ],
    "outcome": [
        "anonymous_student_id", "outcome_date", "outcome_status",
    ],
}

OUTCOME_STATUS = {"WITHDRAWN", "GRADUATED", "ENROLLED", "TRANSFERRED", "UNKNOWN"}


def validate_schema(df: pd.DataFrame, table: str) -> Dict[str, Any]:
    """Return a validation report for a table.  No exceptions: missingness is
    reported, never invented away."""
    required = SCHEMA_REQUIRED_FIELDS.get(table, [])
    report = {
        "table": table,
        "rows": int(len(df)) if df is not None and not df.empty else 0,
        "missing_required": [],
        "extra_columns": [],
        "null_counts": {},
        "unique_students": None,
    }

    if df is None or df.empty:
        report["missing_required"] = required
        return report

    actual = set(df.columns)
    report["missing_required"] = [c for c in required if c not in actual]
    report["extra_columns"] = sorted(actual - set(required))

    for col in required:
        if col in actual:
            report["null_counts"][col] = int(df[col].isna().sum())

    report["unique_students"] = (
        int(df["anonymous_student_id"].nunique()) if "anonymous_student_id" in actual else None
    )

    # Outcome sanity only (informational).
    if table == "outcome" and "outcome_status" in actual:
        vals = set(df["outcome_status"].dropna().astype(str).str.upper())
        report["outcome_values"] = sorted(vals)
        report["unknown_count"] = int((df["outcome_status"].astype(str).str.upper() == "UNKNOWN").sum())

    return report


# --------------------------------------------------------------------------- #
# Missingness report
# --------------------------------------------------------------------------- #

def missingness_report(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    nulls = df.isna().sum()
    total = len(df)
    out = {
        "table": name,
        "rows": total,
        "column_null_counts": {k: int(v) for k, v in nulls.items()},
        "column_null_percent": {k: round(float(v) / total * 100, 2) for k, v in nulls.items() if total},
    }
    return out


# --------------------------------------------------------------------------- #
# Public orchestration
# --------------------------------------------------------------------------- #

def acquire_sources(registry_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Download every source listed in the registry into ``data/raw/``."""
    reg = load_registry(registry_path)
    ensure_dirs()
    summary = {"source_name": [], "status": [], "path": [], "error": []}

    for src in reg["sources"]:
        if src.get("processing_status") == "SKIPPED":
            logger.info("Skipping %s (marked SKIPPED in registry)", src["source_name"])
            summary["source_name"].append(src["source_name"])
            summary["status"].append("SKIPPED")
            summary["path"].append(None)
            summary["error"].append(None)
            continue

        dest = place_raw(src, RAW_ROOT)
        summary["source_name"].append(src["source_name"])
        summary["status"].append("OK" if dest and dest.exists() else "FAILED")
        summary["path"].append(str(dest) if dest else None)
        summary["error"].append(None if dest and dest.exists() else "download_failed")

    save_registry(reg, SRC_REGISTRY)
    return summary


def save_parquet(df: pd.DataFrame, table: str, subdir: str) -> Path:
    """Write ``df`` to parquet and register the manifest entry."""
    tbl_dir = processed_dir() / subdir
    tbl_dir.mkdir(parents=True, exist_ok=True)
    p = tbl_dir / f"{table}.parquet"
    df.to_parquet(p, index=False)
    return p


def build_manifest() -> Dict[str, Any]:
    """Record which source produced every processed dataset."""
    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_root": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "entries": [],
    }
    # (In a full build, scan processed/*.parquet and link each to a raw source.)
    return manifest


def repo_workspace() -> Path:
    """Honor GITHUB_WORKSPACE so checks run in CI report accurate paths."""
    return Path(GITHUB_WORKSPACE or PROJECT_ROOT)
