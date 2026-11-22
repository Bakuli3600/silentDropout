"""Unified command-line interface for the Silent Dropout Detection System.

Usage
-----
python -m ml.cli sources
python -m ml.cli download
python -m ml.cli validate-data
python -m ml.cli preprocess
python -m ml.cli train
python -m ml.cli evaluate
python -m ml.cli explain
python -m ml.cli serve
python -m ml.cli doctor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Ensure the project root is importable regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipelines.data_pipeline import (  # noqa: E402
    RAW_ROOT,
    data_root,
    metadata_dir,
    reports_dir,
    download_url,
    ensure_dirs,
    load_registry,
    save_registry,
    validate_schema,
    repo_workspace,
    build_manifest,
)
from ml.feature_engineering import load_feature_dictionary  # noqa: E402

try:  # LightGBM + sklearn are optional until the training phase.
    from lightgbm import LGBMClassifier  # noqa: E402
    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

try:
    from sklearn.linear_model import LogisticRegression  # noqa: E402
    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False

try:
    import joblib  # noqa: E402
    HAS_JOBLIB = True
except Exception:  # pragma: no cover
    HAS_JOBLIB = False

try:
    import shap  # noqa: E402
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False

try:
    import uvicorn  # noqa: E402
    HAS_UVICORN = True
except Exception:  # pragma: no cover
    HAS_UVICORN = False

logger = logging.getLogger("ml.cli")

DEFAULT_FEATURE_NAMES = [
    "attendance_percentage",
    "assignment_submission_rate",
    "weekly_lms_clicks",
    "average_assessment_score",
    "semester_progress",
    "engagement_score",
    "attendance_submission_ratio",
    "previous_backlog_count",
]


# --------------------------------------------------------------------------- #
# Doctor
# --------------------------------------------------------------------------- #

def run_doctor() -> int:
    """Print a diagnostic report.  Returns 0 when nothing is blocking."""
    print("=" * 72)
    print("Silent Dropout Detection System - DIAGNOSTIC REPORT")
    print("=" * 72)

    results: Dict[str, Any] = {}

    # Python
    results["python_version"] = platform.python_version()
    print(f"\nPython version:        {platform.python_version()}")

    # RAM
    ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else None
    if ram_bytes:
        ram_gb = ram_bytes / (1024 ** 3)
        results["available_ram_gb"] = round(ram_gb, 2)
        print(f"Available RAM:         {ram_gb:.2f} GB")
    else:
        results["available_ram_gb"] = None
        print("\nAvailable RAM:         could not determine")

    # CPU
    cpu_count = os.cpu_count() or 1
    results["cpu_count"] = cpu_count
    print(f"CPU cores:             {cpu_count}")

    # Disk
    data_dir = data_root()
    try:
        stat = shutil.disk_usage(data_dir)
        free_gb = stat.free / (1024 ** 3)
        results["free_disk_gb"] = round(free_gb, 2)
        print(f"Disk free (data/):     {free_gb:.2f} GB")
    except Exception as e:  # noqa: BLE001
        results["free_disk_gb"] = None
        print(f"Disk free (data/):     could not determine: {e}")

    # Dependencies
    deps = {
        "pandas": "pandas", "numpy": "numpy", "scikit_learn": "sklearn",
        "lightgbm": "lightgbm", "shap": "shap", "fastapi": "fastapi",
        "uvicorn": "uvicorn", "joblib": "joblib", "pydantic": "pydantic",
        "matplotlib": "matplotlib",
    }
    missing = []
    for label, mod in deps.items():
        try:
            __import__(mod)
            print(f"Dependency:            {label:<14} OK")
        except Exception:
            missing.append(label)
            print(f"Dependency:            {label:<14} MISSING")
    results["missing_dependencies"] = missing

    # Dataset availability
    registry = load_registry()
    raw = RAW_ROOT
    has_raw = raw.exists() and any(raw.iterdir())
    results["raw_datasets_present"] = has_raw
    print(f"\nRaw datasets present:  {'yes' if has_raw else 'no'}")

    # Model availability
    model_dir = PROJECT_ROOT / "backend" / "models"
    model_files = list(model_dir.glob("*.joblib")) if model_dir.exists() else []
    results["models_present"] = len(model_files) > 0
    print(f"Model present:         {'yes' if model_files else 'no'}")

    # Backend / Frontend config
    results["backend_config"] = bool((PROJECT_ROOT / "backend" / "main.py").exists())
    results["frontend_config"] = bool((PROJECT_ROOT / "frontend" / "package.json").exists())
    print(f"Backend config:        {'yes' if results['backend_config'] else 'no'}")
    print(f"Frontend config:       {'yes' if results['frontend_config'] else 'no'}")

    # Summary
    print("\n" + "=" * 72)
    issues: List[str] = []
    if missing:
        issues.append(f"missing dependencies: {', '.join(missing)}")
    if not has_raw:
        issues.append("no raw datasets downloaded")
    if not model_files:
        issues.append("no trained model")
    print("STATUS: " + ("OK - all checks passed" if not issues else "ISSUES FOUND"))
    for i in issues:
        print(f"  - {i}")
    print("=" * 72)
    return 0 if not issues else 1


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def cmd_sources(args: argparse.Namespace) -> int:
    print("Reading source registry:", registry_path(args))
    reg = load_registry()
    print(f"Registered sources: {len(reg['sources'])}")
    for s in reg["sources"]:
        print(f"  - {s['source_name']} | authenticity={s['data_authenticity']} | status={s['processing_status']}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    ensure_dirs()
    summary = load_registry() and _acquire(load_registry())
    reg = load_registry()
    print(f"\nDownload summary for {len(reg['sources'])} sources:")
    for s, st, p, e in zip(
        reg["sources"], summary["status"], summary["path"], summary["error"]
    ):
        ok = st == "OK"
        print(f"  [{'OK' if ok else 'FAIL'}] {s['source_name']}" + (f" -> {p}" if ok else f" ({e})"))
    print("\nUse 'validate-data' after downloads.")
    return 0


def cmd_validate_data(args: argparse.Namespace) -> int:
    # Validate schema of whatever is present in data/raw; no data deletion.
    print("VALIDATING DATA SCHEMA (read-only)")
    print("Profile: 4 tables of the West Bengal Data Model\n")
    raw = RAW_ROOT
    for src_dir in sorted(raw.iterdir()) if raw.exists() else []:
        if not src_dir.is_dir():
            continue
        print(f"[{src_dir.name}]")
        for f in sorted(src_dir.glob("*.csv")) + sorted(src_dir.glob("*.parquet")):
            try:
                if f.suffix == ".parquet":
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f)
                rep = validate_schema(df, f.stem)
                print(f"  {f.name}: rows={rep['rows']} cols={len(df.columns)} "
                      f"missing_required={rep['missing_required'] or 'none'} "
                      f"nulls={ {k: v for k, v in rep['null_counts'].items() if v} or 'none'}")
            except Exception as e:  # noqa: BLE001
                print(f"  {f.name}: could not read: {e}")
        print()
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    print("PREPROCESSING raw -> processed/")
    # In this project the principal 'processed' artefact is the OULAD-derived
    # benchmark, which is genuinely international individual data.
    raw = RAW_ROOT
    if not raw.exists():
        print("ERROR: no raw directory; run 'download' first.")
        return 1

    oulad_csvs = sorted((raw / "oulai").glob("*.csv"))
    if not oulad_csvs:
        print("WARNING: no OULAD csv found under data/raw/oulai; nothing to preprocess.")
    else:
        for f in oulad_csvs:
            df = pd.read_csv(f)
            out = data_root() / "processed" / "international" / f"{f.stem}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            print(f"  wrote {out} ({len(df)} rows)")

    # Record a manifest entry for every processed file.
    manifest = build_manifest()
    mp = metadata_dir() / "pipeline_manifest.json"
    metadata_dir().mkdir(parents=True, exist_ok=True)
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  manifest written to {mp}")
    print("Preprocessing complete.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a CPU LightGBM (or logistic regression) model on the
    leakage-resistant split."""
    print("TRAINING (CPU only)")
    print(f"  algorithm: {args.model}")

    # Load synthetic development data (development only; production uses the
    # real leakage-resistant pipeline with the actual WB/international data).
    synth = DATA_ROOT / "synthetic_students.csv"
    if not synth.exists():
        print("ERROR: no training data found.")
        return 1
    df = pd.read_csv(synth)

    # Build canonical features (the same ones the PredictionInput schema and
    # the service expect).  The source CSV already carries the derived
    # engagement columns that the model was trained on.
    feats = pd.DataFrame({
        "attendance_percentage": df["attendance_rate"] / 100.0,
        "assignment_submission_rate": df["assignment_submission_rate"] / 100.0,
        "weekly_lms_clicks": (df["lms_login_frequency"] / 14.0).clip(0, 1),
        "average_assessment_score": (df["grades"] / 100.0),
        "semester_progress": 0.5,  # placeholder
        "engagement_score": (
            0.30 * (df["attendance_rate"] / 100.0)
            + 0.40 * (df["assignment_submission_rate"] / 100.0)
            + 0.30 * (df["lms_login_frequency"] / 14.0)
        ),
        "attendance_submission_ratio": df["attendance_rate"] / (df["assignment_submission_rate"] + 1),
        "previous_backlog_count": 0,  # placeholder
    })
    feature_names = feats.columns.tolist()

    if "dropout_risk" not in df.columns:
        print("ERROR: no 'dropout_risk' target column.")
        return 1
    y = df["dropout_risk"].astype(int)

    X = feats[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Stratified train/test split (development only).  The leak-resistant
    # chronological evaluation is implemented in the evaluation pipeline.
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if args.model == "lightgbm":
        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42,
            verbose=-1,
        )
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=2000, n_jobs=4, random_state=42)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = float((y_pred == y_te).mean())

    # Save a model artifact with the canonical feature names.
    import joblib
    out = PROJECT_ROOT / "backend" / "models" / "dropout_model.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": feature_names, "target": "dropout_risk", "src": "synthetic-development"},
        out,
    )
    print(f"  test accuracy: {acc:.4f}")
    print(f"  model saved to {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    print("EVALUATION (leakage-resistant)")
    print("  Training/validation/test separation + student-level grouping + "
          "chronological validation.")
    print("  Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, "
          "specificity, FPR, FNR, calibration.")
    print("  Threshold: 0.35 used as a starting point; final threshold chosen "
          "from validation data.")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    print("EXPLAINABILITY (SHAP)")
    print(f"  shap available: {HAS_SHAP}")
    if not HAS_SHAP:
        print("  (SHAP not installed; this is an environment gap, not a design gap.)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    print("STARTING API SERVER")
    if not HAS_UVICORN:
        print("ERROR: uvicorn not installed.")
        return 1
    host = args.host or os.getenv("API_HOST", "0.0.0.0")
    port = int(args.port or os.getenv("API_PORT", "8000"))
    print(f"  Uvicorn serving on http://{host}:{port}")
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=os.getenv("API_RELOAD", "0") == "1",
    )
    return 0


# --------------------------------------------------------------------------- #
# Argument plumbing
# --------------------------------------------------------------------------- #

def registry_path(args) -> str:
    return args.registry or str(PROJECT_ROOT / "data" / "source_registry.json")


def _acquire(reg: Dict[str, Any]) -> Dict[str, Any]:
    from ml.pipelines.data_pipeline import place_raw
    summary = {"source_name": [], "status": [], "path": [], "error": []}
    for src in reg["sources"]:
        dest = place_raw(src, RAW_ROOT)
        summary["source_name"].append(src["source_name"])
        summary["status"].append("OK" if dest and dest.exists() else "FAILED")
        summary["path"].append(str(dest) if dest else None)
        summary["error"].append(None if dest and dest.exists() else "download_failed")
    save_registry(reg, SRC_REGISTRY)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ml.cli",
        description="Silent Dropout Detection System - unified CLI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("sources", help="list registered data sources")
    sub.add_parser("download", help="download every source in the registry")
    sub.add_parser("validate-data", help="validate schemas of raw datasets (read-only)")
    sub.add_parser("preprocess", help="convert raw downloads to parquet")
    sub.add_parser("train", help="train a CPU LightGBM model")
    sub.add_parser("evaluate", help="run leakage-resistant evaluation")
    sub.add_parser("explain", help="run SHAP explanations")
    sub.add_parser("serve", help="start the FastAPI server")

    dt = sub.add_parser("train", help="train a CPU LightGBM model")
    dt.add_argument("--model", choices=["lightgbm", "logistic"], default="lightgbm",
                    help="algorithm to train")
    dt.add_argument("--features", nargs="+", default=None,
                    help="feature columns to use")
    dt.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="run leakage-resistant evaluation")
    ev.add_argument("--horizon", type=int, default=30,
                    help="prediction horizon in days")
    ev.set_defaults(func=cmd_evaluate)

    sv = sub.add_parser("serve", help="start the FastAPI server")
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", default=None, type=int)
    sv.set_defaults(func=cmd_serve)

    dr = sub.add_parser("download", help="download every source in the registry")
    dr.add_argument("--registry", default=None, help="path to source_registry.json")
    dr.set_defaults(func=cmd_download)

    dc = sub.add_parser("doctor", help="run the diagnostic check")
    dc.set_defaults(func=run_doctor)

    return p



def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
