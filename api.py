"""
FCAR REST API — NFR04 Implementation.

Provides JSON endpoints for programmatic access to the FCAR framework:
  - POST /recourse          → Generate recourse for an individual (UC-101)
  - GET  /audit/{dataset}   → Retrieve MISOB audit scores (UC-103)
  - GET  /benchmark/{dataset} → Benchmark comparison results (UC-104)
  - GET  /health            → Health check
  - GET  /datasets          → List available datasets and their configs

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import time
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.config_loader import (
    load_dataset_config,
    get_mutable_numeric_cols,
    get_mutable_categorical_cols,
    get_numeric_cost_weights,
    get_categorical_step_weights,
    get_sensitive_attributes,
    VALID_DATASETS,
)
from src.recourse.generic_recourse_mip import solve_recourse_mip

PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ART = ROOT / "artifacts"

# ───────────────────────── App Setup ─────────────────────────

app = FastAPI(
    title="FCAR API",
    description=(
        "Fairness Constrained Actionable Recourse — REST API.\n\n"
        "Generate fair, actionable recourse plans for individuals denied by "
        "AI credit-scoring models, and retrieve group-level MISOB audit reports."
    ),
    version="1.0.0",
    contact={"name": "Aqdhas Ali", "url": "https://github.com/aqdhasali/FCAR-Framework"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────── Caches ────────────────────────────

_model_cache: dict = {}


def _load(dataset_name: str):
    """Load model + data for a dataset, with caching."""
    if dataset_name in _model_cache:
        return _model_cache[dataset_name]

    model_path = ART / "models" / f"{dataset_name}_logreg.joblib"
    if not model_path.exists():
        raise HTTPException(404, f"Model not found for dataset '{dataset_name}'.")

    pipe = joblib.load(model_path)
    config = load_dataset_config(dataset_name)

    X = pd.read_csv(PROC / dataset_name / "X.csv")
    A = pd.read_csv(PROC / dataset_name / "A.csv")

    train_idx = np.load(SPLITS / dataset_name / "train_idx.npy")
    test_idx = np.load(SPLITS / dataset_name / "test_idx.npy")

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    A_test = A.iloc[test_idx].reset_index(drop=True)

    bundle = {
        "pipe": pipe,
        "config": config,
        "X_train": X_train,
        "X_test": X_test,
        "A_test": A_test,
    }
    _model_cache[dataset_name] = bundle
    return bundle


# ───────────────────────── Schemas ───────────────────────────


class RecourseRequest(BaseModel):
    """Request body for generating a recourse plan."""
    dataset: str = Field(
        ..., description="Dataset name: 'german', 'adult', or 'default_credit'."
    )
    test_index: int = Field(
        ..., description="Index into the test set (rejected applicant)."
    )
    use_fcar: bool = Field(
        True, description="Apply FCAR fairness-adjusted weights."
    )
    group_col: Optional[str] = Field(
        None,
        description=(
            "Sensitive attribute column for FCAR weight lookup. "
            "If omitted, the first sensitive attribute in the config is used."
        ),
    )
    slack_penalty: float = Field(
        1000.0, description="MIP slack penalty (higher = stricter feasibility)."
    )


class RecourseChange(BaseModel):
    feature: str
    action: str
    before: str
    after: str
    amount: Optional[str] = None
    cost_weight: float


class RecourseResponse(BaseModel):
    dataset: str
    test_index: int
    use_fcar: bool
    fcar_group: Optional[str] = None
    score_before: float
    score_after: float
    flipped: bool
    slack: float
    changes: list[RecourseChange]
    narrative: str
    solve_time_seconds: float


class AuditResponse(BaseModel):
    dataset: str
    group_col: str
    method: str
    audit_score: float
    gap: float
    threshold: float
    passed: bool
    worst_group: str
    best_group: str
    social_burden: dict[str, float]


class DatasetInfo(BaseModel):
    dataset: str
    description: str
    mutable_numeric: list[str]
    mutable_categorical: list[str]
    immutable: list[str]
    sensitive_attributes: list[str]


# ───────────────────────── Endpoints ─────────────────────────


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "FCAR API", "version": "1.0.0"}


@app.get("/datasets", response_model=list[DatasetInfo], tags=["System"])
def list_datasets():
    """List all supported datasets and their feature configurations."""
    result = []
    for ds in VALID_DATASETS:
        try:
            config = load_dataset_config(ds)
            result.append(
                DatasetInfo(
                    dataset=ds,
                    description=config.get("description", ""),
                    mutable_numeric=get_mutable_numeric_cols(config),
                    mutable_categorical=get_mutable_categorical_cols(config),
                    immutable=list(config.get("immutable", [])),
                    sensitive_attributes=get_sensitive_attributes(config),
                )
            )
        except Exception:
            pass
    return result


@app.post("/recourse", response_model=RecourseResponse, tags=["Recourse (UC-101)"])
def generate_recourse(req: RecourseRequest):
    """
    Generate an optimal recourse plan for a rejected applicant (UC-101 / UC-105).

    Returns the required feature changes, a natural-language narrative,
    and projected model score after applying the recourse.
    """
    if req.dataset not in VALID_DATASETS:
        raise HTTPException(400, f"Invalid dataset. Choose from: {list(VALID_DATASETS)}")

    bundle = _load(req.dataset)
    pipe = bundle["pipe"]
    config = bundle["config"]
    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    A_test = bundle["A_test"]

    if req.test_index < 0 or req.test_index >= len(X_test):
        raise HTTPException(400, f"test_index must be 0..{len(X_test)-1}")

    x0 = X_test.iloc[req.test_index]
    target_cls = int(config.get("label_positive", 1))
    p0 = float(pipe.predict_proba(pd.DataFrame([x0]))[:, 1][0])

    # Check if actually rejected
    is_rejected = (p0 < 0.5) if target_cls == 1 else (p0 >= 0.5)
    if not is_rejected:
        raise HTTPException(
            400,
            f"Applicant at test_index={req.test_index} is NOT rejected (score={p0:.4f}). "
            "Recourse is only generated for denied applicants.",
        )

    # Build per-instance config with weight overrides
    num_w = dict(get_numeric_cost_weights(config))
    cat_w = dict(get_categorical_step_weights(config))
    fcar_group_label = None

    if req.use_fcar:
        group_cols = get_sensitive_attributes(config)
        gcol = req.group_col or (group_cols[0] if group_cols else None)
        if gcol:
            gval = str(A_test.iloc[req.test_index].get(gcol, ""))
            summary_path = ART / "reports" / "benchmarks" / f"{req.dataset}_{gcol}_ab_summary.json"
            if summary_path.exists():
                with open(summary_path) as fh:
                    summary = json.load(fh)
                overrides = summary.get("fcar_overrides", {})
                if gval in overrides:
                    num_w.update({k: float(v) for k, v in overrides[gval].get("num", {}).items()})
                    cat_w.update({k: float(v) for k, v in overrides[gval].get("cat", {}).items()})
                    fcar_group_label = f"{gcol}={gval}"

    inst_config = dict(config)
    for col in num_w:
        if col in inst_config.get("mutable_numeric", {}):
            inst_config["mutable_numeric"][col]["cost_weight"] = num_w[col]
    for col in cat_w:
        if col in inst_config.get("mutable_categorical", {}):
            inst_config["mutable_categorical"][col]["step_weight"] = cat_w[col]
    inst_config.setdefault("solver", {})["slack_penalty"] = req.slack_penalty

    # Solve
    t0 = time.perf_counter()
    try:
        x_cf, slack = solve_recourse_mip(pipe, X_train, x0, inst_config)
    except Exception as e:
        raise HTTPException(500, f"Solver error: {e}")
    elapsed = time.perf_counter() - t0

    p1 = float(pipe.predict_proba(pd.DataFrame([x_cf]))[:, 1][0])
    if target_cls == 1:
        flipped = (p0 < 0.5) and (p1 >= 0.5) and (slack == 0.0)
    else:
        flipped = (p0 >= 0.5) and (p1 < 0.5) and (slack == 0.0)

    # Build change list
    changes: list[RecourseChange] = []
    parts: list[str] = []

    for c in get_mutable_numeric_cols(config):
        v0 = float(x0.get(c, np.nan))
        v1 = float(x_cf.get(c, np.nan))
        if pd.notna(v0) and pd.notna(v1) and abs(v1 - v0) > 1e-4:
            direction = "Increase" if v1 > v0 else "Decrease"
            changes.append(
                RecourseChange(
                    feature=c,
                    action=direction,
                    before=f"{v0:.2f}",
                    after=f"{v1:.2f}",
                    amount=f"{abs(v1 - v0):.2f}",
                    cost_weight=num_w.get(c, 1.0),
                )
            )
            parts.append(f"{direction} {c} by {abs(v1 - v0):.2f} (from {v0:.2f} to {v1:.2f})")

    for c in get_mutable_categorical_cols(config):
        v0 = str(x0.get(c, ""))
        v1 = str(x_cf.get(c, ""))
        if v0 != v1:
            changes.append(
                RecourseChange(
                    feature=c,
                    action="Change",
                    before=v0,
                    after=v1,
                    cost_weight=cat_w.get(c, 0.25),
                )
            )
            parts.append(f"Change {c} from {v0} to {v1}")

    narrative = (
        "To overturn the negative decision, the applicant must: " + "; ".join(parts) + "."
        if parts
        else "No feature changes required (edge case)."
    )

    return RecourseResponse(
        dataset=req.dataset,
        test_index=req.test_index,
        use_fcar=req.use_fcar,
        fcar_group=fcar_group_label,
        score_before=round(p0, 6),
        score_after=round(p1, 6),
        flipped=flipped,
        slack=round(float(slack), 6),
        changes=changes,
        narrative=narrative,
        solve_time_seconds=round(elapsed, 3),
    )


@app.get(
    "/audit/{dataset}",
    response_model=list[AuditResponse],
    tags=["Audit (UC-103)"],
)
def get_audit(dataset: str, method: Optional[str] = None):
    """
    Retrieve the MISOB fairness audit scores for a dataset (UC-103).

    Returns audit results for both unconstrained AR and FCAR by default.
    Use `?method=fcar` or `?method=ar` to filter.
    """
    if dataset not in VALID_DATASETS:
        raise HTTPException(400, f"Invalid dataset. Choose from: {list(VALID_DATASETS)}")

    bench_dir = ART / "reports" / "benchmarks"
    results: list[AuditResponse] = []

    for f in sorted(bench_dir.glob(f"{dataset}_*_ab_summary.json")):
        with open(f) as fh:
            s = json.load(fh)

        for m_key, m_label in [("unconstrained_ar", "ar"), ("fcar", "fcar")]:
            if method and m_label != method:
                continue
            d = s[m_key]
            results.append(
                AuditResponse(
                    dataset=s["dataset"],
                    group_col=s["group_col"],
                    method=m_label,
                    audit_score=d["audit"]["audit_score"],
                    gap=d["disparity"]["gap"],
                    threshold=d["audit"]["threshold"],
                    passed=d["audit"].get("passed", False),
                    worst_group=str(d["disparity"]["worst_group"]),
                    best_group=str(d["disparity"]["best_group"]),
                    social_burden={str(k): round(float(v), 6) for k, v in d["social_burden"].items()},
                )
            )

    if not results:
        raise HTTPException(404, f"No benchmark results found for '{dataset}'. Run benchmark_ab.py first.")

    return results


@app.get("/benchmark/{dataset}", tags=["Benchmark (UC-104)"])
def get_benchmark(dataset: str):
    """
    Retrieve the full A/B benchmark comparison for a dataset (UC-104).

    Returns the complete JSON summary including disparity reduction,
    statistical tests, and per-group Social Burden breakdown.
    """
    if dataset not in VALID_DATASETS:
        raise HTTPException(400, f"Invalid dataset. Choose from: {list(VALID_DATASETS)}")

    bench_dir = ART / "reports" / "benchmarks"
    results = []

    for f in sorted(bench_dir.glob(f"{dataset}_*_ab_summary.json")):
        with open(f) as fh:
            results.append(json.load(fh))

    if not results:
        raise HTTPException(404, f"No benchmark results found for '{dataset}'.")

    return results
