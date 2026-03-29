"""
Shared fixtures for the FCAR test-suite.

Provides lightweight, self-contained objects (mini-pipeline, tiny
training set, sample configs) so that unit tests do NOT depend on
real model artefacts or large CSV files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ────────────────────────────────────────────────────────────
# Minimal German-like config
# ────────────────────────────────────────────────────────────

@pytest.fixture()
def german_config():
    """Return a minimal config dict mimicking german.yaml."""
    return {
        "dataset": "german",
        "label_positive": 1,
        "mutable_numeric": {
            "duration": {
                "direction": "decrease",
                "max_decrease": 48,
                "integer": True,
                "cost_weight": 1.0,
            },
            "credit_amount": {
                "direction": "decrease",
                "max_rel_decrease": 0.80,
                "integer": True,
                "cost_weight": 1.0,
            },
            "installment_commitment": {
                "direction": "auto",
                "integer": True,
                "cost_weight": 1.0,
            },
        },
        "mutable_categorical": {
            "checking_status": {
                "order": ["A11", "A12", "A13", "A14"],
                "monotonic": True,
                "step_weight": 0.25,
            },
            "savings_status": {
                "order": ["A61", "A62", "A63", "A64", "A65"],
                "monotonic": True,
                "step_weight": 0.25,
            },
        },
        "immutable": ["age", "personal_status", "foreign_worker"],
        "sensitive_attributes": ["personal_status", "age"],
        "max_cat_changes": 1,
        "solver": {"slack_penalty": 1000.0, "margin": 1e-6},
    }


# ────────────────────────────────────────────────────────────
# Tiny synthetic data + fitted pipeline
# ────────────────────────────────────────────────────────────

def _make_toy_data(n=120, seed=42):
    """Generate a small DataFrame mimicking German credit features."""
    rng = np.random.RandomState(seed)

    df = pd.DataFrame({
        "duration": rng.randint(6, 72, n),
        "credit_amount": rng.randint(500, 15000, n),
        "installment_commitment": rng.randint(1, 5, n),
        "checking_status": rng.choice(["A11", "A12", "A13", "A14"], n),
        "savings_status": rng.choice(["A61", "A62", "A63", "A64", "A65"], n),
        "age": rng.randint(19, 70, n),
        "personal_status": rng.choice(["A91", "A92", "A93", "A94"], n),
        "foreign_worker": rng.choice(["A201", "A202"], n),
    })
    # Simple rule so we get a nontrivial logistic boundary
    y = (
        (df["duration"] < 30).astype(int)
        + (df["credit_amount"] < 5000).astype(int)
        + (df["checking_status"].isin(["A13", "A14"])).astype(int)
    )
    y = (y >= 2).astype(int)
    return df, y


def _fit_toy_pipeline(X, y):
    """
    Fit a Pipeline([("pre", ColumnTransformer), ("clf", LogisticRegression)])
    matching the structure expected by the MIP solver.
    """
    num_cols = ["duration", "credit_amount", "installment_commitment"]
    cat_cols = ["checking_status", "savings_status"]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("scaler", StandardScaler(with_mean=False))]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X, y)
    return pipe


@pytest.fixture()
def toy_data():
    """Return (X, y) — tiny DataFrame + labels."""
    return _make_toy_data()


@pytest.fixture()
def toy_pipeline(toy_data):
    """Return a fitted pipeline on the toy data."""
    X, y = toy_data
    return _fit_toy_pipeline(X, y)


@pytest.fixture()
def toy_train_test(toy_data):
    """Return X_train, X_test, y_train, y_test (80/20 split)."""
    X, y = toy_data
    split = int(0.8 * len(X))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


@pytest.fixture()
def rejected_applicant(toy_pipeline, toy_train_test):
    """Return (index, x0, p0) for a rejected applicant (p < 0.5)."""
    _, X_test, _, _ = toy_train_test
    proba = toy_pipeline.predict_proba(X_test[["duration", "credit_amount",
                                                "installment_commitment",
                                                "checking_status",
                                                "savings_status"]])[:, 1]
    rejected_mask = proba < 0.5
    if not rejected_mask.any():
        pytest.skip("No rejected applicants in toy data")
    idx = np.where(rejected_mask)[0][0]
    x0 = X_test.iloc[idx]
    return idx, x0, float(proba[idx])
