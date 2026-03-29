import sys
from pathlib import Path

# --- make project root importable (fixes "No module named src") ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss, average_precision_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, equalized_odds_difference, demographic_parity_difference


PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ART = ROOT / "artifacts"
(ART / "reports").mkdir(parents=True, exist_ok=True)


def load_test(name: str):
    X = pd.read_csv(PROC / name / "X.csv")
    y = pd.read_csv(PROC / name / "y.csv").iloc[:, 0].astype(int)
    A = pd.read_csv(PROC / name / "A.csv")
    test_idx = np.load(SPLITS / name / "test_idx.npy")

    return X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True), A.iloc[test_idx].reset_index(drop=True)


def add_age_buckets(A_test: pd.DataFrame) -> pd.DataFrame:
    """
    Create age buckets for fairness slicing to avoid treating each unique age as a separate group.
    Drops raw age columns afterwards to keep fairness loop clean.
    """
    A = A_test.copy()

    # Default Credit uses "AGE"
    if "AGE" in A.columns:
        A["AGE_bucket"] = pd.cut(
            A["AGE"],
            bins=[0, 25, 40, 60, 120],
            labels=["<=25", "26-40", "41-60", "60+"],
            include_lowest=True
        )
        A = A.drop(columns=["AGE"])

    # German uses "age"
    if "age" in A.columns:
        A["age_bucket"] = pd.cut(
            A["age"],
            bins=[0, 25, 40, 60, 120],
            labels=["<=25", "26-40", "41-60", "60+"],
            include_lowest=True
        )
        A = A.drop(columns=["age"])

    return A


def evaluate(name: str):
    model_path = ART / "models" / f"{name}_logreg.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first: python scripts/train_baseline.py {name}")

    model = joblib.load(model_path)
    X_test, y_test, A_test = load_test(name)

    # --- bucket age for fairness slicing ---
    A_test = add_age_buckets(A_test)

    # --- predictions ---
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # --- quality metrics ---
    quality = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "pred_pos_rate": float(pred.mean()),
    }

    # --- fairness metrics ---
    fairness = {}
    metrics = {
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
    }

    for col in A_test.columns:
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_test,
            y_pred=pred,
            sensitive_features=A_test[col]
        )

        # mf.by_group is a DataFrame
        by_group = mf.by_group.copy()
        # Convert to plain dict for JSON
        fairness[col] = {
            "overall": {k: float(v) for k, v in mf.overall.items()},
            "by_group": by_group.astype(float).to_dict(),
            "gap_selection_rate": float(mf.difference()["selection_rate"]),
            "gap_tpr": float(mf.difference()["tpr"]),
            "gap_fpr": float(mf.difference()["fpr"]),
            "demographic_parity_diff": float(demographic_parity_difference(y_test, pred, sensitive_features=A_test[col])),
            "equalized_odds_diff": float(equalized_odds_difference(y_test, pred, sensitive_features=A_test[col])),
        }

    out = {"dataset": name, "quality": quality, "fairness": fairness}

    out_path = ART / "reports" / f"{name}_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] evaluation saved: {out_path}")
    print("Quality:", quality)
    print("Fairness keys:", list(fairness.keys()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate_baseline.py <adult|german|default_credit>")
        raise SystemExit(1)

    ds = sys.argv[1]
    print(f"[INFO] Evaluating dataset: {ds}")
    evaluate(ds)
    print("[DONE]")
