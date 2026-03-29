import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.recourse.german_recourse import solve_german_recourse_numeric_only

PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ART = ROOT / "artifacts"


def load_german():
    X = pd.read_csv(PROC / "german" / "X.csv")
    y = pd.read_csv(PROC / "german" / "y.csv").iloc[:, 0].astype(int)
    train_idx = np.load(SPLITS / "german" / "train_idx.npy")
    test_idx = np.load(SPLITS / "german" / "test_idx.npy")
    return X, y, train_idx, test_idx


def main():
    pipe = joblib.load(ART / "models" / "german_logreg.joblib")
    X, y, train_idx, test_idx = load_german()

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    neg_ids = np.where(pred == 0)[0]
    if len(neg_ids) == 0:
        print("No negative instances found in test split (unexpected).")
        return

    # Choose the rejected instance closest to boundary
    i = int(neg_ids[np.argmax(proba[neg_ids])])
    x0 = X_test.iloc[i]
    y0 = int(y_test.iloc[i])
    p0 = float(proba[i])

    key_cols = [c for c in ["duration", "credit_amount", "installment_commitment", "existing_credits", "residence_since", "age"] if c in x0.index]

    print(f"\n[Selected test instance] idx={i}, true_y={y0}, pred_proba={p0:.4f}, pred={(p0 >= 0.5)}")
    print("Original (key cols):")
    print(x0[key_cols])

    # Moderate plausibility: duration -24 max, credit_amount -50% max (only decrease for these two)
    mutable = ("duration", "credit_amount", "installment_commitment", "existing_credits", "residence_since")

    x_cf, slack = solve_german_recourse_numeric_only(
        pipe=pipe,
        X_train=X_train,
        x0=x0,
        mutable_cols=mutable,
        direction="auto",
        margin=1e-6,
        slack_penalty=1000.0,
        duration_max_decrease=24,
        credit_max_rel_decrease=0.50,
        enforce_decrease_for=("duration", "credit_amount"),
        integer_cols=("duration", "credit_amount", "installment_commitment", "existing_credits", "residence_since"),
    )

    p1 = float(pipe.predict_proba(pd.DataFrame([x_cf]))[:, 1][0])
    flipped = (p0 < 0.5 and p1 >= 0.5)

    print("\n[Recourse suggestion] (moderate plausibility + integer vars)")
    print(x_cf[key_cols])

    print(f"\nSlack needed (0 means feasible flip): {slack:.6f}")
    print(f"Proba before: {p0:.4f} | after: {p1:.4f} | flipped={flipped}")

    deltas = {}
    for c in mutable:
        if c in x0.index:
            deltas[c] = float(x_cf[c]) - float(x0[c])
    print("\nDelta (x_cf - x0):")
    print(pd.Series(deltas))


if __name__ == "__main__":
    main()

