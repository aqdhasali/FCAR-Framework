import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.modeling.preprocess import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ART = ROOT / "artifacts"
(ART / "models").mkdir(parents=True, exist_ok=True)
(ART / "reports").mkdir(parents=True, exist_ok=True)


def load_dataset(name: str):
    X = pd.read_csv(PROC / name / "X.csv")
    y = pd.read_csv(PROC / name / "y.csv").iloc[:, 0].astype(int)
    A = pd.read_csv(PROC / name / "A.csv")
    train_idx = np.load(SPLITS / name / "train_idx.npy")
    test_idx = np.load(SPLITS / name / "test_idx.npy")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    A_train, A_test = A.iloc[train_idx], A.iloc[test_idx]

    return X_train, X_test, y_train, y_test, A_train, A_test


def train_logreg(name: str):
    X_train, X_test, y_train, y_test, A_train, A_test = load_dataset(name)

    pre = build_preprocessor(X_train)

    clf = LogisticRegression(
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        C=1.0,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    out_model = ART / "models" / f"{name}_logreg.joblib"
    joblib.dump(pipe, out_model)

    meta = {
        "dataset": name,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_test": float(y_test.mean()),
    }

    with open(ART / "reports" / f"{name}_train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] trained + saved: {out_model}")


if __name__ == "__main__":
    import sys
    train_logreg(sys.argv[1])
