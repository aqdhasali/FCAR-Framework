from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"

DATASETS = ["adult", "german", "default_credit"]

def make_split(name: str, seed: int = 42, test_size: float = 0.2):
    X = pd.read_csv(PROC / name / "X.csv")
    y = pd.read_csv(PROC / name / "y.csv").iloc[:, 0].astype(int)

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )

    out = SPLITS / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_idx.npy", train_idx)
    np.save(out / "test_idx.npy", test_idx)

    print(f"[OK] {name}: train={len(train_idx)}, test={len(test_idx)}, pos_rate={y.mean():.3f}")

if __name__ == "__main__":
    SPLITS.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        make_split(ds)
