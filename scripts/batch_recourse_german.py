import sys
from pathlib import Path
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.recourse.german_recourse import solve_german_recourse_numeric_only

PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ART = ROOT / "artifacts"


def add_age_bucket(age_series: pd.Series) -> pd.Series:
    return pd.cut(
        age_series.astype(float),
        bins=[0, 25, 40, 60, 120],
        labels=["<=25", "26-40", "41-60", "60+"],
        include_lowest=True
    )


def load_german_test():
    X = pd.read_csv(PROC / "german" / "X.csv")
    y = pd.read_csv(PROC / "german" / "y.csv").iloc[:, 0].astype(int)
    A = pd.read_csv(PROC / "german" / "A.csv")

    test_idx = np.load(SPLITS / "german" / "test_idx.npy")
    train_idx = np.load(SPLITS / "german" / "train_idx.npy")

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    A_test = A.iloc[test_idx].reset_index(drop=True)

    if "age" in A_test.columns:
        A_test["age_bucket"] = add_age_bucket(A_test["age"])

    return X_train, X_test, y_test, A_test


def main(args):
    model_path = ART / "models" / "german_logreg.joblib"
    pipe = joblib.load(model_path)

    X_train, X_test, y_test, A_test = load_german_test()

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    neg_ids = np.where(pred == 0)[0]
    print(f"[INFO] German test size: {len(X_test)}")
    print(f"[INFO] Rejected (pred=0) count: {len(neg_ids)}")

    if len(neg_ids) == 0:
        print("[WARN] No rejected instances to generate recourse for.")
        return

    print("\n[SETTINGS]")
    print(f"  duration_max_decrease = {args.duration_max_decrease}")
    print(f"  credit_max_rel_decrease = {args.credit_max_rel_decrease}")
    print(f"  direction (other vars) = {args.direction}")
    print(f"  enforce_decrease_for = {args.enforce_decrease_for}")
    print(f"  integer vars = {args.integer_vars}")
    print(f"  slack_penalty = {args.slack_penalty}\n")

    # Shuffle for unbiased ordering if limit is used
    rng = np.random.default_rng(args.seed)
    neg_ids = neg_ids.copy()
    rng.shuffle(neg_ids)

    if args.limit is not None:
        neg_ids = neg_ids[: min(args.limit, len(neg_ids))]
        print(f"[INFO] Using limit={args.limit} => running on {len(neg_ids)} rejected instances")

    mutable = tuple(args.mutable_vars)

    rows = []
    t0 = time.time()

    for k, i in enumerate(neg_ids, start=1):
        x0 = X_test.iloc[i]
        p0 = float(proba[i])

        x_cf, slack = solve_german_recourse_numeric_only(
            pipe=pipe,
            X_train=X_train,
            x0=x0,
            mutable_cols=mutable,
            direction=args.direction,
            margin=1e-6,
            slack_penalty=args.slack_penalty,
            duration_max_decrease=args.duration_max_decrease,
            credit_max_rel_decrease=args.credit_max_rel_decrease,
            enforce_decrease_for=tuple(args.enforce_decrease_for),
            integer_cols=tuple(args.integer_vars),
        )

        p1 = float(pipe.predict_proba(pd.DataFrame([x_cf]))[:, 1][0])
        flipped = int((p0 < 0.5) and (p1 >= 0.5) and (slack == 0.0))

        def d(col):
            if col in x0.index:
                return float(x_cf[col]) - float(x0[col])
            return np.nan

        sex = A_test.loc[i, "sex"] if "sex" in A_test.columns else ""
        age_bucket = A_test.loc[i, "age_bucket"] if "age_bucket" in A_test.columns else ""

        rows.append({
            "test_idx": int(i),
            "p_before": p0,
            "p_after": p1,
            "slack": float(slack),
            "flipped": flipped,
            "delta_duration": d("duration"),
            "delta_credit_amount": d("credit_amount"),
            "sex": str(sex),
            "age_bucket": str(age_bucket),
        })

        if k % 10 == 0 or k == len(neg_ids):
            print(f"[PROGRESS] {k}/{len(neg_ids)} done")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    out_dir = ART / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"german_recourse_results_d{args.duration_max_decrease}_c{args.credit_max_rel_decrease:.2f}.csv"
    df.to_csv(out_path, index=False)

    feasibility = (df["slack"] == 0.0).mean()
    flip_rate = df["flipped"].mean()

    print("\n[OK] Saved:", out_path)
    print(f"[SUMMARY] Instances evaluated: {len(df)}")
    print(f"[SUMMARY] Feasible (slack==0) rate: {feasibility:.3f}")
    print(f"[SUMMARY] Flip success rate: {flip_rate:.3f}")
    print(f"[SUMMARY] Avg p_before: {df['p_before'].mean():.3f} | Avg p_after: {df['p_after'].mean():.3f}")
    print(f"[SUMMARY] Avg delta_duration: {df['delta_duration'].mean():.3f}")
    print(f"[SUMMARY] Avg delta_credit_amount: {df['delta_credit_amount'].mean():.3f}")
    print(f"[SUMMARY] Total elapsed seconds: {elapsed:.2f}")

    if df["sex"].astype(str).str.len().gt(0).any():
        print("\n[GROUP] Flip rate by sex:")
        print(df.groupby("sex")["flipped"].mean())

    if df["age_bucket"].astype(str).str.len().gt(0).any():
        print("\n[GROUP] Flip rate by age_bucket:")
        print(df.groupby("age_bucket")["flipped"].mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--limit", type=int, default=None, help="Max rejected instances (default: all).")
    ap.add_argument("--seed", type=int, default=42)

    # Option B knobs
    ap.add_argument("--duration_max_decrease", type=int, default=36)          # Option B default
    ap.add_argument("--credit_max_rel_decrease", type=float, default=0.65)    # Option B default
    ap.add_argument("--direction", type=str, default="auto", choices=["auto", "increase", "decrease", "both"])
    ap.add_argument("--slack_penalty", type=float, default=1000.0)

    # What variables are allowed to change
    ap.add_argument("--mutable_vars", nargs="+",
                    default=["duration", "credit_amount", "installment_commitment", "existing_credits", "residence_since"])
    ap.add_argument("--enforce_decrease_for", nargs="+", default=["duration", "credit_amount"])
    ap.add_argument("--integer_vars", nargs="+",
                    default=["duration", "credit_amount", "installment_commitment", "existing_credits", "residence_since"])

    args = ap.parse_args()
    main(args)
