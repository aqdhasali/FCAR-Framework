"""
Auto-FCAR tuning loop for German Credit recourse.

What it does
------------
1) Loads the trained German baseline model (artifacts/models/german_logreg.joblib)
2) Finds all rejected instances in the German TEST split (pred=0)
3) Runs MIP recourse (numeric + checking_status + savings_status)
4) Computes group burden stats (default: age_bucket) using avg_abs_amt (|Δcredit_amount|)
5) Iteratively updates group-specific weights to reduce burden disparity while keeping feasibility high

Outputs
-------
- Per-iteration CSVs under artifacts/reports/
- Console tables per iteration
- Final chosen overrides printed at the end
"""

import sys
from pathlib import Path
import time
import argparse

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recourse.german_recourse_mip import solve_german_recourse_mip  # noqa: E402

# Must match the monotonic orders used in german_recourse_mip.py
CHECKING_ORDER = ["A11", "A12", "A13", "A14"]
SAVINGS_ORDER = ["A61", "A62", "A63", "A64", "A65"]

PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ART = ROOT / "artifacts"


def add_age_bucket(age_series: pd.Series) -> pd.Series:
    # Safe labels (avoid <= for shell parsing)
    return pd.cut(
        age_series.astype(float),
        bins=[0, 25, 40, 60, 120],
        labels=["le25", "26-40", "41-60", "60+"],
        include_lowest=True,
    )


def rank_delta(before, after, order):
    b = str(before)
    a = str(after)
    if b not in order or a not in order:
        return np.nan
    return order.index(a) - order.index(b)


def load_german():
    X = pd.read_csv(PROC / "german" / "X.csv")
    y = pd.read_csv(PROC / "german" / "y.csv").iloc[:, 0].astype(int)
    A = pd.read_csv(PROC / "german" / "A.csv")

    train_idx = np.load(SPLITS / "german" / "train_idx.npy")
    test_idx = np.load(SPLITS / "german" / "test_idx.npy")

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    A_test = A.iloc[test_idx].reset_index(drop=True)

    if "age" in A_test.columns:
        A_test["age_bucket"] = add_age_bucket(A_test["age"])

    return X_train, X_test, y_test, A_test


def compute_group_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Robust grouping:
    - Prefer grouping by df[group_col] if present
    - Else fall back to df["group_val"]
    """
    d = df.copy()

    group_key = group_col if group_col in d.columns else "group_val"
    if group_key not in d.columns:
        raise KeyError(f"Neither '{group_col}' nor 'group_val' exist in the iteration dataframe.")

    d["abs_amt"] = d["delta_credit_amount"].abs()
    d["abs_dur"] = d["delta_duration"].abs()

    d["checking_steps"] = d.apply(
        lambda r: rank_delta(r["checking_status_before"], r["checking_status_after"], CHECKING_ORDER), axis=1
    )
    d["savings_steps"] = d.apply(
        lambda r: rank_delta(r["savings_status_before"], r["savings_status_after"], SAVINGS_ORDER), axis=1
    )

    grp = d.groupby(group_key).agg(
        n=("flipped", "size"),
        feasible_rate=("slack", lambda s: (s == 0.0).mean()),
        flip_rate=("flipped", "mean"),
        avg_abs_amt=("abs_amt", "mean"),
        avg_abs_dur=("abs_dur", "mean"),
        avg_check_steps=("checking_steps", "mean"),
        avg_save_steps=("savings_steps", "mean"),
        avg_p_after=("p_after", "mean"),
    )
    grp.index.name = group_col  # label nicely for printing
    return grp.sort_values("avg_abs_amt", ascending=False)


def run_one_iteration(
    *,
    pipe,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    A_test: pd.DataFrame,
    neg_ids: np.ndarray,
    group_col: str,
    base_num_weights: dict,
    base_cat_weights: dict,
    overrides: dict,
    duration_max_decrease: int,
    credit_max_rel_decrease: float,
    mutable_cat_cols: tuple,
    slack_penalty: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = neg_ids.copy()
    rng.shuffle(ids)

    rows = []
    t0 = time.time()

    for k, i in enumerate(ids, start=1):
        x0 = X_test.iloc[i]

        # fetch group value from A_test
        group_val = ""
        if group_col in A_test.columns:
            group_val = str(A_test.loc[i, group_col])

        # Build per-person weights
        num_w = dict(base_num_weights)
        cat_w = {
            "checking_status": float(base_cat_weights.get("checking_status", 0.25)),
            "savings_status": float(base_cat_weights.get("savings_status", 0.25)),
        }
        if group_val in overrides:
            num_w.update(overrides[group_val].get("num", {}))
            cat_w.update(overrides[group_val].get("cat", {}))

        x_cf, slack = solve_german_recourse_mip(
            pipe=pipe,
            X_train=X_train,
            x0=x0,
            duration_max_decrease=duration_max_decrease,
            credit_max_rel_decrease=credit_max_rel_decrease,
            mutable_cat_cols=mutable_cat_cols,
            num_weights=num_w,
            cat_step_weights=cat_w,
            slack_penalty=slack_penalty,
        )

        p0 = float(pipe.predict_proba(pd.DataFrame([x0]))[:, 1][0])
        p1 = float(pipe.predict_proba(pd.DataFrame([x_cf]))[:, 1][0])
        flipped = int((p0 < 0.5) and (p1 >= 0.5) and (slack == 0.0))

        dur0 = float(x0.get("duration", np.nan))
        dur1 = float(x_cf.get("duration", np.nan))
        amt0 = float(x0.get("credit_amount", np.nan))
        amt1 = float(x_cf.get("credit_amount", np.nan))

        row = {
            "test_idx": int(i),
            "group_col": group_col,
            "group_val": group_val,
            # ✅ critical: add a column named exactly like group_col (age_bucket or sex)
            group_col: group_val,

            "p_before": p0,
            "p_after": p1,
            "slack": float(slack),
            "flipped": flipped,

            "duration_before": dur0,
            "duration_after": dur1,
            "credit_amount_before": amt0,
            "credit_amount_after": amt1,
            "delta_duration": (dur1 - dur0) if np.isfinite(dur0) and np.isfinite(dur1) else np.nan,
            "delta_credit_amount": (amt1 - amt0) if np.isfinite(amt0) and np.isfinite(amt1) else np.nan,

            "checking_status_before": str(x0.get("checking_status", "")),
            "checking_status_after": str(x_cf.get("checking_status", "")),
            "savings_status_before": str(x0.get("savings_status", "")),
            "savings_status_after": str(x_cf.get("savings_status", "")),

            "w_credit_amount": float(num_w.get("credit_amount", 1.0)),
            "w_duration": float(num_w.get("duration", 1.0)),
            "w_check_step": float(cat_w.get("checking_status", 0.25)),
            "w_save_step": float(cat_w.get("savings_status", 0.25)),
        }
        rows.append(row)

        if k % 10 == 0 or k == len(ids):
            print(f"[PROGRESS] {k}/{len(ids)}")

    df = pd.DataFrame(rows)
    df.attrs["elapsed_sec"] = time.time() - t0
    return df


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--group_col", type=str, default="age_bucket", choices=["age_bucket", "sex"])
    ap.add_argument("--max_iters", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=500.0, help="Stop if (max-min) avg_abs_amt <= epsilon.")
    ap.add_argument("--min_group_n", type=int, default=5, help="Ignore groups smaller than this when choosing worst.")

    # Update rule knobs
    ap.add_argument("--credit_mult", type=float, default=1.5, help="Multiply credit_amount weight for worst group.")
    ap.add_argument("--duration_mult", type=float, default=1.1, help="Multiply duration weight for worst group.")
    ap.add_argument("--cat_mult", type=float, default=0.6, help="Multiply cat step weights (lower => cheaper cat moves).")

    ap.add_argument("--min_cat_weight", type=float, default=0.05)
    ap.add_argument("--max_credit_weight", type=float, default=5.0)
    ap.add_argument("--max_duration_weight", type=float, default=3.0)

    # Recourse constraints
    ap.add_argument("--duration_max_decrease", type=int, default=48)
    ap.add_argument("--credit_max_rel_decrease", type=float, default=0.80)
    ap.add_argument("--slack_penalty", type=float, default=1000.0)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    pipe = joblib.load(ART / "models" / "german_logreg.joblib")
    X_train, X_test, y_test, A_test = load_german()

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    neg_ids = np.where(pred == 0)[0]

    print(f"[INFO] Test size: {len(X_test)} | rejected: {len(neg_ids)}")
    print(f"[INFO] group_col={args.group_col}")

    # Base weights
    base_num_weights = {}  # defaults to 1.0 inside solver if missing
    base_cat_weights = {"checking_status": 0.25, "savings_status": 0.25}

    overrides = {}  # overrides[group] = {"num": {...}, "cat": {...}}

    out_dir = ART / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    mutable_cat_cols = ("checking_status", "savings_status")

    history = []

    for it in range(args.max_iters):
        print(f"\n========== Auto-FCAR Iteration {it} ==========")
        print(f"[CURRENT overrides] {overrides if overrides else '(none)'}")

        df = run_one_iteration(
            pipe=pipe,
            X_train=X_train,
            X_test=X_test,
            A_test=A_test,
            neg_ids=neg_ids,
            group_col=args.group_col,
            base_num_weights=base_num_weights,
            base_cat_weights=base_cat_weights,
            overrides=overrides,
            duration_max_decrease=args.duration_max_decrease,
            credit_max_rel_decrease=args.credit_max_rel_decrease,
            mutable_cat_cols=mutable_cat_cols,
            slack_penalty=args.slack_penalty,
            seed=args.seed + it,
        )

        csv_path = out_dir / f"german_autofcar_{args.group_col}_it{it}_d{args.duration_max_decrease}_c{args.credit_max_rel_decrease:.2f}.csv"
        df.to_csv(csv_path, index=False)

        feasible_rate = (df["slack"] == 0.0).mean()
        flip_rate = df["flipped"].mean()
        avg_abs_amt = df["delta_credit_amount"].abs().mean()
        avg_abs_dur = df["delta_duration"].abs().mean()
        elapsed = float(df.attrs.get("elapsed_sec", np.nan))

        print(f"[OK] saved: {csv_path}")
        print(f"[SUMMARY] feasible={feasible_rate:.3f} flip={flip_rate:.3f} avg|Δamt|={avg_abs_amt:.2f} avg|Δdur|={avg_abs_dur:.2f} elapsed={elapsed:.2f}s")

        grp = compute_group_table(df, args.group_col)
        print("\n[GROUP TABLE] (sorted by avg_abs_amt desc)")
        print(grp)

        history.append({"iter": it, "csv": str(csv_path), "feasible": feasible_rate, "flip": flip_rate})

        eligible = grp[grp["n"] >= args.min_group_n].copy()
        if eligible.empty:
            print("[WARN] No eligible groups (n too small). Stopping.")
            break

        gap = float(eligible["avg_abs_amt"].max() - eligible["avg_abs_amt"].min())
        worst_group = str(eligible["avg_abs_amt"].idxmax())
        best_group = str(eligible["avg_abs_amt"].idxmin())

        print(f"\n[DISPARITY] avg_abs_amt gap = {gap:.2f} (worst={worst_group}, best={best_group})")

        if gap <= args.epsilon:
            print(f"[STOP] gap <= epsilon ({args.epsilon}). Done.")
            break

        # Update overrides for worst group
        cur = overrides.get(worst_group, {"num": {}, "cat": {}})
        cur_num = cur.get("num", {})
        cur_cat = cur.get("cat", {})

        w_credit = float(cur_num.get("credit_amount", 1.0))
        w_dur = float(cur_num.get("duration", 1.0))
        w_check = float(cur_cat.get("checking_status", base_cat_weights["checking_status"]))
        w_save = float(cur_cat.get("savings_status", base_cat_weights["savings_status"]))

        new_credit = min(w_credit * args.credit_mult, args.max_credit_weight)
        new_dur = min(w_dur * args.duration_mult, args.max_duration_weight)
        new_check = max(w_check * args.cat_mult, args.min_cat_weight)
        new_save = max(w_save * args.cat_mult, args.min_cat_weight)

        overrides[worst_group] = {
            "num": {"credit_amount": new_credit, "duration": new_dur},
            "cat": {"checking_status": new_check, "savings_status": new_save},
        }

        print(
            f"[UPDATE] {worst_group}: "
            f"credit_amount w {w_credit:.3f}->{new_credit:.3f}, "
            f"duration w {w_dur:.3f}->{new_dur:.3f}, "
            f"cat w check {w_check:.3f}->{new_check:.3f}, "
            f"save {w_save:.3f}->{new_save:.3f}"
        )

    print("\n========== Auto-FCAR Finished ==========")
    print("[FINAL overrides]")
    print(overrides if overrides else "(none)")
    print("\n[HISTORY]")
    for h in history:
        print(h)


if __name__ == "__main__":
    main()