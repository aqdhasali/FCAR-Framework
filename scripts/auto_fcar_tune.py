"""
Auto-FCAR iterative tuning loop (Dataset Agnostic).

Runs recourse iteratively, computing the Social Burden (MISOB) at each step.
Identifies the most burdened group, and adjusts feature weights (making 
recourse "cheaper" for their common changes) to reduce disparity.
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

from src.recourse.generic_recourse_mip import solve_recourse_mip
from src.config.config_loader import (
    load_dataset_config,
    get_mutable_numeric_cols,
    get_mutable_categorical_cols,
    get_numeric_cost_weights,
    get_categorical_step_weights,
    get_sensitive_attributes,
)
from src.metrics.social_burden import compute_recourse_burden, compute_social_burden, compute_burden_disparity

PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ART = ROOT / "artifacts"


def load_test(dataset_name: str):
    X = pd.read_csv(PROC / dataset_name / "X.csv")
    y = pd.read_csv(PROC / dataset_name / "y.csv").iloc[:, 0].astype(int)
    A = pd.read_csv(PROC / dataset_name / "A.csv")

    test_idx = np.load(SPLITS / dataset_name / "test_idx.npy")
    train_idx = np.load(SPLITS / dataset_name / "train_idx.npy")

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    A_test = A.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_test, A_test


def run_one_iteration(
    *,
    dataset_name: str,
    pipe,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    A_test: pd.DataFrame,
    neg_ids: np.ndarray,
    proba: np.ndarray,
    group_col: str,
    config: dict,
    base_num_weights: dict,
    base_cat_weights: dict,
    overrides: dict,
    slack_penalty: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = neg_ids.copy()
    rng.shuffle(ids)

    mutable_num_cols = get_mutable_numeric_cols(config)
    mutable_cat_cols = get_mutable_categorical_cols(config)
    target_cls = int(config.get("label_positive", 1))
    
    rows = []
    t0 = time.time()

    for k, i in enumerate(ids, start=1):
        x0 = X_test.iloc[i]
        p0 = float(proba[i])

        group_val = str(A_test.loc[i, group_col]) if group_col in A_test.columns else ""

        # Build per-person weights
        num_w = dict(base_num_weights)
        cat_w = dict(base_cat_weights)
        
        if group_val in overrides:
            num_w.update(overrides[group_val].get("num", {}))
            cat_w.update(overrides[group_val].get("cat", {}))

        inst_config = dict(config)
        for col in num_w:
            inst_config["mutable_numeric"][col]["cost_weight"] = num_w[col]
        for col in cat_w:
            inst_config["mutable_categorical"][col]["step_weight"] = cat_w[col]
        inst_config["solver"]["slack_penalty"] = slack_penalty

        x_cf, slack = solve_recourse_mip(pipe, X_train, x0, inst_config)

        p1 = float(pipe.predict_proba(pd.DataFrame([x_cf]))[:, 1][0])
        
        if target_cls == 0:
            flipped = int((p0 >= 0.5) and (p1 < 0.5) and (slack == 0.0))
        else:
            flipped = int((p0 < 0.5) and (p1 >= 0.5) and (slack == 0.0))

        row = {
            "test_idx": int(i),
            "group_col": group_col,
            "group_val": group_val,
            group_col: group_val, # for social_burden module compatibility
            "p_before": p0,
            "p_after": p1,
            "slack": float(slack),
            "flipped": flipped,
        }

        for c in mutable_num_cols:
            val0 = float(x0.get(c, np.nan))
            val1 = float(x_cf.get(c, np.nan))
            row[f"{c}_before"] = val0
            row[f"{c}_after"] = val1
            row[f"delta_{c}"] = (val1 - val0) if np.isfinite(val0) and np.isfinite(val1) else np.nan
            row[f"w_{c}"] = float(num_w.get(c, 1.0))

        for col in mutable_cat_cols:
            row[f"{col}_before"] = str(x0.get(col, ""))
            row[f"{col}_after"] = str(x_cf.get(col, ""))
            row[f"w_{col}_step"] = float(cat_w.get(col, 0.25))

        rows.append(row)

        if k % 10 == 0 or k == len(ids):
            print(f"[PROGRESS] {k}/{len(ids)}")

    df = pd.DataFrame(rows)
    df.attrs["elapsed_sec"] = time.time() - t0
    
    # Calculate burden score aligned with solver objective.
    # Use per-individual weights (w_{feature} columns) so that FCAR-adjusted
    # weights for each group are reflected in the measured burden.
    # burden_i = sum( w_i * |delta_i| / range_i )
    num_ranges = {c: max(float(X_train[c].max() - X_train[c].min()), 1e-9) for c in mutable_num_cols}
    burden = pd.Series(0.0, index=df.index)
    for c in mutable_num_cols:
        dc = f"delta_{c}"
        wc = f"w_{c}"
        r = num_ranges[c]
        if dc in df.columns and wc in df.columns:
            burden += df[wc] * df[dc].abs() / r
    df["burden_total"] = burden
    
    return df


def main():
    ap = argparse.ArgumentParser(description="Agnostic Auto-FCAR tuning loop.")
    ap.add_argument("--dataset", type=str, required=True, choices=["german", "adult", "default_credit"])
    ap.add_argument("--group_col", type=str, required=True, help="Sensitive attribute to tune over")
    
    ap.add_argument("--max_iters", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.10, help="Stop if max_burden/min_burden - 1 <= epsilon.")
    ap.add_argument("--min_group_n", type=int, default=5, help="Ignore groups smaller than this.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of instances for quick testing.")

    # Update rule knobs (heuristic multipliers applied to the worst group's weights)
    # If the worst group has highest burden, we lower their feature weights to make recourse "cheaper"
    # Note: weights are used in the objective function. Lower weight = solver penalizes change less.
    ap.add_argument("--num_mult", type=float, default=0.8, help="Multiplier for numeric weights for worst group (<1 to reduce burden)")
    ap.add_argument("--cat_mult", type=float, default=0.8, help="Multiplier for cat weights for worst group (<1 to reduce burden)")
    
    ap.add_argument("--min_weight", type=float, default=0.01)
    
    ap.add_argument("--slack_penalty", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    dataset_name = args.dataset
    config = load_dataset_config(dataset_name)

    model_path = ART / "models" / f"{dataset_name}_logreg.joblib"
    pipe = joblib.load(model_path)
    X_train, X_test, y_test, A_test = load_test(dataset_name)

    target_cls = int(config.get("label_positive", 1))
    proba = pipe.predict_proba(X_test)[:, 1]
    
    if target_cls == 0:
        pred = (proba < 0.5).astype(int)
        neg_ids = np.where(proba >= 0.5)[0]
    else:
        pred = (proba >= 0.5).astype(int)
        neg_ids = np.where(pred == 0)[0]

    print(f"[INFO] Test size: {len(X_test)} | rejected: {len(neg_ids)}")
    print(f"[INFO] Tuning for dataset={dataset_name}, group_col={args.group_col}")

    if args.limit is not None:
        neg_ids = neg_ids[: min(args.limit, len(neg_ids))]
        print(f"[INFO] Limited to {args.limit} instances per iteration.")

    base_num_weights = get_numeric_cost_weights(config)
    base_cat_weights = get_categorical_step_weights(config)
    overrides = {}  # overrides[group] = {"num": {...}, "cat": {...}}

    out_dir = ART / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []

    for it in range(args.max_iters):
        print(f"\n========== Auto-FCAR Iteration {it} ==========")
        print(f"[CURRENT overrides] {overrides if overrides else '(none)'}")

        df = run_one_iteration(
            dataset_name=dataset_name,
            pipe=pipe,
            X_train=X_train,
            X_test=X_test,
            A_test=A_test,
            neg_ids=neg_ids,
            proba=proba,
            group_col=args.group_col,
            config=config,
            base_num_weights=base_num_weights,
            base_cat_weights=base_cat_weights,
            overrides=overrides,
            slack_penalty=args.slack_penalty,
            seed=args.seed + it,
        )

        csv_path = out_dir / f"{dataset_name}_autofcar_{args.group_col}_it{it}.csv"
        df.to_csv(csv_path, index=False)

        feasible_rate = (df["slack"] == 0.0).mean()
        flip_rate = df["flipped"].mean()
        avg_burden = df["burden_total"].mean()
        elapsed = float(df.attrs.get("elapsed_sec", np.nan))

        print(f"[OK] saved: {csv_path}")
        print(f"[SUMMARY] feasible={feasible_rate:.3f} flip={flip_rate:.3f} avg_burden={avg_burden:.4f} elapsed={elapsed:.2f}s")
        
        # Calculate Social Burden safely
        try:
            sb_df = compute_social_burden(df, group_col=args.group_col, cost_col="burden_total", only_feasible=True)
            print("\n[SOCIAL BURDEN] (sorted val desc)")
            print(sb_df)
            
            group_counts = df[args.group_col].value_counts()
            disparity = compute_burden_disparity(sb_df, min_group_n=args.min_group_n, group_counts=group_counts)
            gap = disparity.get("gap", 0.0)
            ratio = disparity.get("ratio", 1.0)
            worst_group = disparity.get("worst_group", "")
            
            print(f"\n[DISPARITY] max-min gap = {gap:.4f}, ratio = {ratio:.3f} (worst={worst_group})")
            
        except Exception as e:
            print(f"[ERROR] Could not compute social burden: {e}")
            break

        history.append({"iter": it, "csv": str(csv_path), "feasible": feasible_rate, "avg_burden": avg_burden, "ratio": ratio})

        if worst_group == "" or pd.isna(gap):
            print("[WARN] No valid worst group found. Stopping.")
            break

        # Stop if ratio is close to 1 (e.g. ratio - 1 <= epsilon)
        if (ratio - 1.0) <= args.epsilon:
            print(f"[STOP] ratio - 1 ({ratio-1.0:.3f}) <= epsilon ({args.epsilon}). Done.")
            break

        # --- Asymmetric feature-level weight adjustment ---
        # Identify which features contribute most to the worst group's
        # EXCESS burden relative to the best group, then selectively lower
        # weights on those features. Uniform scaling doesn't change relative
        # costs, so the solver would pick the same path.
        best_group = disparity.get("best_group", "")
        feasible = df[df["slack"] == 0.0]
        worst_rows = feasible[feasible[args.group_col] == worst_group]
        best_rows = feasible[feasible[args.group_col] == best_group]
        
        mutable_num = get_mutable_numeric_cols(config)
        mutable_cat = get_mutable_categorical_cols(config)
        num_ranges = {c: max(float(X_train[c].max() - X_train[c].min()), 1e-9) for c in mutable_num}
        
        cur = overrides.get(worst_group, {"num": {}, "cat": {}})
        cur_num = cur.get("num", {})
        cur_cat = cur.get("cat", {})
        
        if len(worst_rows) > 0 and len(best_rows) > 0:
            # Compute per-feature burden contribution for worst vs best
            excess = {}
            for c in mutable_num:
                dc = f"delta_{c}"
                wc = f"w_{c}"
                if dc in df.columns and wc in df.columns:
                    r = num_ranges[c]
                    fb_worst = float((worst_rows[wc] * worst_rows[dc].abs() / r).mean())
                    fb_best = float((best_rows[wc] * best_rows[dc].abs() / r).mean())
                    excess[c] = max(0, fb_worst - fb_best)
            
            total_excess = sum(excess.values())
            
            if total_excess > 1e-9:
                overrides[worst_group] = {"num": {}, "cat": {}}
                print(f"[UPDATE weights for {worst_group}] (asymmetric feature-level)")
                
                for c in mutable_num:
                    w = float(cur_num.get(c, base_num_weights.get(c, 1.0)))
                    share = excess.get(c, 0) / total_excess
                    # Reduce more for features with higher excess share
                    mult = 1.0 - (0.5 * share)  # range: 0.5 to 1.0
                    new_w = max(w * mult, args.min_weight)
                    overrides[worst_group]["num"][c] = new_w
                    if share > 0.05:
                        print(f"  {c}: share={share:.2f}, w {w:.3f} -> {new_w:.3f}")
                
                for c in mutable_cat:
                    w = float(cur_cat.get(c, base_cat_weights.get(c, 0.25)))
                    overrides[worst_group]["cat"][c] = w
                    print(f"  {c}: {w:.3f} (unchanged)")
            else:
                print(f"[WARN] No feature-level excess found for {worst_group}. Stopping.")
                break
        else:
            print(f"[WARN] Not enough feasible rows for comparison. Stopping.")
            break

    print("\n========== Auto-FCAR Finished ==========")
    print("[FINAL overrides]")
    print(overrides if overrides else "(none)")
    print("\n[HISTORY]")
    for h in history:
        print(h)

if __name__ == "__main__":
    main()
