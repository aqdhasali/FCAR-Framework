"""
A/B Benchmarking: Unconstrained AR vs FCAR.

For each dataset × each sensitive attribute:
  1) Run unconstrained recourse (default weights, no group overrides)
  2) Run FCAR recourse (Auto-FCAR tuned weights)
  3) Compute Social Burden (MISOB) per group for both
  4) Run Wilcoxon signed-rank test on per-individual burden differences
  5) Output comparison CSV + JSON summary

Usage:
  python scripts/benchmark_ab.py --dataset german --group_col age_bucket --limit 50
  python scripts/benchmark_ab.py --dataset adult --group_col sex --limit 50
"""

import sys
from pathlib import Path
import time
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.recourse.generic_recourse_mip import solve_recourse_mip
from src.config.config_loader import (
    load_dataset_config,
    get_mutable_numeric_cols,
    get_mutable_categorical_cols,
    get_numeric_cost_weights,
    get_categorical_step_weights,
    get_sensitive_attributes,
    get_solver_settings,
)
from src.metrics.social_burden import (
    compute_recourse_burden,
    compute_social_burden,
    compute_burden_disparity,
    compute_audit_score,
)

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

    # Add bucketed age columns so group_col="age_bucket" / "AGE_bucket" work
    if "age" in A_test.columns:
        A_test["age_bucket"] = pd.cut(
            A_test["age"], bins=[0, 25, 40, 60, 120],
            labels=["<=25", "26-40", "41-60", "60+"], include_lowest=True
        ).astype(str)
    if "AGE" in A_test.columns:
        A_test["AGE_bucket"] = pd.cut(
            A_test["AGE"], bins=[0, 25, 40, 60, 120],
            labels=["<=25", "26-40", "41-60", "60+"], include_lowest=True
        ).astype(str)

    return X_train, X_test, y_test, A_test


def run_recourse_batch(
    pipe, X_train, X_test, A_test, neg_ids, proba, config, group_col,
    weight_overrides=None, tag="baseline", limit=None, seed=42,
):
    """
    Run recourse on neg_ids using the generic MIP solver.
    
    weight_overrides: dict of {group_val: {"num": {...}, "cat": {...}}}
        If None, uses default config weights (unconstrained AR).
        If provided, applies group-specific weights (FCAR).
    """
    rng = np.random.default_rng(seed)
    ids = neg_ids.copy()
    rng.shuffle(ids)
    if limit is not None:
        ids = ids[:min(limit, len(ids))]

    mutable_num_cols = get_mutable_numeric_cols(config)
    mutable_cat_cols = get_mutable_categorical_cols(config)
    base_num_weights = get_numeric_cost_weights(config)
    base_cat_weights = get_categorical_step_weights(config)
    target_cls = int(config.get("label_positive", 1))

    if weight_overrides is None:
        weight_overrides = {}

    rows = []
    t0 = time.time()

    for k, i in enumerate(ids, start=1):
        x0 = X_test.iloc[i]
        p0 = float(proba[i])
        group_val = str(A_test.loc[i, group_col]) if group_col in A_test.columns else ""

        # Build weights
        num_w = dict(base_num_weights)
        cat_w = dict(base_cat_weights)
        if group_val in weight_overrides:
            num_w.update({k: v for k, v in weight_overrides[group_val].get("num", {}).items() if k in num_w})
            cat_w.update({k: v for k, v in weight_overrides[group_val].get("cat", {}).items() if k in cat_w})

        inst_config = dict(config)
        for col in num_w:
            if col in inst_config.get("mutable_numeric", {}):
                inst_config["mutable_numeric"][col]["cost_weight"] = num_w[col]
        for col in cat_w:
            if col in inst_config.get("mutable_categorical", {}):
                inst_config["mutable_categorical"][col]["step_weight"] = cat_w[col]

        x_cf, slack = solve_recourse_mip(pipe, X_train, x0, inst_config)

        p1 = float(pipe.predict_proba(pd.DataFrame([x_cf]))[:, 1][0])
        if target_cls == 0:
            flipped = int((p0 >= 0.5) and (p1 < 0.5) and (slack == 0.0))
        else:
            flipped = int((p0 < 0.5) and (p1 >= 0.5) and (slack == 0.0))

        row = {
            "method": tag,
            "test_idx": int(i),
            group_col: group_val,
            "p_before": p0,
            "p_after": p1,
            "slack": float(slack),
            "flipped": flipped,
        }
        for c in mutable_num_cols:
            val0 = float(x0.get(c, np.nan))
            val1 = float(x_cf.get(c, np.nan))
            row[f"delta_{c}"] = (val1 - val0) if np.isfinite(val0) and np.isfinite(val1) else np.nan
            row[f"w_{c}"] = float(num_w.get(c, 1.0))

        for col in mutable_cat_cols:
            row[f"{col}_before"] = str(x0.get(col, ""))
            row[f"{col}_after"] = str(x_cf.get(col, ""))
            row[f"w_{col}_step"] = float(cat_w.get(col, 0.25))

        rows.append(row)
        if k % 10 == 0 or k == len(ids):
            print(f"  [{tag}] {k}/{len(ids)} done")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    # Compute burden using the per-individual cost weights from the solver.
    # Each row has w_{feature} columns that reflect the FCAR-adjusted weights
    # for that individual's group. This ensures burden measurement aligns
    # exactly with the solver objective: burden_i = sum( w_i * |delta_i| / range_i )
    num_ranges = {c: max(float(X_train[c].max() - X_train[c].min()), 1e-9) for c in mutable_num_cols}
    burden = pd.Series(0.0, index=df.index)
    for c in mutable_num_cols:
        dc = f"delta_{c}"
        wc = f"w_{c}"
        r = num_ranges[c]
        if dc in df.columns and wc in df.columns:
            burden += df[wc] * df[dc].abs() / r
    df["burden_total"] = burden
    df.attrs["elapsed_sec"] = elapsed
    return df


def run_statistical_tests(ar_df, fcar_df, group_col):
    """
    Perform paired statistical tests on burden differences.
    
    Returns dict with test results.
    """
    results = {}

    # Align by test_idx
    merged = ar_df[["test_idx", "burden_total"]].merge(
        fcar_df[["test_idx", "burden_total"]],
        on="test_idx", suffixes=("_ar", "_fcar"),
    )

    if len(merged) < 3:
        return {"error": "Too few paired observations for statistical testing."}

    # 1) Paired Wilcoxon on overall burden
    diff = merged["burden_total_ar"] - merged["burden_total_fcar"]
    try:
        stat, p_val = stats.wilcoxon(diff, alternative="two-sided")
        results["overall_wilcoxon"] = {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "significant_at_005": bool(p_val < 0.05),
            "n_pairs": int(len(merged)),
            "mean_diff": round(float(diff.mean()), 6),
            "median_diff": round(float(diff.median()), 6),
        }
    except Exception as e:
        results["overall_wilcoxon"] = {"error": str(e)}

    # 2) Per-group disparity comparison
    ar_with_group = ar_df[["test_idx", "burden_total", group_col]].copy()
    fcar_with_group = fcar_df[["test_idx", "burden_total", group_col]].copy()

    ar_group_burden = ar_with_group.groupby(group_col)["burden_total"].mean()
    fcar_group_burden = fcar_with_group.groupby(group_col)["burden_total"].mean()

    ar_gap = float(ar_group_burden.max() - ar_group_burden.min()) if len(ar_group_burden) > 1 else 0.0
    fcar_gap = float(fcar_group_burden.max() - fcar_group_burden.min()) if len(fcar_group_burden) > 1 else 0.0

    results["disparity_reduction"] = {
        "ar_burden_gap": round(ar_gap, 6),
        "fcar_burden_gap": round(fcar_gap, 6),
        "gap_reduction": round(ar_gap - fcar_gap, 6),
        "gap_reduction_pct": round(100.0 * (ar_gap - fcar_gap) / ar_gap, 2) if ar_gap > 1e-9 else 0.0,
    }

    # 3) Mann-Whitney U per group
    results["per_group"] = {}
    common_groups = set(ar_group_burden.index) & set(fcar_group_burden.index)
    for g in sorted(common_groups):
        ar_g = ar_df[ar_df[group_col] == g]["burden_total"]
        fcar_g = fcar_df[fcar_df[group_col] == g]["burden_total"]
        if len(ar_g) >= 2 and len(fcar_g) >= 2:
            try:
                u_stat, u_p = stats.mannwhitneyu(ar_g, fcar_g, alternative="two-sided")
                results["per_group"][str(g)] = {
                    "ar_mean_burden": round(float(ar_g.mean()), 6),
                    "fcar_mean_burden": round(float(fcar_g.mean()), 6),
                    "change_pct": round(100.0 * (float(fcar_g.mean()) - float(ar_g.mean())) / float(ar_g.mean()), 2) if float(ar_g.mean()) > 1e-9 else 0.0,
                    "mann_whitney_U": round(float(u_stat), 4),
                    "p_value": round(float(u_p), 6),
                    "n_ar": int(len(ar_g)),
                    "n_fcar": int(len(fcar_g)),
                }
            except Exception as e:
                results["per_group"][str(g)] = {"error": str(e)}

    return results


def main(args):
    dataset_name = args.dataset
    group_col = args.group_col
    config = load_dataset_config(dataset_name)

    model_path = ART / "models" / f"{dataset_name}_logreg.joblib"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    pipe = joblib.load(model_path)
    X_train, X_test, y_test, A_test = load_test(dataset_name)

    target_cls = int(config.get("label_positive", 1))
    proba = pipe.predict_proba(X_test)[:, 1]
    if target_cls == 0:
        neg_ids = np.where(proba >= 0.5)[0]
    else:
        neg_ids = np.where(proba < 0.5)[0]

    print(f"[INFO] Dataset: {dataset_name} | Group: {group_col}")
    print(f"[INFO] Test size: {len(X_test)} | Rejected: {len(neg_ids)}")

    # ========== ARM A: Unconstrained AR ==========
    print("\n===== ARM A: Unconstrained AR =====")
    ar_df = run_recourse_batch(
        pipe, X_train, X_test, A_test, neg_ids, proba, config, group_col,
        weight_overrides=None, tag="unconstrained_ar",
        limit=args.limit, seed=args.seed,
    )

    # ========== ARM B: FCAR (with tuned weights) ==========
    print("\n===== ARM B: FCAR =====")
    # Simple heuristic FCAR overrides: reduce weights for all groups proportionally
    # In a real run, these would come from auto_fcar_tune.py output
    # For the benchmark, we run a quick 2-iteration auto-tune inline
    fcar_overrides = _quick_auto_tune(
        pipe, X_train, X_test, A_test, neg_ids, proba, config, group_col,
        limit=args.limit, seed=args.seed, max_iters=args.fcar_iters,
    )

    fcar_df = run_recourse_batch(
        pipe, X_train, X_test, A_test, neg_ids, proba, config, group_col,
        weight_overrides=fcar_overrides, tag="fcar",
        limit=args.limit, seed=args.seed,
    )

    # ========== Metrics ==========
    print("\n===== METRICS =====")

    ar_sb = compute_social_burden(ar_df, group_col=group_col, cost_col="burden_total", only_feasible=True)
    fcar_sb = compute_social_burden(fcar_df, group_col=group_col, cost_col="burden_total", only_feasible=True)

    ar_disp = compute_burden_disparity(ar_sb)
    fcar_disp = compute_burden_disparity(fcar_sb)

    ar_audit = compute_audit_score(ar_sb, epsilon=0.10)
    fcar_audit = compute_audit_score(fcar_sb, epsilon=0.10)

    print("\n[Unconstrained AR - Social Burden]")
    print(ar_sb)
    print(f"  Disparity gap: {ar_disp['gap']:.6f} | Audit score: {ar_audit['audit_score']}")

    print("\n[FCAR - Social Burden]")
    print(fcar_sb)
    print(f"  Disparity gap: {fcar_disp['gap']:.6f} | Audit score: {fcar_audit['audit_score']}")

    # ========== Statistical Tests ==========
    print("\n===== STATISTICAL TESTS =====")
    stat_results = run_statistical_tests(ar_df, fcar_df, group_col)

    if "overall_wilcoxon" in stat_results:
        w = stat_results["overall_wilcoxon"]
        if "error" not in w:
            print(f"  Wilcoxon: stat={w['statistic']}, p={w['p_value']}, sig={w['significant_at_005']}")
            print(f"  Mean burden diff (AR - FCAR): {w['mean_diff']:.6f}")

    if "disparity_reduction" in stat_results:
        dr = stat_results["disparity_reduction"]
        print(f"  AR gap: {dr['ar_burden_gap']:.6f} | FCAR gap: {dr['fcar_burden_gap']:.6f}")
        print(f"  Gap reduction: {dr['gap_reduction']:.6f} ({dr['gap_reduction_pct']:.1f}%)")

    # ========== Save ==========
    out_dir = ART / "reports" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([ar_df, fcar_df], ignore_index=True)
    csv_path = out_dir / f"{dataset_name}_{group_col}_ab_comparison.csv"
    combined.to_csv(csv_path, index=False)

    summary = {
        "dataset": dataset_name,
        "group_col": group_col,
        "n_rejected": int(len(neg_ids)),
        "n_evaluated": int(len(ar_df)),
        "fcar_overrides": _serialize_overrides(fcar_overrides),
        "unconstrained_ar": {
            "feasible_rate": round(float((ar_df["slack"] == 0).mean()), 4),
            "flip_rate": round(float(ar_df["flipped"].mean()), 4),
            "avg_burden": round(float(ar_df["burden_total"].mean()), 6),
            "social_burden": {str(k): round(float(v), 6) for k, v in ar_sb["social_burden"].items()},
            "disparity": ar_disp,
            "audit": ar_audit,
        },
        "fcar": {
            "feasible_rate": round(float((fcar_df["slack"] == 0).mean()), 4),
            "flip_rate": round(float(fcar_df["flipped"].mean()), 4),
            "avg_burden": round(float(fcar_df["burden_total"].mean()), 6),
            "social_burden": {str(k): round(float(v), 6) for k, v in fcar_sb["social_burden"].items()},
            "disparity": fcar_disp,
            "audit": fcar_audit,
        },
        "statistical_tests": stat_results,
    }

    json_path = out_dir / f"{dataset_name}_{group_col}_ab_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[OK] Saved CSV: {csv_path}")
    print(f"[OK] Saved JSON: {json_path}")


def _quick_auto_tune(pipe, X_train, X_test, A_test, neg_ids, proba, config, group_col,
                     limit=None, seed=42, max_iters=2):
    """
    Inline FCAR auto-tune: derive per-group weight overrides that reduce
    Social Burden disparity across sensitive groups.
    
    Strategy (asymmetric feature-level adjustment):
    1. Run recourse with current weights, compute per-group burden
    2. Find the worst (highest burden) and best (lowest burden) groups
    3. For the worst group, identify which features contribute most to
       their *excess* burden relative to the best group
    4. Selectively lower weights on those high-contribution features
       for the worst group only
    
    This ensures the solver actually picks different paths (uniform
    scaling doesn't change relative feature costs, so the solver
    would pick the same path regardless).
    
    Returns the overrides dict.
    """
    from src.metrics.social_burden import compute_burden_disparity

    base_num = get_numeric_cost_weights(config)
    base_cat = get_categorical_step_weights(config)
    mutable_num_cols = get_mutable_numeric_cols(config)
    mutable_cat_cols = get_mutable_categorical_cols(config)
    overrides = {}

    best_overrides = {}
    best_ratio = np.inf
    
    num_ranges = {c: max(float(X_train[c].max() - X_train[c].min()), 1e-9) for c in mutable_num_cols}

    for it in range(max_iters):
        print(f"  [auto-tune iter {it}]")

        df = run_recourse_batch(
            pipe, X_train, X_test, A_test, neg_ids, proba, config, group_col,
            weight_overrides=overrides, tag=f"tune_it{it}",
            limit=limit, seed=seed,
        )

        sb = compute_social_burden(df, group_col=group_col, cost_col="burden_total", only_feasible=True)
        disp = compute_burden_disparity(sb)

        worst = disp.get("worst_group", "")
        best_g = disp.get("best_group", "")
        if not worst or pd.isna(disp.get("gap", np.nan)):
            break

        ratio = disp.get("ratio", 1.0)
        print(f"  [auto-tune] worst={worst}, best={best_g}, ratio={ratio:.3f}")

        if ratio < best_ratio:
            best_ratio = ratio
            best_overrides = {g: {"num": dict(d.get("num", {})), "cat": dict(d.get("cat", {}))} for g, d in overrides.items()}

        if (ratio - 1.0) <= 0.10:
            print(f"  [auto-tune] ratio {ratio:.3f} <= 1.10, converged.")
            break

        # --- Asymmetric feature-level adjustment ---
        # Compute per-feature contribution to burden for worst vs best group
        feasible = df[df["slack"] == 0.0]
        worst_rows = feasible[feasible[group_col] == worst]
        best_rows = feasible[feasible[group_col] == best_g]
        
        if len(worst_rows) == 0 or len(best_rows) == 0:
            break
        
        # Per-feature mean |delta|/range for worst and best groups
        feature_burden_worst = {}
        feature_burden_best = {}
        for c in mutable_num_cols:
            dc = f"delta_{c}"
            if dc in df.columns:
                w = float(base_num.get(c, 1.0))
                r = num_ranges.get(c, 1.0)
                feature_burden_worst[c] = w * float(worst_rows[dc].abs().mean()) / r
                feature_burden_best[c] = w * float(best_rows[dc].abs().mean()) / r
        
        # Excess burden per feature for worst group
        excess = {c: feature_burden_worst.get(c, 0) - feature_burden_best.get(c, 0)
                  for c in mutable_num_cols}
        total_excess = sum(max(0, v) for v in excess.values())
        
        if total_excess < 1e-9:
            print(f"  [auto-tune] No feature-level excess found. Stopping.")
            break
        
        # Reduce weights proportionally to each feature's contribution to excess
        cur = overrides.get(worst, {"num": {}, "cat": {}})
        new_num = {}
        for c in mutable_num_cols:
            w = float(cur.get("num", {}).get(c, base_num.get(c, 1.0)))
            # Feature's share of the excess burden (0-1)
            share = max(0, excess.get(c, 0)) / total_excess
            # Reduce more aggressively for features with higher excess share
            mult = 1.0 - (0.5 * share)  # 0.5 to 1.0 range
            new_num[c] = max(w * mult, 0.01)
            if share > 0.05:
                print(f"    {c}: share={share:.2f}, w {w:.3f} -> {new_num[c]:.3f}")
        
        new_cat = {}
        for c in mutable_cat_cols:
            w = float(cur.get("cat", {}).get(c, base_cat.get(c, 0.25)))
            new_cat[c] = w  # keep cat weights stable unless they contribute to excess
        
        overrides[worst] = {"num": new_num, "cat": new_cat}

    return best_overrides if best_overrides else overrides


def _serialize_overrides(overrides):
    """Make overrides JSON-serializable."""
    out = {}
    for g, d in overrides.items():
        out[str(g)] = {
            "num": {str(k): round(float(v), 4) for k, v in d.get("num", {}).items()},
            "cat": {str(k): round(float(v), 4) for k, v in d.get("cat", {}).items()},
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A/B Benchmark: Unconstrained AR vs FCAR")
    ap.add_argument("--dataset", type=str, required=True, choices=["german", "adult", "default_credit"])
    ap.add_argument("--group_col", type=str, required=True, help="Sensitive attribute column name")
    ap.add_argument("--limit", type=int, default=None, help="Max instances per arm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fcar_iters", type=int, default=2, help="Number of auto-tune iterations for FCAR arm")

    args = ap.parse_args()
    main(args)
