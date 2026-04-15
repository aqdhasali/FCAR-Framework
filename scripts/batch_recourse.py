import sys
from pathlib import Path
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import copy
import joblib
import numpy as np
import pandas as pd

from src.recourse.generic_recourse_mip import solve_recourse_mip
from src.config.config_loader import (
    load_dataset_config,
    get_mutable_numeric_cols,
    get_mutable_categorical_cols,
    get_numeric_cost_weights,
    get_categorical_step_weights,
    get_sensitive_attributes,
)
from src.metrics.social_burden import compute_recourse_burden

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


def parse_kv_tokens(tokens):
    d = {}
    if not tokens:
        return d
    for t in tokens:
        t = t.strip().strip('"').strip("'")
        if "=" not in t:
            continue
        k, v = t.split("=", 1)
        k = k.strip().strip('"').strip("'")
        v = v.strip().strip('"').strip("'")
        try:
            d[k] = float(v)
        except ValueError:
            print(f"[WARN] Skipping invalid weight token: {t}")
    return d


def parse_group_overrides(tokens):
    """
    Tokens like:
      "41-60:credit_amount=2.0"
      "Female:education-num=1.2"
    """
    overrides = {}
    if not tokens:
        return overrides

    for t in tokens:
        t = t.strip().strip('"').strip("'")
        if ":" not in t or "=" not in t:
            print(f"[WARN] Skipping invalid group_override token: {t}")
            continue

        group, rest = t.split(":", 1)
        group = group.strip()

        key, val = rest.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")

        try:
            fval = float(val)
        except ValueError:
            print(f"[WARN] Skipping invalid group_override token: {t}")
            continue

        overrides.setdefault(group, {"num": {}, "cat": {}})

        # Hardcoded short-hand for backward compatibility on German dataset
        if key == "cat":
            overrides[group]["cat"]["checking_status"] = fval
            overrides[group]["cat"]["savings_status"] = fval
        else:
            # We don't strictly know if it's num or cat here, we'll put in both and let solver ignore
            overrides[group]["num"][key] = fval
            overrides[group]["cat"][key] = fval

    return overrides


def safe_tag(s: str) -> str:
    return (
        s.replace(" ", "")
         .replace("/", "-")
         .replace("\\", "-")
         .replace(":", "-")
         .replace("<", "lt")
         .replace(">", "gt")
         .replace("+", "plus")
    )


def main(args):
    dataset_name = args.dataset
    config = load_dataset_config(dataset_name)

    model_path = ART / "models" / f"{dataset_name}_logreg.joblib"
    if not model_path.exists():
        print(f"[ERROR] Trained model not found at {model_path}. Run train_baseline.py first.")
        return

    pipe = joblib.load(model_path)
    X_train, X_test, y_test, A_test = load_test(dataset_name)

    target_cls = int(config.get("label_positive", 1))
    proba = pipe.predict_proba(X_test)[:, 1]
    
    if target_cls == 0:
        pred = (proba < 0.5).astype(int)
        neg_ids = np.where(pred == 0)[0]  # pred == 0 means predicted Class 1 (Default)
        # to clarify: 0 = no-default (good outcome), 1 = default (bad outcome)
        # if pred=0 (bad outcome), proba (of class 1) is >= 0.5. 
        neg_ids = np.where(proba >= 0.5)[0]
    else:
        pred = (proba >= 0.5).astype(int)
        neg_ids = np.where(pred == 0)[0]

    print(f"[INFO] {dataset_name} test size: {len(X_test)}")
    print(f"[INFO] Rejected (negative outcome) count: {len(neg_ids)}")
    if len(neg_ids) == 0:
        return

    base_num_w = parse_kv_tokens(args.base_num_weight)
    base_cat_w = parse_kv_tokens(args.base_cat_weight)
    overrides = parse_group_overrides(args.group_override)

    if args.run_tag:
        run_tag = args.run_tag
    else:
        run_tag = "base" if not overrides else ("ovr_" + "_".join(sorted(overrides.keys())))
    run_tag = safe_tag(run_tag)

    print("\n[SETTINGS]")
    print(f"  dataset={dataset_name}")
    print(f"  run_tag={run_tag}")
    print(f"  fairness_attr={args.fairness_attr}")
    print(f"  base_num_weights={base_num_w if base_num_w else '(all default config)'}")
    print(f"  base_cat_weights={base_cat_w if base_cat_w else '(default config)'}")
    print(f"  overrides(groups)={list(overrides.keys()) if overrides else '(none)'}\n")

    rng = np.random.default_rng(args.seed)
    neg_ids = neg_ids.copy()
    rng.shuffle(neg_ids)

    if args.limit is not None:
        neg_ids = neg_ids[: min(args.limit, len(neg_ids))]
        print(f"[INFO] Using limit={args.limit} => running on {len(neg_ids)} rejected instances")

    mutable_num_cols = get_mutable_numeric_cols(config)
    mutable_cat_cols = get_mutable_categorical_cols(config)
    def_num_weights = get_numeric_cost_weights(config)
    def_cat_weights = get_categorical_step_weights(config)

    rows = []
    t0 = time.time()

    for k, i in enumerate(neg_ids, start=1):
        x0 = X_test.iloc[i]
        p0 = float(proba[i])

        group_val = ""
        if args.fairness_attr and args.fairness_attr in A_test.columns:
            group_val = str(A_test.loc[i, args.fairness_attr])

        # merge default config weights + CLI base weights + CLI group overrides
        num_w = dict(def_num_weights)
        cat_w = dict(def_cat_weights)
        
        num_w.update({k: v for k, v in base_num_w.items() if k in num_w})
        cat_w.update({k: v for k, v in base_cat_w.items() if k in cat_w})
        
        if group_val in overrides:
            num_w.update({k: v for k, v in overrides[group_val]["num"].items() if k in num_w})
            cat_w.update({k: v for k, v in overrides[group_val]["cat"].items() if k in cat_w})

        # Inject modified weights into config dictionary for the solver
        inst_config = copy.deepcopy(config)
        for col in num_w:
            inst_config["mutable_numeric"][col]["cost_weight"] = num_w[col]
        for col in cat_w:
            inst_config["mutable_categorical"][col]["step_weight"] = cat_w[col]
            
        if args.slack_penalty is not None:
            inst_config["solver"]["slack_penalty"] = args.slack_penalty

        x_cf, slack = solve_recourse_mip(pipe, X_train, x0, inst_config)

        p1 = float(pipe.predict_proba(pd.DataFrame([x_cf]))[:, 1][0])
        
        if target_cls == 0:
            flipped = int((p0 >= 0.5) and (p1 < 0.5) and (slack == 0.0))
        else:
            flipped = int((p0 < 0.5) and (p1 >= 0.5) and (slack == 0.0))

        row_data = {
            "dataset": dataset_name,
            "run_tag": run_tag,
            "test_idx": int(i),
            "p_before": p0,
            "p_after": p1,
            "slack": float(slack),
            "flipped": flipped,
            "fairness_attr": args.fairness_attr,
            "group_val": group_val,
        }
        
        for sa in get_sensitive_attributes(config):
            if sa in A_test.columns:
                row_data[sa] = str(A_test.loc[i, sa])

        for c in mutable_num_cols:
            val0 = float(x0.get(c, np.nan))
            val1 = float(x_cf.get(c, np.nan))
            row_data[f"{c}_before"] = val0
            row_data[f"{c}_after"] = val1
            row_data[f"delta_{c}"] = (val1 - val0) if np.isfinite(val0) and np.isfinite(val1) else np.nan
            row_data[f"w_{c}"] = float(num_w.get(c, 1.0))

        for col in mutable_cat_cols:
            row_data[f"{col}_before"] = str(x0.get(col, ""))
            row_data[f"{col}_after"] = str(x_cf.get(col, ""))
            # We don't save exact step counts here, but we save the category values and weights
            row_data[f"w_{col}_step"] = float(cat_w.get(col, 0.25))

        rows.append(row_data)

        if k % 10 == 0 or k == len(neg_ids):
            print(f"[PROGRESS] {k}/{len(neg_ids)} done")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    # Calculate burden score via social_burden metric utility
    num_ranges = {c: max(float(X_train[c].max() - X_train[c].min()), 1e-9) for c in mutable_num_cols}
    num_w_for_burden = {f"delta_{c}": float(def_num_weights.get(c, 1.0)) for c in mutable_num_cols}
    df["burden_total"] = compute_recourse_burden(
        df, 
        numeric_delta_cols=[f"delta_{c}" for c in mutable_num_cols],
        numeric_ranges={f"delta_{c}": num_ranges[c] for c in mutable_num_cols},
        numeric_weights=num_w_for_burden,
    )

    out_dir = ART / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.fairness_attr if args.fairness_attr else "none"
    out_path = out_dir / f"{dataset_name}_recourse_{tag}_{run_tag}.csv"
    df.to_csv(out_path, index=False)

    feasibility = (df["slack"] == 0.0).mean()
    flip_rate = df["flipped"].mean()

    print("\n[OK] Saved:", out_path)
    print(f"[SUMMARY] Dataset: {dataset_name}")
    print(f"[SUMMARY] Instances evaluated: {len(df)}")
    print(f"[SUMMARY] Feasible (slack==0) rate: {feasibility:.3f}")
    print(f"[SUMMARY] Flip success rate: {flip_rate:.3f}")
    print(f"[SUMMARY] Avg p_before: {df['p_before'].mean():.3f} | Avg p_after: {df['p_after'].mean():.3f}")
    print(f"[SUMMARY] Avg metric burden: {df['burden_total'].mean():.3f}")
    print(f"[SUMMARY] Total elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unified batch recourse script for FCAR.")
    ap.add_argument("--dataset", type=str, required=True, choices=["german", "adult", "default_credit"], help="Dataset to run recourse on")
    ap.add_argument("--limit", type=int, default=None, help="Max instances to process")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--slack_penalty", type=float, default=None, help="Override default slack penalty from config")

    ap.add_argument("--fairness_attr", type=str, default="", help="Sensitive attribute to divide groups by (e.g. sex, age_bucket, race)")

    ap.add_argument("--base_num_weight", nargs="*", default=[],
                    help='e.g. credit_amount=1.0 duration=1.0')
    ap.add_argument("--base_cat_weight", nargs="*", default=[],
                    help='e.g. checking_status=0.25 savings_status=0.25')

    ap.add_argument("--group_override", nargs="*", default=[],
                    help='e.g. "41-60:credit_amount=2.0" "Female:education-num=1.2"')

    ap.add_argument("--run_tag", type=str, default=None,
                    help="Optional tag to include in output filename.")

    args = ap.parse_args()
    main(args)
