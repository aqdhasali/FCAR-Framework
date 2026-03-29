import sys
from pathlib import Path
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.recourse.german_recourse_mip import solve_german_recourse_mip

PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ART = ROOT / "artifacts"


def add_age_bucket(age_series: pd.Series) -> pd.Series:
    # Safe labels
    return pd.cut(
        age_series.astype(float),
        bins=[0, 25, 40, 60, 120],
        labels=["le25", "26-40", "41-60", "60+"],
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
      "41-60:duration=1.2"
      "41-60:cat=0.15"   (sets BOTH checking_status and savings_status)
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

        if key == "cat":
            overrides[group]["cat"]["checking_status"] = fval
            overrides[group]["cat"]["savings_status"] = fval
        elif key in ("checking_status", "savings_status"):
            overrides[group]["cat"][key] = fval
        else:
            overrides[group]["num"][key] = fval

    return overrides


def safe_tag(s: str) -> str:
    # filesystem friendly
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
    pipe = joblib.load(ART / "models" / "german_logreg.joblib")
    X_train, X_test, y_test, A_test = load_german_test()

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    neg_ids = np.where(pred == 0)[0]

    print(f"[INFO] German test size: {len(X_test)}")
    print(f"[INFO] Rejected (pred=0) count: {len(neg_ids)}")
    if len(neg_ids) == 0:
        return

    base_num_w = parse_kv_tokens(args.base_num_weight)
    base_cat_w = parse_kv_tokens(args.base_cat_weight)
    overrides = parse_group_overrides(args.group_override)

    # Decide output run tag
    if args.run_tag:
        run_tag = args.run_tag
    else:
        run_tag = "base" if not overrides else ("ovr_" + "_".join(sorted(overrides.keys())))
    run_tag = safe_tag(run_tag)

    print("\n[SETTINGS]")
    print(f"  run_tag={run_tag}")
    print(f"  fairness_attr={args.fairness_attr}")
    print(f"  base_num_weights={base_num_w if base_num_w else '(all default 1.0)'}")
    print(f"  base_cat_weights={base_cat_w if base_cat_w else '(default 0.25 each)'}")
    print(f"  overrides(groups)={list(overrides.keys()) if overrides else '(none)'}")
    print(f"  duration_max_decrease={args.duration_max_decrease}")
    print(f"  credit_max_rel_decrease={args.credit_max_rel_decrease}")
    print(f"  mutable_cat_cols={args.mutable_cat_cols}")
    print(f"  slack_penalty={args.slack_penalty}\n")

    rng = np.random.default_rng(args.seed)
    neg_ids = neg_ids.copy()
    rng.shuffle(neg_ids)

    if args.limit is not None:
        neg_ids = neg_ids[: min(args.limit, len(neg_ids))]
        print(f"[INFO] Using limit={args.limit} => running on {len(neg_ids)} rejected instances")

    rows = []
    t0 = time.time()

    for k, i in enumerate(neg_ids, start=1):
        x0 = X_test.iloc[i]
        p0 = float(proba[i])

        group_val = ""
        if args.fairness_attr and args.fairness_attr in A_test.columns:
            group_val = str(A_test.loc[i, args.fairness_attr])

        # weights per-person
        num_w = dict(base_num_w)
        cat_w = {
            "checking_status": float(base_cat_w.get("checking_status", 0.25)),
            "savings_status": float(base_cat_w.get("savings_status", 0.25)),
        }
        if group_val in overrides:
            num_w.update(overrides[group_val]["num"])
            cat_w.update(overrides[group_val]["cat"])

        x_cf, slack = solve_german_recourse_mip(
            pipe=pipe,
            X_train=X_train,
            x0=x0,
            duration_max_decrease=args.duration_max_decrease,
            credit_max_rel_decrease=args.credit_max_rel_decrease,
            mutable_cat_cols=tuple(args.mutable_cat_cols),
            num_weights=num_w,
            cat_step_weights=cat_w,
            slack_penalty=args.slack_penalty,
        )

        p1 = float(pipe.predict_proba(pd.DataFrame([x_cf]))[:, 1][0])
        flipped = int((p0 < 0.5) and (p1 >= 0.5) and (slack == 0.0))

        sex = str(A_test.loc[i, "sex"]) if "sex" in A_test.columns else ""
        age_bucket = str(A_test.loc[i, "age_bucket"]) if "age_bucket" in A_test.columns else ""

        dur0 = float(x0.get("duration", np.nan))
        dur1 = float(x_cf.get("duration", np.nan))
        amt0 = float(x0.get("credit_amount", np.nan))
        amt1 = float(x_cf.get("credit_amount", np.nan))

        rows.append({
            "run_tag": run_tag,
            "test_idx": int(i),
            "p_before": p0,
            "p_after": p1,
            "slack": float(slack),
            "flipped": flipped,
            "sex": sex,
            "age_bucket": age_bucket,
            "fairness_attr": args.fairness_attr,
            "group_val": group_val,

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
        })

        if k % 10 == 0 or k == len(neg_ids):
            print(f"[PROGRESS] {k}/{len(neg_ids)} done")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    out_dir = ART / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.fairness_attr if args.fairness_attr else "none"
    out_path = out_dir / f"german_recourse_mip_weighted_{tag}_{run_tag}_d{args.duration_max_decrease}_c{args.credit_max_rel_decrease:.2f}.csv"
    df.to_csv(out_path, index=False)

    feasibility = (df["slack"] == 0.0).mean()
    flip_rate = df["flipped"].mean()

    print("\n[OK] Saved:", out_path)
    print(f"[SUMMARY] Instances evaluated: {len(df)}")
    print(f"[SUMMARY] Feasible (slack==0) rate: {feasibility:.3f}")
    print(f"[SUMMARY] Flip success rate: {flip_rate:.3f}")
    print(f"[SUMMARY] Avg p_before: {df['p_before'].mean():.3f} | Avg p_after: {df['p_after'].mean():.3f}")
    print(f"[SUMMARY] Avg |delta_credit_amount|: {df['delta_credit_amount'].abs().mean():.3f}")
    print(f"[SUMMARY] Avg |delta_duration|: {df['delta_duration'].abs().mean():.3f}")
    print(f"[SUMMARY] Total elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--duration_max_decrease", type=int, default=48)
    ap.add_argument("--credit_max_rel_decrease", type=float, default=0.80)

    ap.add_argument("--mutable_cat_cols", nargs="+", default=["checking_status", "savings_status"])
    ap.add_argument("--slack_penalty", type=float, default=1000.0)

    ap.add_argument("--fairness_attr", type=str, default="age_bucket", choices=["sex", "age_bucket", ""])

    ap.add_argument("--base_num_weight", nargs="*", default=[],
                    help='e.g. credit_amount=1.0 duration=1.0')
    ap.add_argument("--base_cat_weight", nargs="*", default=[],
                    help='e.g. checking_status=0.25 savings_status=0.25')

    ap.add_argument("--group_override", nargs="*", default=[],
                    help='e.g. "41-60:credit_amount=2.0" "41-60:duration=1.2" "41-60:cat=0.15"')

    ap.add_argument("--run_tag", type=str, default=None,
                    help="Optional tag to include in output filename (prevents overwriting).")

    args = ap.parse_args()
    main(args)