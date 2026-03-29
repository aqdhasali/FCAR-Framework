from pathlib import Path
import re
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"

CHECKING_ORDER = ["A11", "A12", "A13", "A14"]
SAVINGS_ORDER = ["A61", "A62", "A63", "A64", "A65"]


def rank_delta(before, after, order):
    b = str(before)
    a = str(after)
    if b not in order or a not in order:
        return np.nan
    return order.index(a) - order.index(b)


def compute_group_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    d = df.copy()

    # robust: prefer group_col, else fallback to group_val
    group_key = group_col if group_col in d.columns else "group_val"
    if group_key not in d.columns:
        raise KeyError(f"Missing grouping column: '{group_col}' (and no 'group_val' fallback).")

    d["abs_amt"] = d["delta_credit_amount"].abs()
    d["abs_dur"] = d["delta_duration"].abs()

    d["checking_steps"] = d.apply(
        lambda r: rank_delta(r.get("checking_status_before", ""), r.get("checking_status_after", ""), CHECKING_ORDER),
        axis=1,
    )
    d["savings_steps"] = d.apply(
        lambda r: rank_delta(r.get("savings_status_before", ""), r.get("savings_status_after", ""), SAVINGS_ORDER),
        axis=1,
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
    grp.index.name = group_col
    return grp.sort_values("avg_abs_amt", ascending=False)


def compute_iteration_summary(df: pd.DataFrame, group_col: str, min_group_n: int) -> dict:
    overall = {
        "feasible_rate": float((df["slack"] == 0.0).mean()),
        "flip_rate": float(df["flipped"].mean()),
        "avg_abs_amt": float(df["delta_credit_amount"].abs().mean()),
        "avg_abs_dur": float(df["delta_duration"].abs().mean()),
        "avg_p_after": float(df["p_after"].mean()),
    }

    grp = compute_group_table(df, group_col)
    eligible = grp[grp["n"] >= min_group_n].copy()

    if not eligible.empty:
        gap = float(eligible["avg_abs_amt"].max() - eligible["avg_abs_amt"].min())
        worst = str(eligible["avg_abs_amt"].idxmax())
        best = str(eligible["avg_abs_amt"].idxmin())
    else:
        gap, worst, best = np.nan, "", ""

    # Extract typical weights (they're constant per-group per iteration but stored per-row)
    # We'll report the mean used across all rows (fine for a summary).
    w_credit = float(df.get("w_credit_amount", pd.Series([np.nan])).mean())
    w_dur = float(df.get("w_duration", pd.Series([np.nan])).mean())
    w_check = float(df.get("w_check_step", pd.Series([np.nan])).mean())
    w_save = float(df.get("w_save_step", pd.Series([np.nan])).mean())

    return {
        **overall,
        "gap_avg_abs_amt_eligible": gap,
        "worst_group": worst,
        "best_group": best,
        "mean_w_credit_amount": w_credit,
        "mean_w_duration": w_dur,
        "mean_w_check_step": w_check,
        "mean_w_save_step": w_save,
    }


def find_iteration_files(group_col: str):
    # german_autofcar_age_bucket_it0_d48_c0.80.csv
    pat = re.compile(rf"^german_autofcar_{re.escape(group_col)}_it(\d+)_d(\d+)_c([0-9.]+)\.csv$")
    found = []
    for p in REPORTS.glob(f"german_autofcar_{group_col}_it*_d*_c*.csv"):
        m = pat.match(p.name)
        if m:
            it = int(m.group(1))
            d = int(m.group(2))
            c = float(m.group(3))
            found.append((it, d, c, p))
    found.sort(key=lambda x: x[0])
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group_col", type=str, default="age_bucket", choices=["age_bucket", "sex"])
    ap.add_argument("--min_group_n", type=int, default=5)
    ap.add_argument("--show_group_tables", action="store_true", help="Print per-iteration group tables.")
    args = ap.parse_args()

    files = find_iteration_files(args.group_col)
    if not files:
        print(f"[ERROR] No iteration files found for group_col={args.group_col} in {REPORTS}")
        return

    summaries = []
    last_d, last_c = None, None

    for it, d, c, path in files:
        df = pd.read_csv(path)

        last_d, last_c = d, c

        summ = compute_iteration_summary(df, args.group_col, args.min_group_n)
        summ.update({"iter": it, "file": path.name, "duration_max_decrease": d, "credit_max_rel_decrease": c})
        summaries.append(summ)

        print(f"\n=== Iteration {it} ===")
        print(f"File: {path.name}")
        print(
            f"feasible={summ['feasible_rate']:.3f} flip={summ['flip_rate']:.3f} "
            f"avg|Δamt|={summ['avg_abs_amt']:.2f} avg|Δdur|={summ['avg_abs_dur']:.2f} "
            f"gap(eligible)={summ['gap_avg_abs_amt_eligible']:.2f} "
            f"worst={summ['worst_group']} best={summ['best_group']}"
        )

        if args.show_group_tables:
            grp = compute_group_table(df, args.group_col)
            print("\n[GROUP TABLE]")
            print(grp)

    out_df = pd.DataFrame(summaries).sort_values("iter")
    out_path = REPORTS / f"german_autofcar_{args.group_col}_summary.csv"
    out_df.to_csv(out_path, index=False)

    print("\n==============================")
    print("[OK] Saved summary CSV:", out_path)
    print("==============================\n")
    print(out_df[[
        "iter",
        "feasible_rate",
        "flip_rate",
        "avg_abs_amt",
        "avg_abs_dur",
        "gap_avg_abs_amt_eligible",
        "worst_group",
        "best_group",
        "mean_w_credit_amount",
        "mean_w_duration",
        "mean_w_check_step",
        "mean_w_save_step",
        "file",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()