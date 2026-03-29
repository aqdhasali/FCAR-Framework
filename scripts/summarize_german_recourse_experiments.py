from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"

# Orders used in the MIP solver
CHECKING_ORDER = ["A11", "A12", "A13", "A14"]
SAVINGS_ORDER  = ["A61", "A62", "A63", "A64", "A65"]


def rank_delta(before, after, order):
    """Return steps moved up in order; returns NaN if unknown."""
    try:
        b = str(before); a = str(after)
        if b not in order or a not in order:
            return np.nan
        return order.index(a) - order.index(b)
    except Exception:
        return np.nan


def normalized_l1(df, cols, ranges):
    """Compute normalized L1 burden for given delta columns."""
    total = 0.0
    for c in cols:
        if c not in df.columns:
            continue
        r = ranges.get(c, None)
        if r is None or r <= 0:
            continue
        total += df[c].abs() / r
    return total


def summarize_numeric_only(path: Path):
    df = pd.read_csv(path)
    print(f"\n=== Numeric-only results: {path.name} ===")
    print("Rows:", len(df))
    print("Feasible rate:", (df["slack"] == 0.0).mean())
    print("Flip rate:", df["flipped"].mean())
    print("Avg p_before:", df["p_before"].mean(), "Avg p_after:", df["p_after"].mean())
    print("Avg delta_duration:", df["delta_duration"].mean())
    print("Avg delta_credit_amount:", df["delta_credit_amount"].mean())

    # Group stats with counts
    for g in ["sex", "age_bucket"]:
        if g in df.columns:
            grp = df.groupby(g).agg(
                n=("flipped", "size"),
                flip_rate=("flipped", "mean"),
                feasible_rate=("slack", lambda s: (s == 0.0).mean()),
                avg_abs_dur=("delta_duration", lambda x: x.abs().mean()),
                avg_abs_amt=("delta_credit_amount", lambda x: x.abs().mean()),
            )
            print(f"\n[GROUP] {g}")
            print(grp)


def summarize_mip(path: Path):
    df = pd.read_csv(path)
    print(f"\n=== MIP results: {path.name} ===")
    print("Rows:", len(df))
    print("Feasible rate:", (df["slack"] == 0.0).mean())
    print("Flip rate:", df["flipped"].mean())
    print("Avg p_before:", df["p_before"].mean(), "Avg p_after:", df["p_after"].mean())

    # Numeric deltas (from before/after)
    df["delta_duration"] = df["duration_after"] - df["duration_before"]
    df["delta_credit_amount"] = df["credit_amount_after"] - df["credit_amount_before"]

    print("Avg delta_duration:", df["delta_duration"].mean())
    print("Avg delta_credit_amount:", df["delta_credit_amount"].mean())

    # Categorical steps
    df["checking_steps"] = df.apply(lambda r: rank_delta(r["checking_status_before"], r["checking_status_after"], CHECKING_ORDER), axis=1)
    df["savings_steps"] = df.apply(lambda r: rank_delta(r["savings_status_before"], r["savings_status_after"], SAVINGS_ORDER), axis=1)

    print("Avg checking steps:", df["checking_steps"].mean())
    print("Avg savings steps:", df["savings_steps"].mean())

    # A simple “burden” score (you can refine later)
    # Use rough ranges (German typical):
    ranges = {
        "delta_duration": 72,        # duration range roughly 0–72 months
        "delta_credit_amount": 20000 # conservative range for scaling
    }
    df["burden_numeric"] = normalized_l1(df, ["delta_duration", "delta_credit_amount"], ranges)
    df["burden_total"] = df["burden_numeric"] + 0.25 * (df["checking_steps"].fillna(0) + df["savings_steps"].fillna(0))

    print("Avg burden_total:", df["burden_total"].mean())

    # Group stats with counts
    for g in ["sex", "age_bucket"]:
        if g in df.columns:
            grp = df.groupby(g).agg(
                n=("flipped", "size"),
                flip_rate=("flipped", "mean"),
                feasible_rate=("slack", lambda s: (s == 0.0).mean()),
                avg_burden=("burden_total", "mean"),
                avg_check_steps=("checking_steps", "mean"),
                avg_save_steps=("savings_steps", "mean"),
                avg_abs_dur=("delta_duration", lambda x: x.abs().mean()),
                avg_abs_amt=("delta_credit_amount", lambda x: x.abs().mean()),
            )
            print(f"\n[GROUP] {g}")
            print(grp)


if __name__ == "__main__":
    # Update these names if you change settings/filenames
    numeric_path = REPORTS / "german_recourse_results_d48_c0.80.csv"
    mip_path = REPORTS / "german_recourse_mip_d48_c0.80.csv"

    if numeric_path.exists():
        summarize_numeric_only(numeric_path)
    else:
        print("Missing:", numeric_path)

    if mip_path.exists():
        summarize_mip(mip_path)
    else:
        print("Missing:", mip_path)