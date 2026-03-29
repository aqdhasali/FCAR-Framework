from pathlib import Path
import pandas as pd
import numpy as np

CHECKING_ORDER = ["A11", "A12", "A13", "A14"]
SAVINGS_ORDER  = ["A61", "A62", "A63", "A64", "A65"]

def rank_delta(before, after, order):
    b = str(before); a = str(after)
    if b not in order or a not in order:
        return np.nan
    return order.index(a) - order.index(b)

def add_steps(df):
    df = df.copy()
    df["checking_steps"] = df.apply(lambda r: rank_delta(r["checking_status_before"], r["checking_status_after"], CHECKING_ORDER), axis=1)
    df["savings_steps"]  = df.apply(lambda r: rank_delta(r["savings_status_before"], r["savings_status_after"], SAVINGS_ORDER), axis=1)
    return df

def summarize(df, group_col="age_bucket"):
    df = add_steps(df)
    out = {}
    out["n"] = len(df)
    out["feasible_rate"] = (df["slack"] == 0.0).mean()
    out["flip_rate"] = df["flipped"].mean()
    out["avg_abs_amt"] = df["delta_credit_amount"].abs().mean()
    out["avg_abs_dur"] = df["delta_duration"].abs().mean()
    out["avg_check_steps"] = df["checking_steps"].mean()
    out["avg_save_steps"] = df["savings_steps"].mean()
    out["avg_p_after"] = df["p_after"].mean()

    grp = df.groupby(group_col).agg(
        n=("flipped", "size"),
        avg_abs_amt=("delta_credit_amount", lambda x: x.abs().mean()),
        avg_abs_dur=("delta_duration", lambda x: x.abs().mean()),
        avg_check_steps=("checking_steps", "mean"),
        avg_save_steps=("savings_steps", "mean"),
        avg_p_after=("p_after", "mean"),
    )
    return out, grp

def main(base_path, weighted_path):
    base = pd.read_csv(base_path)
    w = pd.read_csv(weighted_path)

    base_overall, base_grp = summarize(base)
    w_overall, w_grp = summarize(w)

    print("\n=== OVERALL ===")
    print("BASE:", base_overall)
    print("WTD :", w_overall)

    print("\n=== BY age_bucket (BASE) ===")
    print(base_grp)
    print("\n=== BY age_bucket (WTD) ===")
    print(w_grp)

    # Focus metric: avg_abs_amt for 41-60 if present
    if "41-60" in base_grp.index and "41-60" in w_grp.index:
        b = base_grp.loc["41-60", "avg_abs_amt"]
        a = w_grp.loc["41-60", "avg_abs_amt"]
        print(f"\n[FOCUS] 41-60 avg_abs_amt: BASE={b:.2f} -> WTD={a:.2f} (Δ={a-b:.2f})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to baseline CSV")
    ap.add_argument("--weighted", required=True, help="Path to weighted CSV")
    args = ap.parse_args()
    main(args.base, args.weighted)