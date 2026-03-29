import sys
from pathlib import Path
import json
import pandas as pd

# Make project root importable if needed (safe to keep)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "artifacts" / "reports"


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def df_to_markdown(df: pd.DataFrame) -> str:
    """No external deps (doesn't require tabulate)."""
    # Convert everything to string for stable output
    sdf = df.copy()
    for c in sdf.columns:
        sdf[c] = sdf[c].astype(str)

    cols = list(sdf.columns)
    rows = sdf.values.tolist()

    # column widths
    widths = [len(c) for c in cols]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def row_line(items):
        return "| " + " | ".join(str(items[i]).ljust(widths[i]) for i in range(len(items))) + " |"

    header = row_line(cols)
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = "\n".join(row_line(r) for r in rows)
    return "\n".join([header, sep, body])


def load_eval_files():
    files = sorted(REPORTS_DIR.glob("*_eval.json"))
    if not files:
        raise FileNotFoundError(f"No *_eval.json files found in: {REPORTS_DIR}")
    return files


def summarize():
    files = load_eval_files()
    rows = []

    # Track all fairness attributes seen across datasets so we can create consistent columns
    all_attrs = set()

    parsed = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        parsed.append((fp, data))
        fairness = data.get("fairness", {})
        for attr in fairness.keys():
            all_attrs.add(attr)

    all_attrs = sorted(all_attrs)

    for fp, data in parsed:
        dataset = data.get("dataset", fp.stem.replace("_eval", ""))
        quality = data.get("quality", {})
        fairness = data.get("fairness", {})

        row = {
            "dataset": dataset,
            "accuracy": quality.get("accuracy"),
            "f1": quality.get("f1"),
            "roc_auc": quality.get("roc_auc"),
            "pred_pos_rate": quality.get("pred_pos_rate"),
        }

        # Add per-attribute fairness gaps (dynamic columns)
        for attr in all_attrs:
            fa = fairness.get(attr, {})
            row[f"{attr} sel_gap"] = fa.get("gap_selection_rate")
            row[f"{attr} tpr_gap"] = fa.get("gap_tpr")
            row[f"{attr} fpr_gap"] = fa.get("gap_fpr")

        # Also compute overall max gaps (handy for reporting)
        sel_gaps = {attr: fairness.get(attr, {}).get("gap_selection_rate") for attr in fairness.keys()}
        tpr_gaps = {attr: fairness.get(attr, {}).get("gap_tpr") for attr in fairness.keys()}
        fpr_gaps = {attr: fairness.get(attr, {}).get("gap_fpr") for attr in fairness.keys()}

        def max_gap(d):
            d2 = {k: v for k, v in d.items() if v is not None}
            if not d2:
                return None, None
            kmax = max(d2, key=lambda k: d2[k])
            return d2[kmax], kmax

        row["max sel_gap"] , row["max sel_gap attr"]  = max_gap(sel_gaps)
        row["max tpr_gap"] , row["max tpr_gap attr"]  = max_gap(tpr_gaps)
        row["max fpr_gap"] , row["max fpr_gap attr"]  = max_gap(fpr_gaps)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Put core columns first
    core_cols = ["dataset", "accuracy", "f1", "roc_auc", "pred_pos_rate",
                 "max sel_gap", "max sel_gap attr", "max tpr_gap", "max tpr_gap attr", "max fpr_gap", "max fpr_gap attr"]
    other_cols = [c for c in df.columns if c not in core_cols]
    df = df[core_cols + sorted(other_cols)]

    # Round numeric columns for readability
    for c in df.columns:
        if c == "dataset" or c.endswith("attr"):
            continue
        # convert numeric-like columns safely; non-numeric becomes NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")


    out_csv = REPORTS_DIR / "baseline_summary.csv"
    df.to_csv(out_csv, index=False)

    # Markdown version with formatted numbers
    df_md = df.copy()
    for c in df_md.columns:
        if c == "dataset" or c.endswith("attr"):
            continue
        df_md[c] = df_md[c].map(lambda x: fmt(x, 4))

    out_md = REPORTS_DIR / "baseline_summary.md"
    out_md.write_text(df_to_markdown(df_md), encoding="utf-8")

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_md}\n")

    # Print a smaller “core” view in console
    core_view = df_md[["dataset", "accuracy", "f1", "roc_auc", "pred_pos_rate",
                       "max sel_gap", "max sel_gap attr"]]
    print(df_to_markdown(core_view))


if __name__ == "__main__":
    summarize()
