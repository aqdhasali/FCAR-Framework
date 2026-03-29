"""
Generate publication-quality benchmark charts from A/B comparison results.

Reads the JSON summaries produced by benchmark_ab.py and creates:
  1) Grouped bar chart: Social Burden per group (AR vs FCAR)
  2) Disparity reduction bar chart across datasets
  3) Performance trade-offs table (feasibility, flip rate)

Usage:
  python scripts/generate_benchmark_charts.py
  python scripts/generate_benchmark_charts.py --datasets german adult
"""

import sys
from pathlib import Path
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ART = ROOT / "artifacts"
BENCH_DIR = ART / "reports" / "benchmarks"
FIG_DIR = ART / "reports" / "figures"


def load_summaries(datasets=None):
    """Load all *_ab_summary.json files from benchmarks dir."""
    summaries = []
    if not BENCH_DIR.exists():
        print(f"[WARN] Benchmark dir not found: {BENCH_DIR}")
        return summaries

    for f in sorted(BENCH_DIR.glob("*_ab_summary.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if datasets and data.get("dataset") not in datasets:
            continue
        data["_path"] = str(f)
        summaries.append(data)

    return summaries


def plot_social_burden_comparison(summary, ax=None):
    """
    Grouped bar chart: Social Burden per group for AR vs FCAR.
    """
    ds = summary["dataset"]
    gc = summary["group_col"]

    ar_sb = summary["unconstrained_ar"]["social_burden"]
    fcar_sb = summary["fcar"]["social_burden"]

    groups = sorted(set(ar_sb.keys()) | set(fcar_sb.keys()))
    ar_vals = [float(ar_sb.get(g, 0)) for g in groups]
    fcar_vals = [float(fcar_sb.get(g, 0)) for g in groups]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(groups))
    width = 0.35

    bars_ar = ax.bar(x - width/2, ar_vals, width, label="Unconstrained AR",
                     color="#e74c3c", alpha=0.85, edgecolor="white", linewidth=0.8)
    bars_fcar = ax.bar(x + width/2, fcar_vals, width, label="FCAR",
                       color="#2ecc71", alpha=0.85, edgecolor="white", linewidth=0.8)

    ax.set_xlabel(gc.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_ylabel("Social Burden Score", fontsize=12, fontweight="bold")
    ax.set_title(f"Social Burden: AR vs FCAR — {ds.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add value labels
    for bar in bars_ar:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#c0392b")
    for bar in bars_fcar:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#27ae60")

    return ax


def plot_disparity_reduction(summaries, ax=None):
    """
    Bar chart showing disparity gap reduction across datasets.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    labels = []
    ar_gaps = []
    fcar_gaps = []

    for s in summaries:
        label = f"{s['dataset']}:{s['group_col']}"
        labels.append(label)
        ar_gaps.append(float(s["unconstrained_ar"]["disparity"]["gap"]))
        fcar_gaps.append(float(s["fcar"]["disparity"]["gap"]))

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, ar_gaps, width, label="AR Disparity Gap",
           color="#e74c3c", alpha=0.85, edgecolor="white")
    ax.bar(x + width/2, fcar_gaps, width, label="FCAR Disparity Gap",
           color="#2ecc71", alpha=0.85, edgecolor="white")

    ax.set_ylabel("Burden Disparity Gap", fontsize=12, fontweight="bold")
    ax.set_title("Disparity Gap Reduction: AR vs FCAR", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    return ax


def plot_performance_tradeoffs(summaries, ax=None):
    """
    Grouped bar chart comparing feasibility and flip rates.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    labels = []
    metrics = {
        "AR Feasible": [], "FCAR Feasible": [],
        "AR Flip": [], "FCAR Flip": [],
    }

    for s in summaries:
        label = f"{s['dataset']}:{s['group_col']}"
        labels.append(label)
        metrics["AR Feasible"].append(s["unconstrained_ar"]["feasible_rate"])
        metrics["FCAR Feasible"].append(s["fcar"]["feasible_rate"])
        metrics["AR Flip"].append(s["unconstrained_ar"]["flip_rate"])
        metrics["FCAR Flip"].append(s["fcar"]["flip_rate"])

    x = np.arange(len(labels))
    width = 0.2
    colors = ["#e74c3c", "#2ecc71", "#e67e22", "#3498db"]

    for idx, (metric_name, vals) in enumerate(metrics.items()):
        offset = (idx - 1.5) * width
        ax.bar(x + offset, vals, width, label=metric_name, color=colors[idx], alpha=0.85, edgecolor="white")

    ax.set_ylabel("Rate", fontsize=12, fontweight="bold")
    ax.set_title("Performance Trade-offs: AR vs FCAR (RQ3)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(fontsize=10, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.1)

    return ax


def generate_summary_table(summaries):
    """Generate a markdown summary table."""
    rows = []
    for s in summaries:
        ds = s["dataset"]
        gc = s["group_col"]
        ar = s["unconstrained_ar"]
        fc = s["fcar"]
        st = s.get("statistical_tests", {})
        dr = st.get("disparity_reduction", {})
        wil = st.get("overall_wilcoxon", {})

        rows.append({
            "Dataset": ds,
            "Group": gc,
            "N": s["n_evaluated"],
            "AR Feasible": f"{ar['feasible_rate']:.1%}",
            "FCAR Feasible": f"{fc['feasible_rate']:.1%}",
            "AR Flip": f"{ar['flip_rate']:.1%}",
            "FCAR Flip": f"{fc['flip_rate']:.1%}",
            "AR Gap": f"{ar['disparity']['gap']:.4f}",
            "FCAR Gap": f"{fc['disparity']['gap']:.4f}",
            "Gap Δ%": f"{dr.get('gap_reduction_pct', 0):.1f}%",
            "AR Audit": f"{ar['audit']['audit_score']:.2f}",
            "FCAR Audit": f"{fc['audit']['audit_score']:.2f}",
            "Wilcoxon p": f"{wil.get('p_value', 'N/A')}",
        })

    df = pd.DataFrame(rows)
    return df


def main(args):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(args.datasets)
    if not summaries:
        print("[ERROR] No benchmark summaries found. Run benchmark_ab.py first.")
        return

    print(f"[INFO] Found {len(summaries)} benchmark summaries")

    # 1) Per-dataset Social Burden charts
    for s in summaries:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_social_burden_comparison(s, ax)
        fig.tight_layout()
        fname = f"{s['dataset']}_{s['group_col']}_social_burden.png"
        fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved: {FIG_DIR / fname}")

    # 2) Cross-dataset disparity reduction
    if len(summaries) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_disparity_reduction(summaries, ax)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "disparity_reduction_all.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved: {FIG_DIR / 'disparity_reduction_all.png'}")

    # 3) Performance trade-off chart
    if len(summaries) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_performance_tradeoffs(summaries, ax)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "performance_tradeoffs_all.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved: {FIG_DIR / 'performance_tradeoffs_all.png'}")

    # 4) Summary table
    table_df = generate_summary_table(summaries)
    print("\n===== BENCHMARK SUMMARY TABLE =====")
    print(table_df.to_string(index=False))

    table_path = FIG_DIR / "benchmark_summary_table.csv"
    table_df.to_csv(table_path, index=False)
    print(f"\n[OK] Saved: {table_path}")

    # Markdown version
    md_path = FIG_DIR / "benchmark_summary_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# A/B Benchmark Summary: Unconstrained AR vs FCAR\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write("\n")
    print(f"[OK] Saved: {md_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None, help="Filter to these datasets only")
    args = ap.parse_args()
    main(args)
