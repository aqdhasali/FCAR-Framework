"""
Generate a comprehensive FCAR evaluation dashboard.

Creates publication-ready figures that combine:
  1) Model quality metrics across datasets (accuracy, F1, AUC)
  2) Pre-recourse fairness gaps (Demographic Parity, Equalized Odds)
  3) AR vs FCAR Social Burden per-group comparison (all datasets)
  4) Disparity gap reduction summary with % labels
  5) Statistical significance summary (Wilcoxon p-values)
  6) Burden ratio improvement (before/after FCAR)

Usage:
  python scripts/generate_full_evaluation.py
"""

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec

ART = ROOT / "artifacts"
REPORTS = ART / "reports"
BENCH_DIR = REPORTS / "benchmarks"
FIG_DIR = REPORTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_eval_jsons():
    evals = {}
    for f in sorted(REPORTS.glob("*_eval.json")):
        with open(f, "r") as fh:
            data = json.load(fh)
        evals[data["dataset"]] = data
    return evals


def load_benchmark_jsons():
    benchmarks = []
    for f in sorted(BENCH_DIR.glob("*_ab_summary.json")):
        with open(f, "r") as fh:
            data = json.load(fh)
        benchmarks.append(data)
    return benchmarks


# ─────────────────────────────────────────────────────────
# FIGURE 1: Model Quality
# ─────────────────────────────────────────────────────────
def fig_model_quality(evals):
    datasets = list(evals.keys())
    metrics = ["accuracy", "f1", "roc_auc", "pr_auc"]
    labels = ["Accuracy", "F1 Score", "ROC-AUC", "PR-AUC"]
    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    width = 0.18

    for i, (m, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [evals[ds]["quality"][m] for ds in datasets]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Baseline Model Quality Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace("_", " ").title() for ds in datasets], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = FIG_DIR / "eval_model_quality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────
# FIGURE 2: Pre-recourse Fairness Gaps
# ─────────────────────────────────────────────────────────
def fig_fairness_gaps(evals):
    records = []
    for ds, data in evals.items():
        for attr, m in data.get("fairness", {}).items():
            records.append({
                "Dataset": ds.replace("_", " ").title(),
                "Attribute": attr,
                "DPD": m.get("demographic_parity_diff", 0),
                "EOD": m.get("equalized_odds_diff", 0),
            })
    df = pd.DataFrame(records)
    df["Label"] = df["Dataset"] + "\n(" + df["Attribute"] + ")"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    w = 0.35

    bars1 = ax.bar(x - w/2, df["DPD"], w, label="Demographic Parity Diff",
                   color="#f39c12", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, df["EOD"], w, label="Equalized Odds Diff",
                   color="#8e44ad", alpha=0.85, edgecolor="white")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0.10, color="red", ls="--", lw=1.2, alpha=0.7, label="10% Threshold")
    ax.set_ylabel("Difference", fontsize=12, fontweight="bold")
    ax.set_title("Pre-Recourse Fairness Gaps (Baseline Model)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"].tolist(), fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = FIG_DIR / "eval_fairness_gaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────
# FIGURE 3: AR vs FCAR Social Burden (all datasets, 2x2)
# ─────────────────────────────────────────────────────────
def fig_social_burden_grid(benchmarks):
    n = len(benchmarks)
    cols = min(n, 2)
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    for idx, s in enumerate(benchmarks):
        ax = axes[idx // cols][idx % cols]
        ds = s["dataset"].replace("_", " ").title()
        gc = s["group_col"]

        ar_sb = s["unconstrained_ar"]["social_burden"]
        fc_sb = s["fcar"]["social_burden"]
        groups = sorted(set(ar_sb) | set(fc_sb))
        ar_v = [float(ar_sb.get(g, 0)) for g in groups]
        fc_v = [float(fc_sb.get(g, 0)) for g in groups]

        x = np.arange(len(groups))
        w = 0.35
        b1 = ax.bar(x - w/2, ar_v, w, label="Unconstrained AR", color="#e74c3c", alpha=0.85, edgecolor="white")
        b2 = ax.bar(x + w/2, fc_v, w, label="FCAR", color="#2ecc71", alpha=0.85, edgecolor="white")

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ar_gap = s["unconstrained_ar"]["disparity"]["gap"]
        fc_gap = s["fcar"]["disparity"]["gap"]
        pct = 100 * (ar_gap - fc_gap) / ar_gap if ar_gap > 1e-9 else 0
        ax.set_title(f"{ds} ({gc})\nGap: {ar_gap:.4f} → {fc_gap:.4f} (↓{pct:.0f}%)",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylabel("Social Burden")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Social Burden Comparison: Unconstrained AR vs FCAR", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = FIG_DIR / "eval_social_burden_all.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────
# FIGURE 4: Gap Reduction + Ratio Improvement
# ─────────────────────────────────────────────────────────
def fig_gap_reduction(benchmarks):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = []
    ar_gaps = []
    fc_gaps = []
    pcts = []
    ar_ratios = []
    fc_ratios = []

    for s in benchmarks:
        label = f"{s['dataset'].replace('_',' ').title()}\n({s['group_col']})"
        labels.append(label)
        ag = float(s["unconstrained_ar"]["disparity"]["gap"])
        fg = float(s["fcar"]["disparity"]["gap"])
        ar_gaps.append(ag)
        fc_gaps.append(fg)
        pcts.append(100 * (ag - fg) / ag if ag > 1e-9 else 0)
        ar_ratios.append(float(s["unconstrained_ar"]["disparity"]["ratio"]))
        fc_ratios.append(float(s["fcar"]["disparity"]["ratio"]))

    x = np.arange(len(labels))
    w = 0.35

    # Left: Gap reduction
    b1 = ax1.bar(x - w/2, ar_gaps, w, label="AR Gap", color="#e74c3c", alpha=0.85, edgecolor="white")
    b2 = ax1.bar(x + w/2, fc_gaps, w, label="FCAR Gap", color="#2ecc71", alpha=0.85, edgecolor="white")
    for i, pct in enumerate(pcts):
        mid = (ar_gaps[i] + fc_gaps[i]) / 2
        ax1.annotate(f"↓{pct:.0f}%", xy=(x[i], max(ar_gaps[i], fc_gaps[i]) + 0.005),
                     fontsize=11, fontweight="bold", ha="center", color="#27ae60")
    ax1.set_ylabel("Burden Disparity Gap", fontsize=12, fontweight="bold")
    ax1.set_title("Disparity Gap Reduction", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # Right: Ratio improvement
    b3 = ax2.bar(x - w/2, ar_ratios, w, label="AR Ratio (max/min)", color="#e74c3c", alpha=0.85, edgecolor="white")
    b4 = ax2.bar(x + w/2, fc_ratios, w, label="FCAR Ratio", color="#2ecc71", alpha=0.85, edgecolor="white")
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                 f"{h:.2f}×", ha="center", va="bottom", fontsize=8)
    ax2.axhline(1.0, color="green", ls="--", lw=1.2, alpha=0.6, label="Perfect Parity (1.0×)")
    ax2.set_ylabel("Burden Ratio (max / min)", fontsize=12, fontweight="bold")
    ax2.set_title("Burden Ratio Improvement", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    fig.suptitle("FCAR Effectiveness: Disparity Reduction (RQ2)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = FIG_DIR / "eval_gap_reduction.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────
# FIGURE 5: Statistical Significance + Audit Scores
# ─────────────────────────────────────────────────────────
def fig_statistical_summary(benchmarks):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = []
    p_vals = []
    mean_diffs = []
    ar_audits = []
    fc_audits = []

    for s in benchmarks:
        label = f"{s['dataset'].replace('_',' ').title()}\n({s['group_col']})"
        labels.append(label)
        wil = s.get("statistical_tests", {}).get("overall_wilcoxon", {})
        p = wil.get("p_value", 1.0)
        if p != p:  # NaN
            p = 1.0
        p_vals.append(p)
        mean_diffs.append(wil.get("mean_diff", 0))
        ar_audits.append(s["unconstrained_ar"]["audit"]["audit_score"])
        fc_audits.append(s["fcar"]["audit"]["audit_score"])

    x = np.arange(len(labels))

    # Left: P-values (log scale)
    colors = ["#2ecc71" if p < 0.05 else "#e74c3c" for p in p_vals]
    bars = ax1.bar(x, [-np.log10(max(p, 1e-20)) for p in p_vals], 0.5,
                   color=colors, alpha=0.85, edgecolor="white")
    ax1.axhline(-np.log10(0.05), color="red", ls="--", lw=1.2, alpha=0.7,
                label="p = 0.05 threshold")
    for i, (bar, p) in enumerate(zip(bars, p_vals)):
        h = bar.get_height()
        label_text = f"p < 0.001" if p < 0.001 else f"p = {p:.4f}"
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                 label_text, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("-log₁₀(p-value)", fontsize=12, fontweight="bold")
    ax1.set_title("Wilcoxon Signed-Rank Test\n(Higher = More Significant)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # Right: Audit scores
    w = 0.35
    ax2.bar(x - w/2, ar_audits, w, label="AR Audit Score", color="#e74c3c", alpha=0.85, edgecolor="white")
    ax2.bar(x + w/2, fc_audits, w, label="FCAR Audit Score", color="#2ecc71", alpha=0.85, edgecolor="white")
    for i in range(len(labels)):
        for val, offset, color in [(ar_audits[i], -w/2, "#c0392b"), (fc_audits[i], w/2, "#27ae60")]:
            ax2.text(x[i] + offset, val + 0.02, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)
    ax2.axhline(0.5, color="orange", ls="--", lw=1, alpha=0.6, label="Passing threshold")
    ax2.set_ylabel("Audit Score (0-1)", fontsize=12, fontweight="bold")
    ax2.set_title("MISOB Audit Score\n(Higher = Fairer)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    fig.suptitle("Statistical Validation & Audit Results (RQ2 / RQ3)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIG_DIR / "eval_statistical_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────
# FIGURE 6: Combined performance dashboard
# ─────────────────────────────────────────────────────────
def fig_performance_dashboard(benchmarks):
    n = len(benchmarks)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = [f"{s['dataset'].replace('_',' ').title()}\n({s['group_col']})" for s in benchmarks]
    x = np.arange(n)
    w = 0.35

    # Panel 1: Feasibility
    ar_f = [s["unconstrained_ar"]["feasible_rate"] for s in benchmarks]
    fc_f = [s["fcar"]["feasible_rate"] for s in benchmarks]
    axes[0].bar(x - w/2, ar_f, w, label="AR", color="#e74c3c", alpha=0.85, edgecolor="white")
    axes[0].bar(x + w/2, fc_f, w, label="FCAR", color="#2ecc71", alpha=0.85, edgecolor="white")
    axes[0].set_title("Feasibility Rate", fontsize=12, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].set_ylim(0, 1.15)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    # Panel 2: Flip Rate
    ar_fl = [s["unconstrained_ar"]["flip_rate"] for s in benchmarks]
    fc_fl = [s["fcar"]["flip_rate"] for s in benchmarks]
    axes[1].bar(x - w/2, ar_fl, w, label="AR", color="#e74c3c", alpha=0.85, edgecolor="white")
    axes[1].bar(x + w/2, fc_fl, w, label="FCAR", color="#2ecc71", alpha=0.85, edgecolor="white")
    axes[1].set_title("Decision Flip Rate", fontsize=12, fontweight="bold")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].set_ylim(0, 1.15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")

    # Panel 3: Average Burden
    ar_b = [s["unconstrained_ar"]["avg_burden"] for s in benchmarks]
    fc_b = [s["fcar"]["avg_burden"] for s in benchmarks]
    axes[2].bar(x - w/2, ar_b, w, label="AR", color="#e74c3c", alpha=0.85, edgecolor="white")
    axes[2].bar(x + w/2, fc_b, w, label="FCAR", color="#2ecc71", alpha=0.85, edgecolor="white")
    for i in range(n):
        for val, off, c in [(ar_b[i], -w/2, "#c0392b"), (fc_b[i], w/2, "#27ae60")]:
            axes[2].text(x[i]+off, val+0.002, f"{val:.3f}", ha="center", fontsize=7.5, color=c)
    axes[2].set_title("Avg Recourse Burden", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].legend(fontsize=9)
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Performance Trade-offs: FCAR vs Unconstrained AR (RQ3)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = FIG_DIR / "eval_performance_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FCAR Full Evaluation — Generating All Charts")
    print("=" * 60)

    evals = load_eval_jsons()
    benchmarks = load_benchmark_jsons()

    if not evals:
        print("[ERROR] No *_eval.json files found. Run evaluate_baseline.py first.")
        return
    if not benchmarks:
        print("[ERROR] No *_ab_summary.json files found. Run benchmark_ab.py first.")
        return

    print(f"\n[INFO] {len(evals)} evaluation JSONs, {len(benchmarks)} benchmark summaries\n")

    fig_model_quality(evals)
    fig_fairness_gaps(evals)
    fig_social_burden_grid(benchmarks)
    fig_gap_reduction(benchmarks)
    fig_statistical_summary(benchmarks)
    fig_performance_dashboard(benchmarks)

    print(f"\n{'=' * 60}")
    print(f"  All charts saved to: {FIG_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
