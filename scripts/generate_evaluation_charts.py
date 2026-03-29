"""
Generate charts for baseline evaluation metrics.

This script loads the trained baseline models and test sets for all datasets to:
  1) Generate ROC curves
  2) Generate Precision-Recall (PR) curves
  3) Read from evaluation JSONs to generate grouped bar charts for Fairness metrics 
     (Demographic Parity Diff, Equalized Odds Diff, etc.)

Outputs are saved to artifacts/reports/figures/
"""

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ART = ROOT / "artifacts"
REPORTS_DIR = ART / "reports"
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_test(name: str):
    X = pd.read_csv(PROC / name / "X.csv")
    y = pd.read_csv(PROC / name / "y.csv").iloc[:, 0].astype(int)
    test_idx = np.load(SPLITS / name / "test_idx.npy")
    return X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)


def plot_roc_pr_curves(datasets=["german", "adult", "default_credit"]):
    """Plot combined ROC and PR curves for all datasets."""
    
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    
    colors = {"german": "#e74c3c", "adult": "#3498db", "default_credit": "#2ecc71"}
    
    for ds in datasets:
        model_path = ART / "models" / f"{ds}_logreg.joblib"
        if not model_path.exists():
            print(f"[WARN] Cannot find model for {ds}, skipping curves.")
            continue
            
        model = joblib.load(model_path)
        X_test, y_test = load_test(ds)
        proba = model.predict_proba(X_test)[:, 1]
        
        # ROC
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors[ds], lw=2, label=f"{ds.title()} (AUC = {roc_auc:.3f})")
        
        # PR
        precision, recall, _ = precision_recall_curve(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        
        # baseline PR is ratio of positives
        baseline = y_test.mean()
        ax_pr.plot(recall, precision, color=colors[ds], lw=2, label=f"{ds.title()} (AUC = {pr_auc:.3f})")
        ax_pr.axhline(baseline, color=colors[ds], linestyle="--", alpha=0.5, label=f"{ds.title()} Baseline")

    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate", fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate", fontweight="bold")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)
    
    fig_roc.tight_layout()
    fig_roc.savefig(FIG_DIR / "baseline_roc_curves.png", dpi=150)
    plt.close(fig_roc)
    print(f"[OK] Saved ROC curves to {FIG_DIR / 'baseline_roc_curves.png'}")
    
    # PR formatting
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall", fontweight="bold")
    ax_pr.set_ylabel("Precision", fontweight="bold")
    ax_pr.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    # remove duplicate baseline labels if necessary, we just use standard legend
    handles, labels = ax_pr.get_legend_handles_labels()
    # Let's clean up baseline labels to not clutter
    ax_pr.legend(loc="upper right", fontsize=9)
    ax_pr.grid(alpha=0.3)
    
    fig_pr.tight_layout()
    fig_pr.savefig(FIG_DIR / "baseline_pr_curves.png", dpi=150)
    plt.close(fig_pr)
    print(f"[OK] Saved PR curves to {FIG_DIR / 'baseline_pr_curves.png'}")


def plot_fairness_metrics():
    """Plot a grouped bar chart for fairness gaps derived from the JSON evals."""
    
    eval_files = sorted(REPORTS_DIR.glob("*_eval.json"))
    if not eval_files:
        print("[WARN] No evaluation JSONs found in artifacts/reports/. Skipping fairness chart.")
        return
        
    records = []
    for f in eval_files:
        with open(f, "r") as fh:
            data = json.load(fh)
        ds = data.get("dataset", f.stem.replace("_eval", ""))
        fairness = data.get("fairness", {})
        
        for attr, metrics in fairness.items():
            records.append({
                "Dataset": ds.title(),
                "Attribute": attr,
                "Demographic Parity Diff": metrics.get("demographic_parity_diff", 0.0),
                "Equalized Odds Diff": metrics.get("equalized_odds_diff", 0.0),
                "Selection Rate Gap": metrics.get("gap_selection_rate", 0.0)
            })
            
    if not records:
        return
        
    df = pd.DataFrame(records)
    # We will plot the maximum gap per dataset across any attribute, or plot all dataset-attribute pairs
    df["Label"] = df["Dataset"] + "\n(" + df["Attribute"] + ")"
    
    labels = df["Label"].tolist()
    dp = df["Demographic Parity Diff"].tolist()
    eo = df["Equalized Odds Diff"].tolist()
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, dp, width, label="Demographic Parity Diff", color="#f39c12", edgecolor="white", alpha=0.85)
    ax.bar(x + width/2, eo, width, label="Equalized Odds Diff", color="#8e44ad", edgecolor="white", alpha=0.85)
    
    ax.set_ylabel("Difference", fontsize=12, fontweight="bold")
    ax.set_title("Baseline Fairness Metrics Before Recourse", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=0)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    out_path = FIG_DIR / "baseline_fairness_metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved Fairness charts to {out_path}")


def main():
    print("[INFO] Generating evaluation charts...")
    plot_roc_pr_curves()
    plot_fairness_metrics()
    print("[DONE]")


if __name__ == "__main__":
    main()
