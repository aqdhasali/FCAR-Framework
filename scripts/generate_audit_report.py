"""
Generate structured audit reports from a recourse generation output CSV.

Reads a recourse run CSV (e.g., from batch_recourse.py or benchmark_ab.py), 
computes the MISOB Audit Score (UC-103), and generates a human-readable 
JSON/Markdown report detailing the fairness metrics and "Why" explanations (UC-105).

Usage:
  python scripts/generate_audit_report.py \
    --input artifacts/reports/benchmarks/german_age_bucket_ab_comparison.csv \
    --group_col age_bucket --method fcar
"""

import sys
from pathlib import Path
import json
import argparse
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.social_burden import compute_social_burden, compute_burden_disparity, compute_audit_score

ART = ROOT / "artifacts"
REPORTS_DIR = ART / "reports" / "audits"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_why_explanations(df: pd.DataFrame, limit: int = 5):
    """
    Generate natural language 'Why' explanations for individuals (UC-105).
    Extracts the feature deltas and formatting them into a readable format.
    """
    explanations = []
    
    # Get just the flipped/feasible ones
    feasible = df[(df["slack"] == 0.0) & (df["flipped"] == 1)].head(limit)
    
    # Identify delta columns dynamically
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    
    for _, row in feasible.iterrows():
        idx = int(row["test_idx"])
        changes = []
        
        # Numeric changes
        for dc in delta_cols:
            val = row.get(dc, np.nan)
            if pd.notna(val) and abs(val) > 1e-6:
                feat_name = dc.replace("delta_", "")
                direction = "Increase" if val > 0 else "Decrease"
                changes.append(f"{direction} {feat_name} by {abs(val):.2f}")
                
        # Categorical changes
        cat_befores = [c for c in df.columns if c.endswith("_before") and not c.startswith("p_") and not c.startswith("duration") and not c.startswith("credit") and not c.startswith("delta")]
        for cb in cat_befores:
            base = cb.replace("_before", "")
            ca = f"{base}_after"
            if ca in df.columns:
                val_b = row.get(cb, "")
                val_a = row.get(ca, "")
                if val_b != val_a and pd.notna(val_b) and pd.notna(val_a):
                     changes.append(f"Change {base} from '{val_b}' to '{val_a}'")
                     
        explanation = {
            "test_idx": idx,
            "original_probability": round(float(row["p_before"]), 4),
            "new_probability": round(float(row["p_after"]), 4),
            "burden_score": round(float(row.get("burden_total", 0.0)), 4),
            "required_changes": changes,
            "narrative": f"To overturn the rejection, the applicant must: {', '.join(changes)}."
        }
        explanations.append(explanation)
        
    return explanations


def generate_markdown(audit_data: dict) -> str:
    """Formats the JSON audit dictionary into a readable Markdown report."""
    md = [
        f"# Fairness Audit Report",
        f"**Dataset/Source**: `{audit_data['source_file']}`",
        f"**Sensitive Group**: `{audit_data['group_col']}`",
        f"**Method**: `{audit_data['method_filter']}`",
        f"**Total Instances Evaluated**: {audit_data['metrics']['n_evaluated']}",
        f"**Feasible Recourse Rate**: {audit_data['metrics']['feasible_rate']:.2%}",
        "\n---",
        "## MISOB Fairness Audit",
        f"- **Audit Status**: {'Pass ✅' if audit_data['audit'].get('is_passing', False) else 'Fail ❌'}",
        f"- **Audit Score**: **{audit_data['audit']['audit_score']:.3f}** (0 = Perfect Equity)",
        f"- **Maximum Disparity Gap**: {audit_data['disparity']['gap']:.4f}",
        f"- **Most Burdened Group**: `{audit_data['disparity']['worst_group']}`",
        "\n### Social Burden by Group Matrix",
        "| Group | Rejection Rate | Avg Cost | Social Burden Score |",
        "|---|---|---|---|"
    ]
    
    sb_matrix = audit_data['social_burden']
    for row in sb_matrix:
        g = row['group']
        r = row.get('rejection_rate', 0.0)
        c = row.get('avg_recourse_cost', 0.0)
        s = row.get('social_burden', 0.0)
        md.append(f"| {g} | {r:.4f} | {c:.4f} | {s:.4f} |")
        
    md.extend([
        "\n---",
        "## Example Individual Recourse Explanations (Why & How)"
    ])
    
    for ex in audit_data['explanations']:
        md.append(f"### Applicant ID: {ex['test_idx']}")
        md.append(f"- **Original Score**: {ex['original_probability']:.4f}")
        md.append(f"- **Counterfactual Score**: {ex['new_probability']:.4f}")
        md.append(f"- **Individual Burden**: {ex['burden_score']:.4f}")
        md.append(f"- **Narrative**: _{ex['narrative']}_")
        md.append("")
        
    return "\n".join(md)


def main(args):
    in_file = Path(args.input)
    if not in_file.exists():
        print(f"[ERROR] Input file not found: {in_file}")
        return

    df = pd.read_csv(in_file)
    
    if args.method and "method" in df.columns:
        df = df[df["method"] == args.method].copy()
        
    if df.empty:
        print("[ERROR] DataFrame is empty after filtering. Check your inputs.")
        return

    # Compute overall metrics
    feasible_rate = (df["slack"] == 0.0).mean()
    flip_rate = df["flipped"].mean()

    # Compute Social Burden
    sb_df = compute_social_burden(df, group_col=args.group_col, cost_col="burden_total", only_feasible=True)
    
    # Compute Disparity & Audit Score
    disp = compute_burden_disparity(sb_df)
    audit = compute_audit_score(sb_df, epsilon=args.epsilon)
    
    # Get Explanations
    explanations = parse_why_explanations(df, limit=args.limit_explanations)

    audit_data = {
        "source_file": str(in_file.name),
        "group_col": args.group_col,
        "method_filter": args.method or "all",
        "timestamp": pd.Timestamp.now().isoformat(),
        "metrics": {
            "n_evaluated": int(len(df)),
            "feasible_rate": float(feasible_rate),
            "flip_rate": float(flip_rate),
        },
        "disparity": disp,
        "audit": audit,
        "social_burden": sb_df.reset_index().to_dict(orient="records"),
        "explanations": explanations
    }
    
    # Fix pandas dtypes in dict for JSON serialization
    audit_data_str = json.dumps(audit_data, default=lambda x: str(x) if pd.isna(x) else float(x) if isinstance(x, (np.float64, np.float32)) else x)
    audit_data = json.loads(audit_data_str)

    prefix = f"{in_file.stem}_{args.method}" if args.method else in_file.stem
    json_path = REPORTS_DIR / f"{prefix}_audit.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
        
    md_path = REPORTS_DIR / f"{prefix}_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown(audit_data))
        
    print(f"[OK] Audit Report saved to: {json_path}")
    print(f"[OK] Markdown Report saved to: {md_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Audit Report")
    ap.add_argument("--input", required=True, type=str, help="Path to recourse CSV file")
    ap.add_argument("--group_col", required=True, type=str, help="Sensitive group column")
    ap.add_argument("--method", type=str, default=None, help="Filter by method column (e.g. fcar or unconstrained_ar)")
    ap.add_argument("--epsilon", type=float, default=0.10, help="Disparity tolerance for passing audit")
    ap.add_argument("--limit_explanations", type=int, default=5, help="Number of individuals to explain")
    
    args = ap.parse_args()
    main(args)
