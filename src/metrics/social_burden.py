"""
Social Burden (MISOB) metric module for FCAR.

Implements the Social Burden framework based on Barrainkua et al. (2025),
measuring the disproportionate total effort imposed on protected groups.

Core formula:
    SB(g) = rejection_rate(g) × avg_recourse_cost(g)

The Social Burden captures two compounding effects:
  1. Decision Disparity: some groups are rejected more often
  2. Recourse Disparity: rejected members of some groups face costlier changes

This module provides reusable functions for computing per-group burden,
burden disparity, and an audit score.
"""

import numpy as np
import pandas as pd


def compute_rejection_rate(
    y_pred: np.ndarray,
    groups: pd.Series,
    positive_label: int = 1,
) -> pd.Series:
    """
    Compute per-group rejection rate (proportion predicted negative).

    Parameters
    ----------
    y_pred : array-like
        Binary predictions (1 = approved, 0 = rejected).
    groups : pd.Series
        Group labels aligned with y_pred.
    positive_label : int
        Which label means 'approved' (default 1).

    Returns
    -------
    pd.Series indexed by group with rejection rate values.
    """
    df = pd.DataFrame({"pred": np.asarray(y_pred).ravel(), "group": groups.values})
    rejection = df.groupby("group")["pred"].apply(
        lambda s: (s != positive_label).mean()
    )
    rejection.name = "rejection_rate"
    return rejection


def compute_avg_recourse_cost(
    recourse_df: pd.DataFrame,
    group_col: str,
    cost_col: str = "burden_total",
    only_feasible: bool = True,
) -> pd.Series:
    """
    Compute per-group average recourse cost from a recourse results DataFrame.

    Parameters
    ----------
    recourse_df : pd.DataFrame
        Must contain `group_col`, `cost_col`, and optionally 'slack'.
    group_col : str
        Column identifying the sensitive group.
    cost_col : str
        Column containing the per-individual recourse cost/burden.
    only_feasible : bool
        If True, only include rows where slack == 0 (feasible recourse).

    Returns
    -------
    pd.Series indexed by group with avg recourse cost values.
    """
    df = recourse_df.copy()
    if only_feasible and "slack" in df.columns:
        df = df[df["slack"] == 0.0]

    if df.empty:
        return pd.Series(dtype=float, name="avg_recourse_cost")

    avg_cost = df.groupby(group_col)[cost_col].mean()
    avg_cost.name = "avg_recourse_cost"
    return avg_cost


def compute_social_burden(
    recourse_df: pd.DataFrame,
    group_col: str,
    cost_col: str = "burden_total",
    y_pred: np.ndarray = None,
    groups_all: pd.Series = None,
    only_feasible: bool = True,
    positive_label: int = 1,
) -> pd.DataFrame:
    """
    Compute per-group Social Burden: SB(g) = rejection_rate(g) × avg_cost(g).

    Two usage modes:
      A) If y_pred and groups_all are provided, rejection rate is computed
         from the full test set predictions.
      B) If not provided, rejection rate is estimated from the recourse_df
         itself (all rows are assumed rejected, and group proportions come
         from the data).

    Parameters
    ----------
    recourse_df : pd.DataFrame
        Recourse results. Must contain group_col, cost_col, 'slack'.
    group_col : str
        Column identifying the sensitive group.
    cost_col : str
        Column containing per-individual recourse cost.
    y_pred : np.ndarray, optional
        Full test set predictions for computing true rejection rates.
    groups_all : pd.Series, optional
        Full test set group labels, aligned with y_pred.
    only_feasible : bool
        If True, only feasible recourse rows contribute to avg cost.

    Returns
    -------
    pd.DataFrame with columns:
        rejection_rate, avg_recourse_cost, social_burden
    Indexed by group.
    """
    # Avg recourse cost per group
    avg_cost = compute_avg_recourse_cost(
        recourse_df, group_col, cost_col, only_feasible
    )

    # Rejection rate per group
    if y_pred is not None and groups_all is not None:
        rej_rate = compute_rejection_rate(y_pred, groups_all, positive_label=positive_label)
    else:
        # Fallback: estimate from recourse_df (all rows are rejected)
        # Use count per group / total to weight, but rate is effectively 1.0
        # for all since they're all rejected individuals.
        # Better: if 'flipped' column exists, treat non-flipped as still-rejected
        rej_rate = pd.Series(1.0, index=avg_cost.index, name="rejection_rate")

    # Align indices
    common_groups = avg_cost.index.intersection(rej_rate.index)
    rej_rate = rej_rate.reindex(common_groups).fillna(0.0)
    avg_cost = avg_cost.reindex(common_groups).fillna(0.0)

    result = pd.DataFrame({
        "rejection_rate": rej_rate,
        "avg_recourse_cost": avg_cost,
        "social_burden": rej_rate * avg_cost,
    })
    result.index.name = group_col
    return result.sort_values("social_burden", ascending=False)


def compute_burden_disparity(
    sb_df: pd.DataFrame,
    min_group_n: int = 0,
    group_counts: pd.Series = None,
) -> dict:
    """
    Compute burden disparity statistics from a Social Burden DataFrame.

    Parameters
    ----------
    sb_df : pd.DataFrame
        Output of compute_social_burden (must have 'social_burden' column).
    min_group_n : int
        Exclude groups smaller than this count (requires group_counts).
    group_counts : pd.Series, optional
        Number of individuals per group (for filtering small groups).

    Returns
    -------
    dict with keys:
        max_burden, min_burden, gap, ratio, worst_group, best_group
    """
    eligible = sb_df.copy()

    if group_counts is not None and min_group_n > 0:
        valid_groups = group_counts[group_counts >= min_group_n].index
        eligible = eligible.loc[eligible.index.intersection(valid_groups)]

    if eligible.empty:
        return {
            "max_burden": np.nan,
            "min_burden": np.nan,
            "gap": np.nan,
            "ratio": np.nan,
            "worst_group": "",
            "best_group": "",
        }

    max_sb = float(eligible["social_burden"].max())
    min_sb = float(eligible["social_burden"].min())
    gap = max_sb - min_sb
    ratio = (max_sb / min_sb) if min_sb > 1e-12 else np.inf

    return {
        "max_burden": max_sb,
        "min_burden": min_sb,
        "gap": gap,
        "ratio": ratio,
        "worst_group": str(eligible["social_burden"].idxmax()),
        "best_group": str(eligible["social_burden"].idxmin()),
    }


def compute_audit_score(
    sb_df: pd.DataFrame,
    epsilon: float = 0.10,
    min_group_n: int = 0,
    group_counts: pd.Series = None,
) -> dict:
    """
    Compute an auditability score (0.0–1.0) indicating fairness compliance.

    A score of 1.0 means burden is perfectly equal across groups.
    A score near 0.0 means extreme disparity.

    The score is: 1 - min(1, gap / (epsilon * overall_mean_burden)).

    If gap ≤ epsilon * overall_mean_burden → score ≥ 0 (passing).

    Parameters
    ----------
    sb_df : pd.DataFrame
        Output of compute_social_burden.
    epsilon : float
        Acceptable relative disparity tolerance (default 0.10 = 10%,
        derived from survey: 61% of respondents required <10% disparity).
    min_group_n : int
        Exclude small groups.
    group_counts : pd.Series, optional
        Counts per group.

    Returns
    -------
    dict with keys: audit_score, gap, mean_burden, threshold, passed
    """
    disparity = compute_burden_disparity(sb_df, min_group_n, group_counts)

    mean_burden = float(sb_df["social_burden"].mean()) if not sb_df.empty else 0.0
    threshold = epsilon * mean_burden if mean_burden > 1e-12 else 1e-12

    gap = disparity["gap"] if not np.isnan(disparity["gap"]) else 0.0
    score = max(0.0, 1.0 - (gap / threshold)) if threshold > 0 else 0.0

    return {
        "audit_score": round(float(score), 4),
        "gap": round(float(gap), 6),
        "mean_burden": round(float(mean_burden), 6),
        "threshold": round(float(threshold), 6),
        "passed": gap <= threshold,
        "worst_group": disparity["worst_group"],
        "best_group": disparity["best_group"],
    }


def compute_recourse_burden(
    recourse_df: pd.DataFrame,
    numeric_delta_cols: list[str],
    numeric_ranges: dict[str, float],
    numeric_weights: dict[str, float] = None,
    cat_step_cols: list[str] = None,
    cat_step_weight: float = 0.25,
) -> pd.Series:
    """
    Compute per-individual burden score (weighted normalized L1 + weighted cat steps).

    This is a convenience function to add a 'burden_total' column to
    recourse DataFrames before feeding into compute_social_burden.

    The formula matches the MIP solver objective:
        burden = sum( w_i * |delta_i| / range_i ) + sum( w_cat * steps_j )

    Parameters
    ----------
    recourse_df : pd.DataFrame
        Must contain the delta columns.
    numeric_delta_cols : list[str]
        Columns with numeric feature deltas (e.g. 'delta_credit_amount').
    numeric_ranges : dict
        {col: range_value} for normalizing each delta column.
    numeric_weights : dict, optional
        {col: cost_weight} for per-feature cost weights, matching the solver
        objective. If None, all weights default to 1.0 (unweighted L1).
    cat_step_cols : list[str], optional
        Columns with categorical step counts (e.g. 'checking_steps').
    cat_step_weight : float
        Weight per categorical step in the burden score.

    Returns
    -------
    pd.Series with per-individual burden score.
    """
    if numeric_weights is None:
        numeric_weights = {}

    burden = pd.Series(0.0, index=recourse_df.index)

    for col in numeric_delta_cols:
        if col in recourse_df.columns:
            r = numeric_ranges.get(col, 1.0)
            w = numeric_weights.get(col, 1.0)
            if r > 1e-9:
                burden += w * recourse_df[col].abs() / r

    if cat_step_cols:
        for col in cat_step_cols:
            if col in recourse_df.columns:
                burden += cat_step_weight * recourse_df[col].fillna(0.0).abs()

    burden.name = "burden_total"
    return burden
