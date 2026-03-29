"""
Unit tests for src.metrics.social_burden — MISOB metric functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics.social_burden import (
    compute_rejection_rate,
    compute_avg_recourse_cost,
    compute_social_burden,
    compute_burden_disparity,
    compute_audit_score,
    compute_recourse_burden,
)


# ═══════════════════════════════════════════════════════════════
# compute_rejection_rate
# ═══════════════════════════════════════════════════════════════


class TestComputeRejectionRate:

    def test_perfect_approval(self):
        y_pred = np.array([1, 1, 1, 1])
        groups = pd.Series(["A", "A", "B", "B"])
        rr = compute_rejection_rate(y_pred, groups)
        assert rr["A"] == 0.0
        assert rr["B"] == 0.0

    def test_full_rejection(self):
        y_pred = np.array([0, 0, 0, 0])
        groups = pd.Series(["A", "A", "B", "B"])
        rr = compute_rejection_rate(y_pred, groups)
        assert rr["A"] == 1.0
        assert rr["B"] == 1.0

    def test_mixed(self):
        y_pred = np.array([1, 0, 1, 1, 0, 0])
        groups = pd.Series(["A", "A", "A", "B", "B", "B"])
        rr = compute_rejection_rate(y_pred, groups)
        assert abs(rr["A"] - 1 / 3) < 1e-9
        assert abs(rr["B"] - 2 / 3) < 1e-9

    def test_single_group(self):
        y_pred = np.array([0, 1])
        groups = pd.Series(["X", "X"])
        rr = compute_rejection_rate(y_pred, groups)
        assert abs(rr["X"] - 0.5) < 1e-9

    def test_custom_positive_label(self):
        y_pred = np.array([0, 0, 1, 1])
        groups = pd.Series(["A", "A", "B", "B"])
        rr = compute_rejection_rate(y_pred, groups, positive_label=0)
        assert rr["A"] == 0.0  # all predicted 0 = approved
        assert rr["B"] == 1.0  # all predicted 1 != 0 = rejected


# ═══════════════════════════════════════════════════════════════
# compute_avg_recourse_cost
# ═══════════════════════════════════════════════════════════════


class TestComputeAvgRecourseCost:

    def test_basic_avg(self):
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "burden_total": [1.0, 3.0, 2.0, 4.0],
            "slack": [0, 0, 0, 0],
        })
        avg = compute_avg_recourse_cost(df, "group")
        assert abs(avg["A"] - 2.0) < 1e-9
        assert abs(avg["B"] - 3.0) < 1e-9

    def test_filters_infeasible(self):
        df = pd.DataFrame({
            "group": ["A", "A", "B"],
            "burden_total": [2.0, 10.0, 5.0],
            "slack": [0, 1.0, 0],
        })
        avg = compute_avg_recourse_cost(df, "group", only_feasible=True)
        assert abs(avg["A"] - 2.0) < 1e-9  # only first row (slack=0)
        assert abs(avg["B"] - 5.0) < 1e-9

    def test_no_filter_when_disabled(self):
        df = pd.DataFrame({
            "group": ["A", "A"],
            "burden_total": [2.0, 10.0],
            "slack": [0, 1.0],
        })
        avg = compute_avg_recourse_cost(df, "group", only_feasible=False)
        assert abs(avg["A"] - 6.0) < 1e-9

    def test_empty_df(self):
        df = pd.DataFrame(columns=["group", "burden_total", "slack"])
        avg = compute_avg_recourse_cost(df, "group")
        assert avg.empty

    def test_custom_cost_col(self):
        df = pd.DataFrame({
            "g": ["X", "X"],
            "my_cost": [3.0, 5.0],
            "slack": [0, 0],
        })
        avg = compute_avg_recourse_cost(df, "g", cost_col="my_cost")
        assert abs(avg["X"] - 4.0) < 1e-9


# ═══════════════════════════════════════════════════════════════
# compute_social_burden
# ═══════════════════════════════════════════════════════════════


class TestComputeSocialBurden:

    def test_with_predictions(self):
        recourse_df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "burden_total": [2.0, 4.0, 1.0, 3.0],
            "slack": [0, 0, 0, 0],
        })
        y_pred = np.array([1, 0, 0, 1, 0, 0])
        groups_all = pd.Series(["A", "A", "A", "B", "B", "B"])
        sb = compute_social_burden(recourse_df, "group",
                                   y_pred=y_pred, groups_all=groups_all)
        # A: rejection_rate = 2/3, avg_cost = 3.0, SB = 2.0
        assert abs(sb.loc["A", "social_burden"] - (2 / 3) * 3.0) < 1e-6
        # B: rejection_rate = 2/3, avg_cost = 2.0, SB ≈ 1.333
        assert abs(sb.loc["B", "social_burden"] - (2 / 3) * 2.0) < 1e-6

    def test_fallback_without_predictions(self):
        recourse_df = pd.DataFrame({
            "grp": ["X", "X", "Y"],
            "burden_total": [2.0, 4.0, 6.0],
            "slack": [0, 0, 0],
        })
        sb = compute_social_burden(recourse_df, "grp")
        # Fallback: rejection_rate = 1.0 for all groups
        assert abs(sb.loc["X", "rejection_rate"] - 1.0) < 1e-9
        assert abs(sb.loc["X", "avg_recourse_cost"] - 3.0) < 1e-9
        assert abs(sb.loc["X", "social_burden"] - 3.0) < 1e-9
        assert abs(sb.loc["Y", "social_burden"] - 6.0) < 1e-9

    def test_sorted_descending(self):
        recourse_df = pd.DataFrame({
            "g": ["A", "B", "C"],
            "burden_total": [1.0, 3.0, 2.0],
            "slack": [0, 0, 0],
        })
        sb = compute_social_burden(recourse_df, "g")
        burdens = sb["social_burden"].values
        assert burdens[0] >= burdens[-1]  # sorted descending


# ═══════════════════════════════════════════════════════════════
# compute_burden_disparity
# ═══════════════════════════════════════════════════════════════


class TestComputeBurdenDisparity:

    def _make_sb(self, burdens):
        return pd.DataFrame({
            "social_burden": burdens,
        }, index=[f"G{i}" for i in range(len(burdens))])

    def test_basic(self):
        sb_df = self._make_sb([0.5, 0.1])
        d = compute_burden_disparity(sb_df)
        assert abs(d["max_burden"] - 0.5) < 1e-9
        assert abs(d["min_burden"] - 0.1) < 1e-9
        assert abs(d["gap"] - 0.4) < 1e-9
        assert abs(d["ratio"] - 5.0) < 1e-9
        assert d["worst_group"] == "G0"
        assert d["best_group"] == "G1"

    def test_equal_burden(self):
        sb_df = self._make_sb([0.3, 0.3, 0.3])
        d = compute_burden_disparity(sb_df)
        assert abs(d["gap"]) < 1e-9
        assert abs(d["ratio"] - 1.0) < 1e-9

    def test_zero_min_burden(self):
        sb_df = self._make_sb([0.5, 0.0])
        d = compute_burden_disparity(sb_df)
        assert d["ratio"] == np.inf

    def test_empty_after_filtering(self):
        sb_df = self._make_sb([0.5])
        counts = pd.Series([2], index=["G0"])
        d = compute_burden_disparity(sb_df, min_group_n=10, group_counts=counts)
        assert np.isnan(d["gap"])

    def test_min_group_filter(self):
        sb_df = self._make_sb([0.5, 0.1, 0.3])
        counts = pd.Series([50, 3, 50], index=["G0", "G1", "G2"])
        d = compute_burden_disparity(sb_df, min_group_n=10, group_counts=counts)
        # G1 should be excluded (count=3 < 10)
        assert d["worst_group"] == "G0"
        assert d["best_group"] == "G2"
        assert abs(d["gap"] - 0.2) < 1e-9


# ═══════════════════════════════════════════════════════════════
# compute_audit_score
# ═══════════════════════════════════════════════════════════════


class TestComputeAuditScore:

    def _make_sb(self, burdens):
        return pd.DataFrame({"social_burden": burdens},
                            index=[f"G{i}" for i in range(len(burdens))])

    def test_perfect_equality(self):
        sb_df = self._make_sb([0.3, 0.3, 0.3])
        result = compute_audit_score(sb_df)
        assert result["audit_score"] == 1.0
        assert result["passed"] is True

    def test_large_disparity_fails(self):
        sb_df = self._make_sb([1.0, 0.0])
        result = compute_audit_score(sb_df)
        assert result["audit_score"] == 0.0
        assert result["passed"] is False

    def test_moderate_disparity(self):
        sb_df = self._make_sb([0.20, 0.18])
        result = compute_audit_score(sb_df, epsilon=0.10)
        # gap = 0.02, mean = 0.19, threshold = 0.019
        # gap > threshold → not passed (barely)
        assert isinstance(result["audit_score"], float)
        assert 0.0 <= result["audit_score"] <= 1.0

    def test_keys_present(self):
        sb_df = self._make_sb([0.3])
        result = compute_audit_score(sb_df)
        for k in ["audit_score", "gap", "mean_burden", "threshold", "passed",
                   "worst_group", "best_group"]:
            assert k in result


# ═══════════════════════════════════════════════════════════════
# compute_recourse_burden
# ═══════════════════════════════════════════════════════════════


class TestComputeRecourseBurden:

    def test_basic_numeric_burden(self):
        df = pd.DataFrame({
            "delta_dur": [10.0, 20.0],
            "delta_amt": [1000.0, 2000.0],
        })
        ranges = {"delta_dur": 60.0, "delta_amt": 10000.0}
        burden = compute_recourse_burden(df, ["delta_dur", "delta_amt"], ranges)
        # row 0: 10/60 + 1000/10000 = 0.1667 + 0.1 = 0.2667
        assert abs(burden.iloc[0] - (10 / 60 + 1000 / 10000)) < 1e-6
        # row 1: 20/60 + 2000/10000 = 0.3333 + 0.2 = 0.5333
        assert abs(burden.iloc[1] - (20 / 60 + 2000 / 10000)) < 1e-6

    def test_with_weights(self):
        df = pd.DataFrame({"delta_x": [10.0]})
        ranges = {"delta_x": 100.0}
        weights = {"delta_x": 0.5}
        burden = compute_recourse_burden(df, ["delta_x"], ranges,
                                         numeric_weights=weights)
        assert abs(burden.iloc[0] - 0.5 * 10 / 100) < 1e-9

    def test_with_cat_steps(self):
        df = pd.DataFrame({
            "delta_num": [0.0],
            "check_steps": [2],
        })
        ranges = {"delta_num": 1.0}
        burden = compute_recourse_burden(
            df, ["delta_num"], ranges,
            cat_step_cols=["check_steps"], cat_step_weight=0.25,
        )
        assert abs(burden.iloc[0] - 0.5) < 1e-9  # 0 + 0.25 * 2

    def test_negative_deltas_use_abs(self):
        df = pd.DataFrame({"d": [-5.0]})
        burden = compute_recourse_burden(df, ["d"], {"d": 10.0})
        assert abs(burden.iloc[0] - 0.5) < 1e-9

    def test_missing_col_ignored(self):
        df = pd.DataFrame({"a": [1.0]})
        burden = compute_recourse_burden(df, ["a", "nonexistent"], {"a": 10.0})
        assert abs(burden.iloc[0] - 0.1) < 1e-9

    def test_name_is_burden_total(self):
        df = pd.DataFrame({"d": [1.0]})
        burden = compute_recourse_burden(df, ["d"], {"d": 1.0})
        assert burden.name == "burden_total"
