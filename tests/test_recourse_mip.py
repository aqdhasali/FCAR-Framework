"""
Unit tests for src.recourse.generic_recourse_mip — MIP solver + helpers.

Uses the toy pipeline / config from conftest.py so tests run in seconds
without needing real model artefacts.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src.recourse.generic_recourse_mip import (
    _full_logit,
    _extract_numeric_beta,
    _extract_cat_weights,
    _get_cat_categories_from_encoder,
    _rank_map_for,
    solve_recourse_mip,
)


# ═══════════════════════════════════════════════════════════════
# Helper function tests
# ═══════════════════════════════════════════════════════════════


class TestFullLogit:

    def test_returns_float(self, toy_pipeline, toy_data):
        X, _ = toy_data
        logit = _full_logit(toy_pipeline, X.iloc[0])
        assert isinstance(logit, float)

    def test_logit_consistent_with_predict_proba(self, toy_pipeline, toy_data):
        X, _ = toy_data
        x = X.iloc[0]
        logit = _full_logit(toy_pipeline, x)
        prob = 1.0 / (1.0 + np.exp(-logit))
        proba_sklearn = toy_pipeline.predict_proba(
            pd.DataFrame([x])
        )[:, 1][0]
        assert abs(prob - proba_sklearn) < 1e-6


class TestExtractNumericBeta:

    def test_returns_dict(self, toy_pipeline):
        beta = _extract_numeric_beta(toy_pipeline)
        assert isinstance(beta, dict)

    def test_has_numeric_cols(self, toy_pipeline):
        beta = _extract_numeric_beta(toy_pipeline)
        assert "duration" in beta
        assert "credit_amount" in beta
        assert "installment_commitment" in beta

    def test_betas_are_floats(self, toy_pipeline):
        beta = _extract_numeric_beta(toy_pipeline)
        for v in beta.values():
            assert isinstance(v, float)

    def test_no_categorical_cols(self, toy_pipeline):
        beta = _extract_numeric_beta(toy_pipeline)
        assert "checking_status" not in beta
        assert "savings_status" not in beta


class TestExtractCatWeights:

    def test_returns_nested_dict(self, toy_pipeline):
        w = _extract_cat_weights(toy_pipeline, ["checking_status", "savings_status"])
        assert isinstance(w, dict)
        assert "checking_status" in w
        assert isinstance(w["checking_status"], dict)

    def test_all_categories_have_weights(self, toy_pipeline):
        w = _extract_cat_weights(toy_pipeline, ["checking_status"])
        cats = w["checking_status"]
        assert len(cats) >= 3  # at least a few categories


class TestGetCatCategoriesFromEncoder:

    def test_returns_string_lists(self, toy_pipeline):
        cats = _get_cat_categories_from_encoder(
            toy_pipeline, ["checking_status", "savings_status"]
        )
        assert isinstance(cats["checking_status"], list)
        assert all(isinstance(c, str) for c in cats["checking_status"])
        assert "A11" in cats["checking_status"]

    def test_unknown_col_returns_empty(self, toy_pipeline):
        cats = _get_cat_categories_from_encoder(toy_pipeline, ["nonexistent_col"])
        assert cats["nonexistent_col"] == []


class TestRankMapFor:

    def test_ordered_categories(self):
        categories = ["A11", "A12", "A13", "A14"]
        orders = {"checking_status": ["A11", "A12", "A13", "A14"]}
        rank, order = _rank_map_for("checking_status", categories, orders)
        assert rank["A11"] == 0
        assert rank["A14"] == 3
        assert order == ["A11", "A12", "A13", "A14"]

    def test_extra_categories_appended(self):
        categories = ["A11", "A12", "A99"]
        orders = {"col": ["A11", "A12"]}
        rank, order = _rank_map_for("col", categories, orders)
        assert "A99" in rank
        assert rank["A99"] == 2

    def test_no_order_defined(self):
        categories = ["X", "Y", "Z"]
        rank, order = _rank_map_for("col", categories, {})
        assert len(rank) == 3
        assert set(order) == {"X", "Y", "Z"}


# ═══════════════════════════════════════════════════════════════
# MIP Solver — integration tests
# ═══════════════════════════════════════════════════════════════


class TestSolveRecourseMIP:

    def test_returns_series_and_slack(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, slack = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        assert isinstance(x_cf, pd.Series)
        assert isinstance(slack, float)

    def test_feasible_solution_flips_decision(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, p0 = rejected_applicant
        x_cf, slack = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        if slack == 0.0:
            p1 = toy_pipeline.predict_proba(pd.DataFrame([x_cf]))[:, 1][0]
            assert p1 >= 0.5, (
                f"Solver returned slack=0 but P(approved)={p1:.4f} < 0.5"
            )

    def test_logit_constraint_satisfied(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, slack = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        logit_cf = _full_logit(toy_pipeline, x_cf)
        # logit + slack >= margin (1e-6)
        assert logit_cf + slack >= -1e-4, (
            f"Logit constraint violated: logit={logit_cf}, slack={slack}"
        )

    def test_plausibility_duration_decrease_only(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, _ = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        # duration has direction=decrease: x_cf <= x0
        assert x_cf["duration"] <= x0["duration"] + 1e-6

    def test_plausibility_credit_decrease_only(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, _ = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        # credit_amount has direction=decrease
        assert x_cf["credit_amount"] <= x0["credit_amount"] + 1e-6

    def test_plausibility_max_decrease_respected(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, _ = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        # max_decrease = 48 for duration
        assert x0["duration"] - x_cf["duration"] <= 48 + 1e-6

    def test_plausibility_max_rel_decrease_respected(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, _ = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        # max_rel_decrease = 0.80 for credit_amount
        if x0["credit_amount"] > 0:
            min_allowed = x0["credit_amount"] * (1 - 0.80)
            assert x_cf["credit_amount"] >= min_allowed - 1e-2

    def test_monotonic_categorical_no_worsening(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, _ = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        # checking_status is monotonic [A11 < A12 < A13 < A14]
        order = ["A11", "A12", "A13", "A14"]
        v0 = str(x0["checking_status"])
        v1 = str(x_cf["checking_status"])
        if v0 in order and v1 in order:
            assert order.index(v1) >= order.index(v0), (
                f"Monotonic violation: {v0} -> {v1}"
            )

    def test_max_cat_changes_respected(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, _ = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        # max_cat_changes = 1
        n_changed = 0
        for c in ["checking_status", "savings_status"]:
            if str(x0[c]) != str(x_cf[c]):
                n_changed += 1
        assert n_changed <= 1

    def test_integer_features_are_integer(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, _ = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        for c in ["duration", "credit_amount", "installment_commitment"]:
            val = x_cf[c]
            assert abs(val - round(val)) < 0.1, (
                f"{c}={val} is not integer-valued"
            )

    def test_immutable_features_unchanged(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, _ = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        for c in ["age", "personal_status", "foreign_worker"]:
            if c in x0.index and c in x_cf.index:
                assert x0[c] == x_cf[c], f"Immutable feature {c} was changed"

    def test_fcar_weights_change_objective_not_constraints(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        """FCAR adjusts weights but the constraint (logit >= 0) still holds."""
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        fcar_config = copy.deepcopy(german_config)
        fcar_config["mutable_numeric"]["duration"]["cost_weight"] = 0.3
        fcar_config["mutable_numeric"]["credit_amount"]["cost_weight"] = 0.2
        x_cf, slack = solve_recourse_mip(
            toy_pipeline, X_train, x0, fcar_config
        )
        if slack == 0.0:
            p1 = toy_pipeline.predict_proba(pd.DataFrame([x_cf]))[:, 1][0]
            assert p1 >= 0.5

    def test_within_training_bounds(
        self, toy_pipeline, toy_train_test, german_config, rejected_applicant
    ):
        X_train, _, _, _ = toy_train_test
        _, x0, _ = rejected_applicant
        x_cf, _ = solve_recourse_mip(
            toy_pipeline, X_train, x0, german_config
        )
        for c in ["duration", "credit_amount", "installment_commitment"]:
            lo = X_train[c].min()
            hi = X_train[c].max()
            assert x_cf[c] >= lo - 1e-6, f"{c}={x_cf[c]} below train min {lo}"
            assert x_cf[c] <= hi + 1e-6, f"{c}={x_cf[c]} above train max {hi}"
