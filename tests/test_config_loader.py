"""
Unit tests for src.config.config_loader — all public helpers.
"""

import pytest
from src.config.config_loader import (
    load_dataset_config,
    get_mutable_numeric_cols,
    get_mutable_categorical_cols,
    get_immutable_cols,
    get_sensitive_attributes,
    get_integer_cols,
    get_decrease_only_cols,
    get_increase_only_cols,
    get_numeric_cost_weights,
    get_categorical_step_weights,
    get_categorical_orders,
    get_monotonic_categorical_cols,
    get_solver_settings,
    get_survey_bounds,
    get_max_cat_changes,
    get_plausibility_params,
    VALID_DATASETS,
)


# ═══════════════════════════════════════════════════════════════
# load_dataset_config
# ═══════════════════════════════════════════════════════════════


class TestLoadDatasetConfig:

    def test_load_german(self):
        cfg = load_dataset_config("german")
        assert cfg["dataset"] == "german"
        assert "mutable_numeric" in cfg
        assert "mutable_categorical" in cfg

    def test_load_adult(self):
        cfg = load_dataset_config("adult")
        assert cfg["dataset"] == "adult"

    def test_load_default_credit(self):
        cfg = load_dataset_config("default_credit")
        assert cfg["dataset"] == "default_credit"

    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_config("nonexistent")

    def test_valid_datasets_constant(self):
        assert "german" in VALID_DATASETS
        assert "adult" in VALID_DATASETS
        assert "default_credit" in VALID_DATASETS


# ═══════════════════════════════════════════════════════════════
# Getters on the mini german_config fixture
# ═══════════════════════════════════════════════════════════════


class TestGetMutableNumericCols:

    def test_returns_list(self, german_config):
        result = get_mutable_numeric_cols(german_config)
        assert isinstance(result, list)

    def test_expected_cols(self, german_config):
        result = get_mutable_numeric_cols(german_config)
        assert "duration" in result
        assert "credit_amount" in result
        assert "installment_commitment" in result

    def test_empty_config(self):
        assert get_mutable_numeric_cols({}) == []


class TestGetMutableCategoricalCols:

    def test_returns_list(self, german_config):
        result = get_mutable_categorical_cols(german_config)
        assert isinstance(result, list)

    def test_expected_cols(self, german_config):
        result = get_mutable_categorical_cols(german_config)
        assert "checking_status" in result
        assert "savings_status" in result

    def test_empty_config(self):
        assert get_mutable_categorical_cols({}) == []


class TestGetImmutableCols:

    def test_returns_list(self, german_config):
        result = get_immutable_cols(german_config)
        assert "age" in result
        assert "personal_status" in result

    def test_empty_config(self):
        assert get_immutable_cols({}) == []


class TestGetSensitiveAttributes:

    def test_returns_list(self, german_config):
        result = get_sensitive_attributes(german_config)
        assert "personal_status" in result
        assert "age" in result

    def test_empty_config(self):
        assert get_sensitive_attributes({}) == []


class TestGetIntegerCols:

    def test_finds_integer_features(self, german_config):
        result = get_integer_cols(german_config)
        assert "duration" in result
        assert "credit_amount" in result
        assert "installment_commitment" in result

    def test_empty_config(self):
        assert get_integer_cols({}) == []


class TestGetDecreaseOnlyCols:

    def test_finds_decrease_features(self, german_config):
        result = get_decrease_only_cols(german_config)
        assert "duration" in result
        assert "credit_amount" in result
        # installment_commitment is "auto", should NOT be here
        assert "installment_commitment" not in result

    def test_empty_config(self):
        assert get_decrease_only_cols({}) == []


class TestGetIncreaseOnlyCols:

    def test_no_increase_cols_in_german(self, german_config):
        result = get_increase_only_cols(german_config)
        assert result == []

    def test_increase_col(self):
        cfg = {"mutable_numeric": {"income": {"direction": "increase"}}}
        assert get_increase_only_cols(cfg) == ["income"]


class TestGetNumericCostWeights:

    def test_default_weights(self, german_config):
        result = get_numeric_cost_weights(german_config)
        assert result["duration"] == 1.0
        assert result["credit_amount"] == 1.0

    def test_custom_weight(self):
        cfg = {"mutable_numeric": {"x": {"cost_weight": 0.5}}}
        assert get_numeric_cost_weights(cfg) == {"x": 0.5}

    def test_missing_weight_defaults_to_one(self):
        cfg = {"mutable_numeric": {"x": {}}}
        assert get_numeric_cost_weights(cfg) == {"x": 1.0}


class TestGetCategoricalStepWeights:

    def test_expected_weights(self, german_config):
        result = get_categorical_step_weights(german_config)
        assert result["checking_status"] == 0.25
        assert result["savings_status"] == 0.25

    def test_missing_weight_defaults(self):
        cfg = {"mutable_categorical": {"col": {}}}
        assert get_categorical_step_weights(cfg) == {"col": 0.25}


class TestGetCategoricalOrders:

    def test_returns_orders(self, german_config):
        result = get_categorical_orders(german_config)
        assert result["checking_status"] == ["A11", "A12", "A13", "A14"]
        assert len(result["savings_status"]) == 5

    def test_empty_order(self):
        cfg = {"mutable_categorical": {"col": {}}}
        assert get_categorical_orders(cfg) == {"col": []}


class TestGetMonotonicCategoricalCols:

    def test_finds_monotonic(self, german_config):
        result = get_monotonic_categorical_cols(german_config)
        assert "checking_status" in result
        assert "savings_status" in result

    def test_non_monotonic_excluded(self):
        cfg = {"mutable_categorical": {"col": {"monotonic": False}}}
        assert get_monotonic_categorical_cols(cfg) == []


class TestGetSolverSettings:

    def test_defaults_when_missing(self):
        result = get_solver_settings({})
        assert result["slack_penalty"] == 1000.0
        assert result["margin"] == 1e-6
        assert result["time_limit_seconds"] == 30

    def test_override(self, german_config):
        result = get_solver_settings(german_config)
        assert result["slack_penalty"] == 1000.0
        assert result["margin"] == 1e-6

    def test_custom_values(self):
        cfg = {"solver": {"slack_penalty": 500.0, "margin": 0.01}}
        result = get_solver_settings(cfg)
        assert result["slack_penalty"] == 500.0
        assert result["margin"] == 0.01
        # default time_limit should still be present
        assert result["time_limit_seconds"] == 30


class TestGetSurveyBounds:

    def test_empty(self, german_config):
        result = get_survey_bounds(german_config)
        assert isinstance(result, dict)


class TestGetMaxCatChanges:

    def test_returns_int(self, german_config):
        result = get_max_cat_changes(german_config)
        assert result == 1

    def test_none_when_missing(self):
        assert get_max_cat_changes({}) is None


class TestGetPlausibilityParams:

    def test_duration_params(self, german_config):
        result = get_plausibility_params(german_config)
        assert result["duration"]["max_decrease"] == 48

    def test_credit_params(self, german_config):
        result = get_plausibility_params(german_config)
        assert result["credit_amount"]["max_rel_decrease"] == 0.80

    def test_installment_has_no_params(self, german_config):
        result = get_plausibility_params(german_config)
        assert "installment_commitment" not in result

    def test_empty_config(self):
        assert get_plausibility_params({}) == {}
