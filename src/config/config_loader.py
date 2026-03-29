"""
Config loader for FCAR dataset-specific configurations.

Loads YAML config files from src/config/ and provides a clean API
for the recourse solvers, tuning scripts, and benchmarks.
"""

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent

VALID_DATASETS = ("german", "adult", "default_credit")


def load_dataset_config(dataset_name: str) -> dict[str, Any]:
    """
    Load a dataset configuration YAML and return as a dict.

    Parameters
    ----------
    dataset_name : str
        One of: 'german', 'adult', 'default_credit'.

    Returns
    -------
    dict with keys: dataset, mutable_numeric, mutable_categorical,
                    immutable, sensitive_attributes, survey_bounds, solver
    """
    if dataset_name not in VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {VALID_DATASETS}"
        )

    config_path = CONFIG_DIR / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Create it at src/config/{dataset_name}.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_mutable_numeric_cols(config: dict) -> list[str]:
    """Return list of mutable numeric feature names."""
    return list(config.get("mutable_numeric", {}).keys())


def get_mutable_categorical_cols(config: dict) -> list[str]:
    """Return list of mutable categorical feature names."""
    return list(config.get("mutable_categorical", {}).keys())


def get_immutable_cols(config: dict) -> list[str]:
    """Return list of immutable feature names."""
    return list(config.get("immutable", []))


def get_sensitive_attributes(config: dict) -> list[str]:
    """Return list of sensitive attribute names used for fairness slicing."""
    return list(config.get("sensitive_attributes", []))


def get_integer_cols(config: dict) -> list[str]:
    """Return list of numeric features that must be integer-valued."""
    return [
        col for col, spec in config.get("mutable_numeric", {}).items()
        if spec.get("integer", False)
    ]


def get_decrease_only_cols(config: dict) -> list[str]:
    """Return list of numeric features that can only decrease."""
    return [
        col for col, spec in config.get("mutable_numeric", {}).items()
        if spec.get("direction") == "decrease"
    ]


def get_increase_only_cols(config: dict) -> list[str]:
    """Return list of numeric features that can only increase."""
    return [
        col for col, spec in config.get("mutable_numeric", {}).items()
        if spec.get("direction") == "increase"
    ]


def get_numeric_cost_weights(config: dict) -> dict[str, float]:
    """Return {col: weight} for numeric features."""
    return {
        col: float(spec.get("cost_weight", 1.0))
        for col, spec in config.get("mutable_numeric", {}).items()
    }


def get_categorical_step_weights(config: dict) -> dict[str, float]:
    """Return {col: step_weight} for categorical features."""
    return {
        col: float(spec.get("step_weight", 0.25))
        for col, spec in config.get("mutable_categorical", {}).items()
    }


def get_categorical_orders(config: dict) -> dict[str, list[str]]:
    """Return {col: [ordered categories]} for categorical features."""
    return {
        col: spec.get("order", [])
        for col, spec in config.get("mutable_categorical", {}).items()
    }


def get_monotonic_categorical_cols(config: dict) -> list[str]:
    """Return categorical columns with monotonic improvement constraint."""
    return [
        col for col, spec in config.get("mutable_categorical", {}).items()
        if spec.get("monotonic", False)
    ]


def get_solver_settings(config: dict) -> dict:
    """Return solver settings with defaults."""
    defaults = {"slack_penalty": 1000.0, "margin": 1e-6, "time_limit_seconds": 30}
    solver = config.get("solver", {})
    return {**defaults, **solver}


def get_survey_bounds(config: dict) -> dict:
    """Return survey-derived plausibility bounds."""
    return config.get("survey_bounds", {})


def get_max_cat_changes(config: dict) -> int | None:
    """Return the maximum number of categorical features that may change.

    Returns None if no limit is configured (backwards compatible).
    """
    val = config.get("max_cat_changes", None)
    return int(val) if val is not None else None


def get_plausibility_params(config: dict) -> dict:
    """
    Extract plausibility constraint parameters from numeric feature specs.

    Returns dict like:
      {
        "duration": {"max_decrease": 48},
        "credit_amount": {"max_rel_decrease": 0.80},
        ...
      }
    """
    result = {}
    for col, spec in config.get("mutable_numeric", {}).items():
        params = {}
        if "max_decrease" in spec:
            params["max_decrease"] = float(spec["max_decrease"])
        if "max_rel_decrease" in spec:
            params["max_rel_decrease"] = float(spec["max_rel_decrease"])
        if "max_increase" in spec:
            params["max_increase"] = float(spec["max_increase"])
        if "max_rel_increase" in spec:
            params["max_rel_increase"] = float(spec["max_rel_increase"])
        if params:
            result[col] = params
    return result
