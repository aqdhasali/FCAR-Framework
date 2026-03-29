"""
Unit tests for src.modeling.preprocess — ColumnTransformer builder.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer

from src.modeling.preprocess import build_preprocessor


@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0],
        "num_b": [10, 20, 30, 40],
        "cat_a": ["X", "Y", "X", "Z"],
        "cat_b": pd.Categorical(["A", "B", "A", "B"]),
    })


class TestBuildPreprocessor:

    def test_returns_column_transformer(self, sample_df):
        pre = build_preprocessor(sample_df)
        assert isinstance(pre, ColumnTransformer)

    def test_fit_transform_shape(self, sample_df):
        pre = build_preprocessor(sample_df)
        out = pre.fit_transform(sample_df)
        if issparse(out):
            out = out.toarray()
        # 2 numeric cols + one-hot for cat_a (3 unique) + cat_b (2 unique) = 7
        assert out.shape[0] == 4
        assert out.shape[1] == 7

    def test_numeric_cols_scaled(self, sample_df):
        pre = build_preprocessor(sample_df)
        pre.fit(sample_df)
        num_pipe = pre.named_transformers_["num"]
        scaler = num_pipe.named_steps["scaler"]
        assert hasattr(scaler, "scale_")
        assert len(scaler.scale_) == 2  # num_a, num_b

    def test_categorical_onehot_encoded(self, sample_df):
        pre = build_preprocessor(sample_df)
        pre.fit(sample_df)
        cat_pipe = pre.named_transformers_["cat"]
        enc = cat_pipe.named_steps["onehot"]
        # Should have categories for cat_a and cat_b
        assert len(enc.categories_) == 2

    def test_handles_unknown_categories(self, sample_df):
        pre = build_preprocessor(sample_df)
        pre.fit(sample_df)
        new = sample_df.copy()
        new.loc[0, "cat_a"] = "NEVER_SEEN"
        # Should not raise; handle_unknown="ignore"
        out = pre.transform(new)
        assert out is not None

    def test_all_numeric(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        pre = build_preprocessor(df)
        out = pre.fit_transform(df)
        if issparse(out):
            out = out.toarray()
        assert out.shape == (2, 2)

    def test_all_categorical(self):
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": ["d", "e", "f"]})
        pre = build_preprocessor(df)
        out = pre.fit_transform(df)
        if issparse(out):
            out = out.toarray()
        assert out.shape[0] == 3
        assert out.shape[1] == 6  # 3 + 3 categories

    def test_nan_handling(self):
        df = pd.DataFrame({
            "num": [1.0, np.nan, 3.0],
            "cat": ["A", None, "B"],
        })
        pre = build_preprocessor(df)
        out = pre.fit_transform(df)
        if issparse(out):
            out = out.toarray()
        # Should not have NaN after imputation
        assert not np.isnan(out).any()

    def test_feature_names_out(self, sample_df):
        pre = build_preprocessor(sample_df)
        pre.fit(sample_df)
        names = pre.get_feature_names_out()
        assert len(names) == 7
        # Should have num__ and cat__ prefixes
        assert any("num__" in n for n in names)
        assert any("cat__" in n for n in names)
