import numpy as np
import pandas as pd
import pytest

from functions.data import DATASETS_LS, load_dataset


def check_quality(X, name, dataset):
    """Checks that the feature matrix X has no missing values and is numeric."""
    if isinstance(X, np.ndarray):
        assert not np.isnan(X).any(), f"Missing values found in {name} for {dataset}"
        assert np.issubdtype(
            X.dtype, np.number
        ), f"Non-numeric features found in {name} for {dataset}, dtype: {X.dtype}"
    elif isinstance(X, pd.DataFrame):
        assert (
            not X.isnull().any().any()
        ), f"Missing values found in {name} for {dataset}"
        non_numeric = X.select_dtypes(exclude=["number"])
        assert (
            non_numeric.empty
        ), f"Non-numeric features found in {name} for {dataset}: {non_numeric.columns.tolist()}"
    else:
        raise TypeError(f"Unknown type {type(X)} for {name} in {dataset}")


def check_target_quality(y, name, dataset):
    """Checks that the target y contains only 0s and 1s, and both are present."""
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    unique_values = np.unique(y)
    assert np.isin(
        unique_values, [0, 1]
    ).all(), f"Values other than 0 and 1 found in {name} for {dataset}: {unique_values}"
    assert (
        0 < np.mean(y) < 1
    ), f"Target variable {name} in {dataset} does not contain both 0 and 1 classes."


@pytest.mark.parametrize("dataset", DATASETS_LS)
def test_dataset_loading_and_quality(dataset):
    """Loads a dataset and validates that X and y meet quality standards."""
    X_train, X_test, y_train, y_test = load_dataset(dataset)

    check_quality(X_train, "X_train", dataset)
    check_quality(X_test, "X_test", dataset)
    check_target_quality(y_train, "y_train", dataset)
    check_target_quality(y_test, "y_test", dataset)
