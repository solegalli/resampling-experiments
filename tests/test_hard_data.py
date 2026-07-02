import urllib.error

import numpy as np
import pytest

from functions.hard_data import DATASETS_HARD, load_hard_dataset


@pytest.fixture(scope="module", params=DATASETS_HARD)
def loaded(request):
    """Load each hard dataset once; skip if it cannot be fetched (offline)."""
    try:
        return request.param, load_hard_dataset(request.param)
    except (urllib.error.URLError, ConnectionError) as exc:
        pytest.skip(f"could not fetch {request.param}: {exc}")


def test_returns_four_splits(loaded):
    _, output = loaded
    assert len(output) == 4


def test_target_is_binary(loaded):
    _, (_, _, y_train, y_test) = loaded
    assert set(np.unique(y_train)) <= {0, 1}
    assert set(np.unique(y_test)) <= {0, 1}


def test_features_have_no_missing_values(loaded):
    # scikit-learn estimators (RF, AdaBoost, GBM, the special ensembles) cannot
    # be fitted on NaNs, so the loaders must impute/encode them away.
    _, (X_train, X_test, _, _) = loaded
    assert not np.isnan(np.asarray(X_train, dtype="float64")).any()
    assert not np.isnan(np.asarray(X_test, dtype="float64")).any()


def test_sizes_consistent(loaded):
    _, (X_train, X_test, y_train, y_test) = loaded
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_datasets_are_imbalanced(loaded):
    name, (_, _, y_train, _) = loaded
    pos_rate = y_train.mean()
    assert 0 < pos_rate < 0.5, f"{name} is not imbalanced (pos_rate={pos_rate:.2f})"


def test_diabetes130_drops_id_columns():
    """encounter_id and patient_nbr are per-visit/patient identifiers, not
    predictors, and must not reach the model."""
    try:
        X_train, X_test, _, _ = load_hard_dataset("diabetes130")
    except (urllib.error.URLError, ConnectionError) as exc:
        pytest.skip(f"could not fetch diabetes130: {exc}")
    for col in ("encounter_id", "patient_nbr"):
        assert col not in X_train.columns
        assert col not in X_test.columns


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_hard_dataset("not_a_dataset")
