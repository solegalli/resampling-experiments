import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from functions.cv import get_sample_weights, train_model


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=10)
    return X, y


@pytest.fixture
def fitted_search(classification_data):
    X_train, y_train = classification_data
    estimator = RandomForestClassifier(random_state=10)
    params = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": range(2, 20),
        "min_samples_leaf": range(2, 20),
        "max_features": ["log2", "sqrt"],
    }
    return train_model(estimator, params, X_train, y_train)


def test_n_resources(fitted_search):
    # the number of n_estimators used at each iteration
    assert list(fitted_search.n_resources_) == [10, 30, 90, 270, 810]


def test_n_iterations(fitted_search):
    assert fitted_search.n_iterations_ == 5


def test_n_candidates(fitted_search):
    # the number of hyperparameter combinations tested at each iteration
    assert list(fitted_search.n_candidates_) == [100, 34, 12, 4, 2]


def test_sample_weight_default(mocker, classification_data):
    X_train, y_train = classification_data
    estimator = RandomForestClassifier(random_state=10)
    params = {"max_depth": [2, 3]}
    mock_search = mocker.patch("functions.cv.HalvingRandomSearchCV")

    train_model(estimator, params, X_train, y_train)

    mock_search.return_value.fit.assert_called_once_with(
        X_train, y_train, sample_weight=None
    )


def test_sample_weight_passed(mocker, classification_data):
    X_train, y_train = classification_data
    sample_weight = np.ones(len(y_train))
    estimator = RandomForestClassifier(random_state=10)
    params = {"max_depth": [2, 3]}
    mock_search = mocker.patch("functions.cv.HalvingRandomSearchCV")

    train_model(estimator, params, X_train, y_train, sample_weight=sample_weight)

    mock_search.return_value.fit.assert_called_once_with(
        X_train, y_train, sample_weight=sample_weight
    )

@pytest.mark.parametrize(
    "IR, expected_weights",
    [
        (2, {2}),
        (3, {3}),
        (10, {10, 5, 3}),
        (100, {100, 50, 33}),
    ],
)
def test_get_sample_weights_keys(IR, expected_weights):
    y_train = np.array([0] * (IR * 100) + [1] * 100)
    result = get_sample_weights(y_train)
    assert set(result.keys()) == expected_weights


@pytest.mark.parametrize("IR", [2, 3, 10, 100])
def test_get_sample_weights_arrays(IR):
    n_minority = 100
    y_train = np.array([0] * (IR * n_minority) + [1] * n_minority)
    result = get_sample_weights(y_train)
    for w, sw in result.items():
        assert sw.shape == y_train.shape
        assert (sw[y_train == 1] == w).all()
        assert (sw[y_train == 0] == 1).all()
        assert sw.dtype == int