import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from functions.evaluation import evaluate_model_on_test_set


@pytest.fixture
def fitted_model():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        weights=[0.8, 0.2],
        random_state=10,
    )
    model = RandomForestClassifier(n_estimators=100, random_state=10)
    model.fit(X, y)
    return model


@pytest.fixture
def test_data():
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        weights=[0.8, 0.2],
        random_state=42,
    )
    return X, y


def test_output_length(fitted_model, test_data):
    X, y = test_data
    result, _ = evaluate_model_on_test_set(fitted_model, X, y)
    assert len(result) == 20


def test_output_is_list_of_floats(fitted_model, test_data):
    X, y = test_data
    result, _ = evaluate_model_on_test_set(fitted_model, X, y)
    assert all(isinstance(v, float) for v in result)


def test_std_values_are_non_negative(fitted_model, test_data):
    X, y = test_data
    result, _ = evaluate_model_on_test_set(fitted_model, X, y)
    # std values are at odd indices
    stds = result[1::2]
    assert all(s >= 0 for s in stds)


def test_probabilities_based_metrics_between_0_and_1(fitted_model, test_data):
    X, y = test_data
    result, _ = evaluate_model_on_test_set(fitted_model, X, y)
    mean_roc, mean_ap, mean_brier = result[0], result[2], result[14]
    assert 0 <= mean_roc <= 1
    assert 0 <= mean_ap <= 1
    assert 0 <= mean_brier <= 1


def test_threshold_between_0_and_1(fitted_model, test_data):
    X, y = test_data
    result, _ = evaluate_model_on_test_set(fitted_model, X, y)
    mean_thresh = result[18]
    assert 0 <= mean_thresh <= 1


def test_precision_recall_between_0_and_1(fitted_model, test_data):
    X, y = test_data
    result, _ = evaluate_model_on_test_set(fitted_model, X, y)
    mean_precision, mean_recall = result[4], result[6]
    assert 0 <= mean_precision <= 1
    assert 0 <= mean_recall <= 1


def test_output_order(fitted_model, test_data):
    X, y = test_data
    result, _ = evaluate_model_on_test_set(fitted_model, X, y)
    keys = [
        "roc",
        "roc_std",
        "ap",
        "ap_std",
        "precision",
        "precision_std",
        "recall",
        "recall_std",
        "f1",
        "f1_std",
        "mcc",
        "mcc_std",
        "ba",
        "ba_std",
        "brier",
        "brier_std",
        "gmean",
        "gmean_std",
        "thresh",
        "thresh_std",
    ]
    assert len(result) == len(keys)


def test_return_bootstrap_samples(fitted_model, test_data):
    X, y = test_data
    _, bootstrap = evaluate_model_on_test_set(fitted_model, X, y)
    assert isinstance(bootstrap, dict)
    assert len(bootstrap) == 10
    assert len(bootstrap['roc']) == 5
