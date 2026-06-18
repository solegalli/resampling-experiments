import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from functions.instance_hardness import InstanceHardnessUnderSampler


@pytest.fixture
def imbalanced_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        weights=[0.8, 0.2],
        random_state=42,
    )
    return X, y


def test_removes_majority_class_samples(imbalanced_data):
    X, y = imbalanced_data
    sampler = InstanceHardnessUnderSampler(threshold=0.4, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)

    # Original counts
    _, counts_orig = np.unique(y, return_counts=True)
    # Resampled counts
    _, counts_res = np.unique(y_res, return_counts=True)

    # Majority class (0) should have fewer samples
    assert counts_res[0] < counts_orig[0]
    # Minority class (1) should have the same number of samples
    assert counts_res[1] == counts_orig[1]


def test_raises_error_for_non_binary(imbalanced_data):
    X, y = imbalanced_data
    # Convert to multiclass
    y_multi = y.copy()
    y_multi[y == 1] = 2
    y_multi[0:10] = 1  # now classes 0, 1, 2

    sampler = InstanceHardnessUnderSampler()
    with pytest.raises(ValueError, match="only supports binary classification"):
        sampler.fit_resample(X, y_multi)


def test_custom_estimator(imbalanced_data):
    X, y = imbalanced_data
    sampler = InstanceHardnessUnderSampler(
        estimator=LogisticRegression(), threshold=0.4, random_state=42
    )
    X_res, y_res = sampler.fit_resample(X, y)
    assert len(X_res) < len(X)


def test_instance_hardness_removal_logic(imbalanced_data):
    X, y = imbalanced_data
    threshold = 0.7
    estimator = LogisticRegression(random_state=42)

    # Manually calculate probabilities
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    hardness_scores = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X, y):
        fold_estimator = clone(estimator)
        fold_estimator.fit(X[train_idx], y[train_idx])
        proba = fold_estimator.predict_proba(X[test_idx])
        hardness_scores[test_idx] = proba[:, 1]

    # Run sampler
    sampler = InstanceHardnessUnderSampler(
        estimator=estimator, threshold=threshold, cv=5, random_state=42
    )
    X_res, y_res = sampler.fit_resample(X, y)

    # Compare scores
    np.testing.assert_array_almost_equal(hardness_scores, sampler.hardness_scores_)

    # Verify removed instances
    # Identify majority class
    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    majority_indices = np.where(y == majority_class)[0]

    expected_removed = majority_indices[hardness_scores[majority_indices] > threshold]

    # Sort indices to compare
    np.testing.assert_array_equal(
        np.sort(expected_removed), np.sort(sampler.removed_indices_)
    )
