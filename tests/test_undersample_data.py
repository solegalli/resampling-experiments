import numpy as np
import pytest
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from functions.instance_hardness import InstanceHardnessUnderSampler

from functions.cv_undersamplers import undersample_data


@pytest.fixture
def imbalanced_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        weights=[0.8, 0.2],
        random_state=10,
    )
    return X, y


def test_returns_seven_elements(imbalanced_data):
    X, y = imbalanced_data
    output = undersample_data(RandomUnderSampler(random_state=10), X, y, scale=False)
    assert len(output) == 7


def test_three_folds(imbalanced_data):
    X, y = imbalanced_data
    xtrainu, ytrainu, xtest, ytest, _, _, _ = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    assert len(xtrainu) == 3
    assert len(ytrainu) == 3
    assert len(xtest) == 3
    assert len(ytest) == 3


def test_train_test_sizes_consistent(imbalanced_data):
    X, y = imbalanced_data
    xtrainu, ytrainu, xtest, ytest, _, _, _ = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    for i in range(3):
        assert len(xtrainu[i]) == len(ytrainu[i])
        assert len(xtest[i]) == len(ytest[i])
        assert len(xtrainu[i]) + len(xtest[i]) <= len(X)


def test_undersampled_folds_are_balanced(imbalanced_data):
    X, y = imbalanced_data
    xtrainu, ytrainu, _, _, _, _, _ = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    for ytrain in ytrainu:
        unique, counts = np.unique(ytrain, return_counts=True)
        assert counts[0] == counts[1]


def test_full_undersampled_set_is_balanced(imbalanced_data):
    X, y = imbalanced_data
    _, _, _, _, _, yu, _ = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    unique, counts = np.unique(yu, return_counts=True)
    assert counts[0] == counts[1]


def test_test_folds_not_undersampled(imbalanced_data):
    X, y = imbalanced_data
    _, _, _, ytest, _, _, _ = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    for yt in ytest:
        unique, counts = np.unique(yt, return_counts=True)
        assert counts[0] > counts[1]


def test_xu_smaller_than_original(imbalanced_data):
    X, y = imbalanced_data
    _, _, _, _, Xu, yu, _ = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    assert len(Xu) < len(X)
    assert len(yu) < len(y)


def test_stats_keys(imbalanced_data):
    X, y = imbalanced_data
    _, _, _, _, _, _, stats = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    assert set(stats.keys()) == {
        "original_size",
        "undersampled_size",
        "removed",
        "removed_pct",
    }


def test_stats_values_consistent(imbalanced_data):
    X, y = imbalanced_data
    _, _, _, _, Xu, yu, stats = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    assert stats["original_size"] == len(X)
    assert stats["undersampled_size"] == len(Xu)
    assert stats["removed"] == len(X) - len(Xu)
    assert stats["removed_pct"] == round((1 - len(Xu) / len(X)) * 100, 1)


def test_scale_true_returns_original_scale(imbalanced_data):
    X, y = imbalanced_data
    xtrainu, ytrainu, xtest, ytest, Xu, yu, stats = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=True
    )

    # check full undersampled set is not scaled
    assert Xu.max() > 1 or Xu.min() < 0

    # check each fold is not scaled
    for fold in xtrainu:
        assert fold.max() > 1 or fold.min() < 0


def test_instance_hardness_undersampler(imbalanced_data):
    X, y = imbalanced_data
    sampler = InstanceHardnessUnderSampler(threshold=0.4, random_state=10)
    xtrainu, ytrainu, xtest, ytest, Xu, yu, stats = undersample_data(
        sampler, X, y, scale=False
    )
    assert len(Xu) < len(X)
    assert len(yu) < len(y)
    assert "original_size" in stats
    assert "undersampled_size" in stats
