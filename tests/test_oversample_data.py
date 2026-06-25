import numpy as np
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.datasets import make_classification

from functions.cv_oversamplers import oversample_data


@pytest.fixture
def imbalanced_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        weights=[0.8, 0.2],
        random_state=10,
    )
    return X, y


def test_returns_six_elements(imbalanced_data):
    X, y = imbalanced_data
    output = oversample_data(RandomOverSampler(random_state=10), X, y)
    assert len(output) == 6


def test_three_folds(imbalanced_data):
    X, y = imbalanced_data
    xtraino, ytraino, xtest, ytest, _, _ = oversample_data(
        RandomOverSampler(random_state=10), X, y
    )
    assert len(xtraino) == 3
    assert len(ytraino) == 3
    assert len(xtest) == 3
    assert len(ytest) == 3


def test_train_test_sizes_consistent(imbalanced_data):
    X, y = imbalanced_data
    xtraino, ytraino, xtest, ytest, _, _ = oversample_data(
        RandomOverSampler(random_state=10), X, y
    )
    for i in range(3):
        assert len(xtraino[i]) == len(ytraino[i])
        assert len(xtest[i]) == len(ytest[i])
        assert len(xtraino[i]) >= len(X) - len(xtest[i])


def test_oversampled_folds_are_balanced(imbalanced_data):
    X, y = imbalanced_data
    xtraino, ytraino, _, _, _, _ = oversample_data(
        RandomOverSampler(random_state=10), X, y
    )
    for ytrain in ytraino:
        unique, counts = np.unique(ytrain, return_counts=True)
        assert counts[0] == counts[1]


def test_full_oversampled_set_is_balanced(imbalanced_data):
    X, y = imbalanced_data
    _, _, _, _, _, yo = oversample_data(
        RandomOverSampler(random_state=10), X, y
    )
    unique, counts = np.unique(yo, return_counts=True)
    assert counts[0] == counts[1]


def test_test_folds_not_oversampled(imbalanced_data):
    X, y = imbalanced_data
    _, _, _, ytest, _, _ = oversample_data(
        RandomOverSampler(random_state=10), X, y
    )
    for yt in ytest:
        unique, counts = np.unique(yt, return_counts=True)
        assert counts[0] > counts[1]


def test_xo_larger_than_original(imbalanced_data):
    X, y = imbalanced_data
    _, _, _, _, Xo, yo = oversample_data(
        RandomOverSampler(random_state=10), X, y
    )
    assert len(Xo) > len(X)
    assert len(yo) > len(y)


def test_smote_oversampler(imbalanced_data):
    X, y = imbalanced_data
    xtraino, ytraino, xtest, ytest, Xo, yo = oversample_data(
        SMOTE(random_state=10), X, y
    )
    assert len(Xo) > len(X)
    assert len(yo) > len(y)
    unique, counts = np.unique(yo, return_counts=True)
    assert counts[0] == counts[1]
