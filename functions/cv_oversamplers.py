"""
Cross-validation utilities for training models on pre-oversampled data.

The key design choice here is to oversample once before hyperparameter search
rather than embedding the oversampler inside a pipeline. This avoids re-running
the oversampler on every candidate and resource level during successive halving,
at the cost of some statistical purity.

Oversampling is applied only to training folds; test folds always reflect the
original class distribution.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold


def oversample_data(oversampler, X, y):
    """
    Apply an oversampler to a 3-fold stratified split and return the results.

    Oversampling is applied to each training fold independently, so the
    test folds always reflect the original class distribution. A final
    oversampled version of the full training set is also returned for
    refitting the best model after hyperparameter search.

    Parameters
    ----------
    oversampler : imblearn sampler
        An unfitted imblearn oversampler with a `fit_resample` method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.

    Returns
    -------
    xtraino : list of 3 arrays
        Oversampled feature matrices for each training fold.
    ytraino : list of 3 arrays
        Oversampled target vectors for each training fold.
    xtest : list of 3 arrays
        Original (non-oversampled) feature matrices for each test fold.
    ytest : list of 3 arrays
        Original target vectors for each test fold.
    Xo : array
        Oversampled feature matrix for the full training set.
    yo : array
        Oversampled target vector for the full training set.
    stats : dict
        Summary of oversampling effect on the full training set:
        original_size, oversampled_size, added, added_pct.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)

    xtrain, ytrain, xtest, ytest = [], [], [], []

    for train_index, test_index in skf.split(X, y):
        xtrain.append(X[train_index])
        ytrain.append(y[train_index])
        xtest.append(X[test_index])
        ytest.append(y[test_index])

    xtraino, ytraino = [], []

    for data, target in zip(xtrain, ytrain):
        datao, targeto = oversampler.fit_resample(data, target)
        xtraino.append(datao)
        ytraino.append(targeto)

    Xo, yo = oversampler.fit_resample(X, y)

    return xtraino, ytraino, xtest, ytest, Xo, yo
