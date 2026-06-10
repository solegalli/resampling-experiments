"""
Cross-validation utilities for training models on pre-undersampled data.

The key design choice here is to undersample once before hyperparameter search
rather than embedding the undersampler inside a pipeline. This avoids re-running
slow undersamplers (e.g., CNN) on every candidate and resource level during
successive halving, at the cost of some statistical purity.

https://stackoverflow.com/questions/79748461/how-to-pass-pre-computed-folds-to-successivehalving-in-sklearn?
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


def undersample_data(undersampler, X, y, scale):
    """
    Apply an undersampler to a 3-fold stratified split and return the results.

    Undersampling is applied to each training fold independently, so the
    test folds always reflect the original class distribution. A final
    undersampled version of the full training set is also returned for
    refitting the best model after hyperparameter search.

    Parameters
    ----------
    undersampler : imblearn sampler
        A fitted/unfitted imblearn undersampler with a `fit_resample` method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    scale : bool, default=False
        If True, apply MinMaxScaler before undersampling so that distance-based
        undersamplers (e.g., CNN) operate on comparable feature ranges. The
        returned data is always in the original scale.

    Returns
    -------
    If return_index=False (default):
        xtrainu : list of 3 arrays
            Undersampled feature matrices for each training fold.
        ytrainu : list of 3 arrays
            Undersampled target vectors for each training fold.
        xtest : list of 3 arrays
            Original (non-undersampled) feature matrices for each test fold.
        ytest : list of 3 arrays
            Original target vectors for each test fold.
        Xu : array
            Undersampled feature matrix for the full training set.
        yu : array
            Undersampled target vector for the full training set.
        stats : dict
            Summary of undersampling effect on the full training set:
            original_size, undersampled_size, removed, removed_pct.
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

    xtrainu, ytrainu = [], []

    for data, target in zip(xtrain, ytrain):

        if scale is True:
            # Scale only to guide the undersampler (e.g., distance-based
            # methods like CNN need consistent feature ranges), but return
            # the original-scale data so models are trained on raw values.
            undersampler.fit_resample(MinMaxScaler().fit_transform(data), target)
            datau = data[undersampler.sample_indices_]
            targetu = target[undersampler.sample_indices_]
        else:
            datau, targetu = undersampler.fit_resample(data, target)

        xtrainu.append(datau)
        ytrainu.append(targetu)

    # also return undersampled train set for training final model
    if scale is True:
        undersampler.fit_resample(MinMaxScaler().fit_transform(X), y)
        Xu = X[undersampler.sample_indices_]
        yu = y[undersampler.sample_indices_]
    else:
        Xu, yu = undersampler.fit_resample(X, y)

    stats = {
        "original_size": len(X),
        "undersampled_size": len(Xu),
        "removed": len(X) - len(Xu),
        "removed_pct": round((1 - len(Xu) / len(X)) * 100, 1),
    }

    return xtrainu, ytrainu, xtest, ytest, Xu, yu, stats


def train_model_w_undersampling(
    model, params, xtrainu, ytrainu, xtest, ytest, Xu, yu, scoring="roc_auc", n_iter=100
):
    """
    Tune and train a model on pre-undersampled data.

    Undersampling is applied once upfront (via `undersample_data`) rather than
    inside a pipeline, so slow undersamplers like CNN are not re-run on every
    candidate during hyperparameter search.
    The tuning uses a manual successive halving approach:
      - Round 1: 100 candidates with n_estimators=10, scored on 3 folds
      - Round 2: top 10 candidates with n_estimators=300, scored on 3 folds
      - Final:   best candidate retrained with n_estimators=800 on full data

    Parameters
    ----------
    model : estimator
        An unfitted sklearn-compatible classifier with `n_estimators`.
    params : dict
        Hyperparameter search space passed to ParameterSampler.
    xtrainu, ytrainu : list of arrays
        Per-fold undersampled training sets produced by `undersample_data`.
    xtest, ytest : list of arrays
        Per-fold held-out test sets produced by `undersample_data`.
    Xu, yu : arrays
        Full undersampled training set for final refit.
    scoring : str, default="roc_auc"
        Scoring metric to use for hyperparameter tuning.
        One of "log_loss", "roc_auc", or "brier".
    """

    valid_scorings = ("log_loss", "roc_auc", "brier")
    if scoring not in valid_scorings:
        raise ValueError(f"scoring must be one of {valid_scorings}, got '{scoring}'")

    param_list = list(ParameterSampler(params, n_iter=n_iter, random_state=42))

    # --- Round 1: screen all candidates with n_estimators=10 ---
    results = []
    for params_ in param_list:
        fold_scores = []
        temp_params = params_.copy()
        temp_params["n_estimators"] = 10

        for i in range(3):
            clf = clone(model)
            clf.set_params(**temp_params)
            clf.fit(xtrainu[i], ytrainu[i])
            y_pred = clf.predict_proba(xtest[i])[:, 1]

            if scoring == "log_loss":
                fold_scores.append(log_loss(ytest[i], y_pred))
            elif scoring == "roc_auc":
                fold_scores.append(roc_auc_score(ytest[i], y_pred))
            elif scoring == "brier":
                fold_scores.append(brier_score_loss(ytest[i], y_pred))

        avg_score = np.mean(fold_scores)
        results.append((params_, avg_score))

    # Promote top 10
    if scoring == "roc_auc":
        top10 = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    else:
        top10 = sorted(results, key=lambda x: x[1])[:10]

    # --- Round 2: re-evaluate top 10 with n_estimators=300 ---
    results_r2 = []
    for params_, _ in top10:
        fold_scores = []
        temp_params = params_.copy()
        temp_params["n_estimators"] = 300

        for i in range(3):
            clf = clone(model)
            clf.set_params(**temp_params)
            clf.fit(xtrainu[i], ytrainu[i])
            y_pred = clf.predict_proba(xtest[i])[:, 1]

            if scoring == "log_loss":
                fold_scores.append(log_loss(ytest[i], y_pred))
            elif scoring == "roc_auc":
                fold_scores.append(roc_auc_score(ytest[i], y_pred))
            elif scoring == "brier":
                fold_scores.append(brier_score_loss(ytest[i], y_pred))

        avg_score = np.mean(fold_scores)
        results_r2.append((params_, avg_score))

    # Select best configuration
    if scoring == "roc_auc":
        best_params, best_score = max(results_r2, key=lambda x: x[1])
    else:
        best_params, best_score = min(results_r2, key=lambda x: x[1])

    # --- Final refit with n_estimators=800 on full undersampled data ---
    best_params_final = best_params.copy()
    best_params_final["n_estimators"] = 800

    model = clone(model)
    model.set_params(**best_params_final)
    model.fit(Xu, yu)

    return model
