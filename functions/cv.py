import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV


def get_sample_weights(y_train):
    """
    Compute sample weight arrays for cost-sensitive learning.

    Returns a dict mapping integer weight -> sample weight array, where the
    minority class (label 1) receives the weight and the majority class gets 1.
    Candidates are IR, IR//2, and IR//3 (integers >= 2, duplicates dropped).
    """
    IR = int(round((y_train == 0).sum() / (y_train == 1).sum()))
    candidates = dict.fromkeys(w for w in (IR, IR // 2, IR // 3) if w >= 2)
    return {w: np.where(y_train == 1, w, 1).astype(int) for w in candidates}


def train_model(
    estimator,
    params,
    X_train,
    y_train,
    scoring="roc_auc",
    refit=True,
    n_jobs=-1,
    sample_weight=None,
):
    """
    Train classifier with hyperparameter tuning
    using successive halving and without undersampling.

    ``n_jobs`` controls the parallelism of the search itself (defaults to -1, all
    cores). Set ``n_jobs=1`` to run the candidate search sequentially; this avoids
    a nested-parallelism deadlock that can occur on macOS when the search workers
    and a parallel estimator both spawn joblib/loky processes. It does not change
    the fitted models, only how they are scheduled.
    """

    # CatBoostClassifier does not recognize n_estimators as hyperparameter.
    # https://github.com/scikit-learn/scikit-learn/issues/19844

    search = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_candidates="exhaust",  # the number of candidates to evaluate at the first iteration
        factor=3,  # only a third of the candidates are promoted
        resource="n_estimators",  # the limiting resource
        max_resources=1000,  # max number of trees (or samples)
        min_resources=10,  # min number of trees (or samples)
        scoring=scoring,  # proper scoring function (ensures probabilistic distribution)
        cv=3,  # uses StratifiedKFold by default
        random_state=10,
        refit=refit,
        n_jobs=n_jobs,
    )
    # EasyEnsembleClassifier doesn not accept sample_weight, 
    # so we need to handle it separately.
    if type(estimator).__name__ == "EasyEnsembleClassifier":
        search.fit(X_train, y_train)
    else:
        search.fit(X_train, y_train, sample_weight=sample_weight)
    return search


def train_basic_model(
    estimator, params, X_train, y_train, scoring="roc_auc", refit=True
):
    """
    Train classifier with hyperparameter tuning
    using randomized search and without undersampling.
    """

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=20,
        scoring=scoring,
        cv=3,
        random_state=10,
        refit=refit,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    return search
