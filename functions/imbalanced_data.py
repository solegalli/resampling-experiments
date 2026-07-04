"""Loaders for *strong-signal but severely imbalanced* datasets.

Companion to :mod:`functions.hard_data`. Where the hard datasets are difficult
because the features overlap the classes, these are the opposite: a standard
ensemble already discriminates the classes well (ROC-AUC > 0.9), but the
positive class is rare. They probe the regime that is *most* favourable to
resampling -- severe class imbalance with learnable signal -- to check whether
the special ensembles (RUSBoost, EasyEnsemble, BalancedRandomForest) add value
there, closing the main caveat of the hard-dataset comparison.

Datasets
--------
- ``htru2``      (UCI id 372)     ~9.2% positive -- pulsar detection
- ``creditcard`` (OpenML id 1597) ~0.17% positive -- credit-card fraud
  (full 284,807-row dataset)

Licenses (see ``docs/imbalanced_datasets.md``)
- htru2: CC BY 4.0 (commercial use permitted with attribution).
- creditcard: OpenML lists it as "Public"; the Kaggle distribution (mlg-ulb)
  uses the Open Data Commons Database Contents License (DbCL) v1.0, which permits
  commercial use. Provenance: Worldline / ULB Machine Learning Group.

Both datasets are clean numeric tables with no missing values. Preprocessing is
just the 70/30 split with ``random_state=0`` used elsewhere in the project.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASETS_IMBALANCED = ["htru2", "creditcard"]

CACHE_DIR = Path(__file__).resolve().parent.parent / ".data_cache"


def _cache_path(name):
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}.pkl"


def _split(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=0)


def _fetch_uci(name, dataset_id):
    cache = _cache_path(name)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=dataset_id)
    X = data.data.features.copy()
    y = data.data.targets.copy()
    with open(cache, "wb") as f:
        pickle.dump((X, y), f)
    return X, y


def _fetch_openml(name, dataset_id):
    cache = _cache_path(name)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=dataset_id, as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    with open(cache, "wb") as f:
        pickle.dump((X, y), f)
    return X, y


def _load_htru2():
    """Pulsar detection (UCI id 372). Positive class: pulsar (~9.2%)."""
    X, y = _fetch_uci("htru2", 372)
    target = y.iloc[:, 0].astype(int).to_numpy()
    return _split(X.copy(), target)


def _load_creditcard():
    """Credit-card fraud (OpenML id 1597). Positive class: fraud.

    Features are 28 PCA components plus the transaction amount; the OpenML
    version already excludes the raw time column. No missing values. The full
    284,807-row dataset is used (~0.17% positive).
    """
    X, y = _fetch_openml("creditcard", 1597)
    target = pd.to_numeric(y).astype(int).to_numpy()
    X = X.apply(pd.to_numeric).reset_index(drop=True)
    return _split(X, target)


_LOADERS = {
    "htru2": _load_htru2,
    "creditcard": _load_creditcard,
}


def load_imbalanced_dataset(dataset):
    """Load a strong-signal, severely imbalanced dataset by name.

    Parameters
    ----------
    dataset : str
        One of ``DATASETS_IMBALANCED``.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Features as DataFrames and binary targets as 1-D integer arrays, split
        70/30 with ``random_state=0`` (matching ``functions.data.load_dataset``).
    """
    if dataset not in _LOADERS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose from {DATASETS_IMBALANCED}."
        )
    return _LOADERS[dataset]()
