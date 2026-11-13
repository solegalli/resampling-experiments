"""
Tabular foundation models: fit on the training set (which only stores it as
context) and return the probability of the positive class for the test set.

TabPFN-2, TabICL and TabDPT run on Apple MPS. TabDPT needs float32 arrays,
because MPS does not support float64, and flash attention and compile turned
off, because they need CUDA.
"""

import os

import numpy as np

DEVICE = os.environ.get("TFM_DEVICE", "mps")

# TabPFN-2 was pre-trained on datasets with up to 10,000 training samples.
MAX_TRAIN_ROWS = {"tabpfn": 10_000}


def tabpfn(X_train, y_train, X_test, seed=0):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    # the package defaults to TabPFN-3, whose weights are non-commercial
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device=DEVICE, random_state=seed
    )
    return model.fit(X_train, y_train).predict_proba(X_test)[:, 1]


def tabicl(X_train, y_train, X_test, seed=0):
    from tabicl import TabICLClassifier

    model = TabICLClassifier(device=DEVICE, random_state=seed)
    return model.fit(X_train, y_train).predict_proba(X_test)[:, 1]


def tabdpt(X_train, y_train, X_test, seed=0):
    from tabdpt import TabDPTClassifier

    model = TabDPTClassifier(device=DEVICE, compile=False, use_flash=False, verbose=False)
    model.fit(np.asarray(X_train, dtype="float32"), np.asarray(y_train))
    X_test = np.asarray(X_test, dtype="float32")
    return model.ensemble_predict_proba(X_test, seed=seed)[:, 1]


FOUNDATION_MODELS = {"tabpfn": tabpfn, "tabicl": tabicl, "tabdpt": tabdpt}
