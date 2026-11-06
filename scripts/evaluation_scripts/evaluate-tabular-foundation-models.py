"""
Evaluate open-source tabular foundation models on various imbalanced datasets.

Tabular foundation models are pre-trained: calling ``fit`` does not train
anything, it only registers the training set that the model conditions on at
inference time (in-context learning). There is nothing worth persisting to
disk, so unlike the other evaluation scripts this one does not load a pickled
estimator. Instead it loops over every dataset, splits it with the shared
``load_dataset`` helper, "fits" a fresh model on the train set, evaluates it on
the test set with bootstrapped samples via ``evaluate_model_on_test_set``, and
stores the results as pickle files under ``models/<model_name>`` using the same
structure as ``evaluate-ensembles.py`` so the numbers can be compared against
the ensembles.

Models evaluated (both fully open source, weights on the Hugging Face Hub and
downloaded automatically on first use):

* ``tabicl``  -- TabICL v2 (soda-inria), ``pip install tabicl``
* ``tabdpt``  -- TabDPT (Layer6 AI),      ``pip install tabdpt``

Both are scikit-learn compatible (``fit`` / ``predict_proba``), so no API key
or network account is required. Run on a GPU if you have one: inference on the
larger datasets is slow on CPU.
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm

from functions.data import DATASETS_LS, load_dataset
from functions.evaluation import evaluate_model_on_test_set

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

# "cuda", "mps" or "cpu"; auto-detected when left as None.
DEVICE = os.environ.get("TFM_DEVICE") or None


def _resolve_device():
    if DEVICE is not None:
        return DEVICE
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SklearnFoundationModel:
    """
    Adapter that gives a tabular foundation model the exact surface
    ``evaluate_model_on_test_set`` needs.

    TabICL and TabDPT are purely numeric in-context learners (they do not use
    column-name semantics), and TabDPT in fact rejects anything that is not a
    NumPy array, so everything is coerced to a float array here. ``predict_proba``
    is delegated to the wrapped estimator; its columns follow ``classes_`` order,
    so ``[:, 1]`` is the positive class as the evaluation helper assumes.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.classes_ = None

    @staticmethod
    def _as_array(X):
        return np.asarray(X, dtype="float64")

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.estimator.fit(self._as_array(X), y)
        return self

    def predict(self, X):
        return np.asarray(self.estimator.predict(self._as_array(X)))

    def predict_proba(self, X):
        p = np.asarray(self.estimator.predict_proba(self._as_array(X)), dtype="float64")
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        return p


def build_tabicl():
    from tabicl import TabICLClassifier

    return SklearnFoundationModel(
        TabICLClassifier(random_state=0, device=_resolve_device())
    )


def build_tabdpt():
    from tabdpt import TabDPTClassifier

    # compile/flash-attention are GPU features; disable so CPU/MPS runs work.
    return SklearnFoundationModel(
        TabDPTClassifier(device=_resolve_device(), compile=False, use_flash=False)
    )


# model name -> zero-arg factory returning a fresh, unfitted estimator
MODEL_BUILDERS = {
    "tabicl": build_tabicl,
    "tabdpt": build_tabdpt,
}


for model_name, build in MODEL_BUILDERS.items():
    output_dir = REPO_ROOT / "models" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_dict = {}
    folds_dict = {}

    for dataset in tqdm(DATASETS_LS, desc=model_name):
        X_train, X_test, y_train, y_test = load_dataset(dataset)

        try:
            model = build()
            model.fit(X_train, y_train)
            results, fold_metrics = evaluate_model_on_test_set(model, X_test, y_test)
        except Exception as exc:  # noqa: BLE001 - one dataset failing is not fatal
            tqdm.write(
                f"[{model_name}] skipping {dataset}: {type(exc).__name__}: {exc}"
            )
            continue

        scores_dict[dataset] = {model_name: results}
        folds_dict[dataset] = {model_name: fold_metrics}

        # Persist after every dataset so a late failure keeps completed results.
        with open(output_dir / "results", "wb") as fp:
            pickle.dump(scores_dict, fp)

        with open(output_dir / "results_folds", "wb") as fp:
            pickle.dump(folds_dict, fp)
