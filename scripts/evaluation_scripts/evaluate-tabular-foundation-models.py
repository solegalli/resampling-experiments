"""
Evaluate open tabular foundation models (TabPFN-2, TabICL and TabDPT, see
functions/foundation_models.py) on the datasets where the best ensemble has a
ROC-AUC below 0.95.

These models are pre-trained: ``fit`` only stores the training set, which the
model conditions on when it predicts (in-context learning). There is nothing to
tune and nothing to save, so each model is fitted and evaluated here directly.
Following TabArena, TabPFN-2 is only run on datasets within its limit of 10,000
training samples.

Results are saved in models/<model> with the same structure as the ensembles,
plus the time to fit and predict per dataset.
"""

import pickle
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from functions.data import load_dataset
from functions.evaluation import evaluate_probabilities
from functions.foundation_models import FOUNDATION_MODELS, MAX_TRAIN_ROWS

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

# ROC-AUC is the first value of each model's results
with open(REPO_ROOT / "models" / "ensembles" / "results", "rb") as f:
    baseline = pickle.load(f)

datasets = [d for d in baseline if max(r[0] for r in baseline[d].values()) < 0.95]
data = {d: load_dataset(d) for d in datasets}

# smallest datasets first, so partial runs are still useful
datasets.sort(key=lambda d: len(data[d][2]))

for model_name, fit_predict in FOUNDATION_MODELS.items():
    output_dir = REPO_ROOT / "models" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_dict, folds_dict, times_dict = {}, {}, {}

    for dataset in tqdm(datasets, desc=model_name):
        X_train, X_test, y_train, y_test = data[dataset]

        if len(y_train) > MAX_TRAIN_ROWS.get(model_name, float("inf")):
            continue

        start_time = time.time()
        prob = fit_predict(X_train, y_train, X_test)
        times_dict[dataset] = time.time() - start_time

        results, fold_metrics = evaluate_probabilities(y_test, prob)
        scores_dict[dataset] = {model_name: results}
        folds_dict[dataset] = {model_name: fold_metrics}

        # save after every dataset, so a stopped run keeps its results
        for name, obj in [
            ("results", scores_dict),
            ("results_folds", folds_dict),
            ("times", times_dict),
        ]:
            with open(output_dir / name, "wb") as fp:
                pickle.dump(obj, fp)
