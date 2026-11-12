"""
Train CatBoost, LightGBM and XGBoost after Random Undersampling (RUS) and Condensed
Nearest Neighbour (CNN), tuned with Optuna and early stopping (see
functions/cv_boosting.py), and evaluate them on the test set.

These models are compared with the same models tuned with successive halving
(models/rus and models/cnn). Only datasets where the best baseline model (no
resampling) has a ROC-AUC below 0.997 are used.

Models and results are saved in models/rus-es and models/cnn-es.
"""

import pickle
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
from tqdm import tqdm

from configs.hyperparams import hyperparam_ensemble_dict
from configs.undersamplers import cnn, rus
from functions.cv_undersamplers import train_model_w_undersampling, undersample_data
from functions.data import DATASETS_LS, load_dataset
from functions.evaluation import evaluate_model_on_test_set

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

UNDERSAMPLERS = {"rus": (rus, False), "cnn": (cnn, True)}  # name: (sampler, scale)
BOOSTERS = ["cat", "lgbm", "xgb"]

# ROC-AUC is the first value of each model's results
with open(REPO_ROOT / "models" / "ensembles" / "results", "rb") as f:
    baseline = pickle.load(f)

datasets = [
    d for d in DATASETS_LS if max(r[0] for r in baseline[d].values()) < 0.997
]

for name, (sampler, scale) in UNDERSAMPLERS.items():
    output_dir = REPO_ROOT / "models" / f"{name}-es"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_dict, folds_dict = {}, {}

    for dataset in tqdm(datasets, desc=name):
        X_train, X_test, y_train, y_test = load_dataset(dataset)
        xtrainu, ytrainu, xtest, ytest, Xu, yu, _ = undersample_data(
            sampler, X_train, y_train, scale=scale
        )

        scores_dict[dataset], folds_dict[dataset] = {}, {}

        for booster in BOOSTERS:
            start_time = time.time()
            model = train_model_w_undersampling(
                booster,
                hyperparam_ensemble_dict[f"{booster}_params"],
                xtrainu,
                ytrainu,
                xtest,
                ytest,
                Xu,
                yu,
            )
            model.fit_time = time.time() - start_time
            joblib.dump(model, output_dir / f"{dataset}_{booster}_{name}.pkl")

            results, fold_metrics = evaluate_model_on_test_set(model, X_test, y_test)
            scores_dict[dataset][f"{booster}_{name}"] = results
            folds_dict[dataset][f"{booster}_{name}"] = fold_metrics

    with open(output_dir / "results", "wb") as fp:
        pickle.dump(scores_dict, fp)

    with open(output_dir / "results_folds", "wb") as fp:
        pickle.dump(folds_dict, fp)
