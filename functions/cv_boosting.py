"""
Hyperparameter tuning of CatBoost, LightGBM and XGBoost with their native
training APIs and Optuna, optimising ROC-AUC.

Each trial trains the candidate on 3 folds. On every fold, boosting stops when
the validation AUC has not improved for ``patience`` rounds (early stopping).
On the first fold, a pruning callback reports the AUC after each round to a
``SuccessiveHalvingPruner``, which stops weak candidates mid-training. The
trial value is the mean AUC across folds.

The best candidate is refit on the full training set for the mean number of
rounds at which early stopping halted on the folds.
"""

import warnings

import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from catboost import CatBoost, Pool
from optuna_integration import (
    CatBoostPruningCallback,
    LightGBMPruningCallback,
    XGBoostPruningCallback,
)
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
# silence the experimental-feature warning raised by CatBoostPruningCallback
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# Parameters that are not tuned.
FIXED_PARAMS = {
    "cat": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 10,
        "verbose": False,
        "allow_writing_files": False,
    },
    "lgbm": {"objective": "binary", "metric": "auc", "seed": 10, "verbosity": -1},
    "xgb": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 10,
        "verbosity": 0,
        "tree_method": "hist",
    },
}


class TunedBooster:
    """
    A fitted native booster with the results of its hyperparameter search.
    ``predict_proba`` returns the scikit-learn layout, shape (n_samples, 2).
    """

    def __init__(
        self, library, booster, best_params, best_num_rounds, best_score, trials
    ):
        self.library = library
        self.booster = booster
        self.best_params_ = best_params
        self.best_num_rounds_ = best_num_rounds
        self.best_score_ = best_score
        self.trials_ = trials

    def predict_proba(self, X):
        p = _predict(self.library, self.booster, X)
        return np.column_stack([1 - p, p])


def _predict(library, booster, X):
    """Probability of the positive class."""
    if library == "lgbm":
        return booster.predict(X)
    if library == "xgb":
        return booster.predict(xgb.DMatrix(X))
    return booster.predict(X, prediction_type="Probability")[:, 1]


# Per-library training on one fold with early stopping. Each returns
# (booster, number of rounds); `trial`, when not None, is used for pruning.


def _train_lgbm(params, X, y, w, X_va, y_va, max_rounds, patience, trial):
    dtrain = lgb.Dataset(X, y, weight=w)
    callbacks = [lgb.early_stopping(patience, verbose=False)]
    if trial:
        callbacks.append(LightGBMPruningCallback(trial, "auc"))
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        valid_sets=[lgb.Dataset(X_va, y_va, reference=dtrain)],
        callbacks=callbacks,
    )
    return booster, booster.best_iteration


def _train_xgb(params, X, y, w, X_va, y_va, max_rounds, patience, trial):
    booster = xgb.train(
        params,
        xgb.DMatrix(X, label=y, weight=w),
        num_boost_round=max_rounds,
        evals=[(xgb.DMatrix(X_va, label=y_va), "valid")],
        early_stopping_rounds=patience,
        callbacks=[XGBoostPruningCallback(trial, "valid-auc")] if trial else None,
        verbose_eval=False,
    )
    # best_iteration is 0-based; keep only the trees up to the best one.
    n_rounds = booster.best_iteration + 1
    return booster[:n_rounds], n_rounds


def _train_cat(params, X, y, w, X_va, y_va, max_rounds, patience, trial):
    model = CatBoost({**params, "iterations": max_rounds})
    pruning_cb = CatBoostPruningCallback(trial, "AUC") if trial else None
    model.fit(
        Pool(X, y, weight=w),
        eval_set=Pool(X_va, y_va),
        early_stopping_rounds=patience,
        callbacks=[pruning_cb] if pruning_cb else None,
    )
    if pruning_cb:
        pruning_cb.check_pruned()
    # get_best_iteration is 0-based; the model is already shrunk to it.
    return model, model.get_best_iteration() + 1


def _refit(library, params, X, y, w, rounds):
    """Train on the full data for a fixed number of rounds."""
    if library == "lgbm":
        return lgb.train(params, lgb.Dataset(X, y, weight=w), num_boost_round=rounds)
    if library == "xgb":
        dtrain = xgb.DMatrix(X, label=y, weight=w)
        return xgb.train(params, dtrain, num_boost_round=rounds)
    model = CatBoost({**params, "iterations": rounds})
    model.fit(Pool(X, y, weight=w))
    return model


def tune_native_gbm(
    library,
    suggest_fn,
    folds,
    X,
    y,
    sample_weight=None,
    n_trials=100,
    max_rounds=2000,
    patience=50,
):
    """
    Tune a native gradient boosting model with Optuna and refit it on (X, y).

    Parameters
    ----------
    library : {"cat", "lgbm", "xgb"}
    suggest_fn : callable
        ``suggest_fn(trial) -> dict`` with the candidate hyperparameters,
        using native parameter names.
    folds : list of tuples
        ``(X_train, y_train, w_train, X_valid, y_valid)`` per fold. ``w_train``
        is None without cost-sensitive learning. Validation folds are not
        weighted.
    X, y, sample_weight : arrays
        Training data (and weights) for the final refit.
    n_trials : int
        Number of Optuna trials.
    max_rounds : int
        Upper bound on boosting rounds; early stopping normally ends sooner.
    patience : int
        Stop boosting when the validation AUC has not improved for this many
        rounds.

    Returns
    -------
    TunedBooster
    """
    train = {"lgbm": _train_lgbm, "xgb": _train_xgb, "cat": _train_cat}[library]

    def objective(trial):
        params = {**FIXED_PARAMS[library], **suggest_fn(trial)}
        scores, rounds = [], []
        for i, (X_tr, y_tr, w_tr, X_va, y_va) in enumerate(folds):
            pruning_trial = trial if i == 0 else None  # prune on the first fold only
            booster, n = train(
                params,
                X_tr,
                y_tr,
                w_tr,
                X_va,
                y_va,
                max_rounds,
                patience,
                pruning_trial,
            )
            scores.append(roc_auc_score(y_va, _predict(library, booster, X_va)))
            rounds.append(n)
        trial.set_user_attr("best_iterations", rounds)
        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=10),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=50, reduction_factor=3
        ),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    n_rounds = int(round(np.mean(best.user_attrs["best_iterations"])))
    booster = _refit(
        library,
        {**FIXED_PARAMS[library], **best.params},
        X,
        y,
        sample_weight,
        n_rounds,
    )

    return TunedBooster(
        library=library,
        booster=booster,
        best_params=best.params,
        best_num_rounds=n_rounds,
        best_score=best.value,
        trials=study.trials_dataframe(),
    )
