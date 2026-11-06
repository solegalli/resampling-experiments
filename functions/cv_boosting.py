"""
Hyperparameter tuning for CatBoost, LightGBM and XGBoost using their native
training APIs and Optuna.

Instead of treating the number of boosting rounds as a successive-halving
resource with arbitrary limits, each candidate is trained with native early
stopping on the held-out fold: boosting stops when the validation metric has not
improved for ``patience`` rounds. Optuna's ``SuccessiveHalvingPruner`` is fed,
round by round, by the optuna-integration pruning callbacks, so weak candidates
are stopped mid-training.

Per trial, 3 folds are trained sequentially. The pruning callback is attached to
the first fold only, because Optuna cannot receive the same step more than once
per trial; the pruner therefore compares candidates on the first fold's
validation curve. The trial value is the mean score across the 3 folds.

The final model is refit on the full training data with the number of rounds set
to the (rounded) mean of the per-fold best iterations.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import optuna
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
# CatBoostPruningCallback is flagged experimental; the warning would repeat
# once per trial.
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

LIBRARIES = ("cat", "lgbm", "xgb")
VALID_SCORINGS = ("roc_auc", "log_loss", "brier")

# Native evaluation metric name per library and scoring. "brier" is computed
# with a custom eval function for LightGBM and XGBoost.
_NATIVE_METRIC = {
    "lgbm": {"roc_auc": "auc", "log_loss": "binary_logloss", "brier": "brier"},
    "xgb": {"roc_auc": "auc", "log_loss": "logloss", "brier": "brier"},
    "cat": {"roc_auc": "AUC", "log_loss": "Logloss", "brier": "BrierScore"},
}


@dataclass
class NativeGBM:
    """
    Specification of a gradient boosting model trained with its native API.

    Parameters
    ----------
    library : {"cat", "lgbm", "xgb"}
        Which boosting library to use.
    fixed_params : dict
        Parameters that are not tuned (e.g. seed, verbosity).
    """

    library: str
    fixed_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.library not in LIBRARIES:
            raise ValueError(
                f"library must be one of {LIBRARIES}, got '{self.library}'"
            )


class TunedBooster:
    """
    A natively trained booster plus a summary of its hyperparameter search.

    Exposes ``predict_proba`` with the scikit-learn output layout, so it can be
    evaluated like the other models in this repository.
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
        p = _predict_positive(self.library, self.booster, X)
        return np.column_stack([1 - p, p])


def _predict_positive(library, booster, X):
    """Probability of the positive class from a native booster."""
    X = np.asarray(X, dtype=np.float64)
    if library == "lgbm":
        return booster.predict(X)
    if library == "xgb":
        import xgboost as xgb

        return booster.predict(xgb.DMatrix(X))
    return booster.predict(X, prediction_type="Probability")[:, 1]


def _score(scoring, y, p):
    if scoring == "roc_auc":
        return roc_auc_score(y, p)
    if scoring == "log_loss":
        return log_loss(y, p, labels=[0, 1])
    return brier_score_loss(y, p)


# ---------------------------------------------------------------------------
# Per-library training. Each returns (fitted booster, number of rounds).
# When X_va is None the model is trained for exactly `max_rounds` rounds
# (used for the final refit); otherwise early stopping is applied.
# ---------------------------------------------------------------------------


def _train_lgbm(
    params, scoring, X_tr, y_tr, w_tr, X_va, y_va, max_rounds, patience, trial
):
    import lightgbm as lgb

    metric = _NATIVE_METRIC["lgbm"][scoring]
    params = {"objective": "binary", **params}
    params["metric"] = "None" if scoring == "brier" else metric
    dtrain = lgb.Dataset(X_tr, y_tr, weight=w_tr, free_raw_data=False)

    if X_va is None:
        return lgb.train(params, dtrain, num_boost_round=max_rounds), max_rounds

    def brier_eval(preds, data):
        return "brier", float(np.mean((preds - data.get_label()) ** 2)), False

    dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
    callbacks = [lgb.early_stopping(patience, first_metric_only=True, verbose=False)]
    if trial is not None:
        from optuna_integration import LightGBMPruningCallback

        callbacks.append(LightGBMPruningCallback(trial, metric, valid_name="valid_0"))

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        valid_sets=[dvalid],
        valid_names=["valid_0"],
        feval=brier_eval if scoring == "brier" else None,
        callbacks=callbacks,
    )
    return booster, booster.best_iteration


def _train_xgb(
    params, scoring, X_tr, y_tr, w_tr, X_va, y_va, max_rounds, patience, trial
):
    import xgboost as xgb

    metric = _NATIVE_METRIC["xgb"][scoring]
    params = {"objective": "binary:logistic", **params}
    if scoring == "brier":
        params["disable_default_eval_metric"] = 1
    else:
        params["eval_metric"] = metric
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)

    if X_va is None:
        return xgb.train(params, dtrain, num_boost_round=max_rounds), max_rounds

    def brier_eval(preds, data):
        return "brier", float(np.mean((preds - data.get_label()) ** 2))

    dvalid = xgb.DMatrix(X_va, label=y_va)
    callbacks = []
    if trial is not None:
        from optuna_integration import XGBoostPruningCallback

        callbacks.append(XGBoostPruningCallback(trial, f"valid-{metric}"))

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=patience,
        custom_metric=brier_eval if scoring == "brier" else None,
        maximize=scoring == "roc_auc",
        callbacks=callbacks,
        verbose_eval=False,
    )
    # best_iteration is 0-based; keep only the trees up to the best one.
    n_rounds = booster.best_iteration + 1
    return booster[:n_rounds], n_rounds


def _train_cat(
    params, scoring, X_tr, y_tr, w_tr, X_va, y_va, max_rounds, patience, trial
):
    from catboost import CatBoost, Pool

    metric = _NATIVE_METRIC["cat"][scoring]
    params = {
        "loss_function": "Logloss",
        "eval_metric": metric,
        **params,
        "iterations": max_rounds,
    }
    model = CatBoost(params)
    train_pool = Pool(X_tr, y_tr, weight=w_tr)

    if X_va is None:
        model.fit(train_pool)
        return model, max_rounds

    callbacks, pruning_cb = None, None
    if trial is not None:
        from optuna_integration import CatBoostPruningCallback

        pruning_cb = CatBoostPruningCallback(trial, metric)
        callbacks = [pruning_cb]

    model.fit(
        train_pool,
        eval_set=Pool(X_va, y_va),
        early_stopping_rounds=patience,
        use_best_model=True,
        callbacks=callbacks,
    )
    if pruning_cb is not None:
        pruning_cb.check_pruned()
    # get_best_iteration is 0-based; use_best_model already shrank the model.
    return model, model.get_best_iteration() + 1


_TRAINERS = {"lgbm": _train_lgbm, "xgb": _train_xgb, "cat": _train_cat}


def tune_native_gbm(
    spec,
    suggest_fn,
    folds,
    X_full,
    y_full,
    w_full=None,
    scoring="roc_auc",
    n_trials=100,
    max_rounds=2000,
    patience=3,
):
    """
    Tune and refit a native gradient boosting model with Optuna.

    Parameters
    ----------
    spec : NativeGBM
        Which library to use and its fixed (non-tuned) parameters.
    suggest_fn : callable
        ``suggest_fn(trial) -> dict`` returning the candidate hyperparameters,
        using native parameter names.
    folds : list of tuples
        ``(X_train, y_train, w_train, X_valid, y_valid)`` per fold. ``w_train``
        may be None. Validation folds are never weighted.
    X_full, y_full : arrays
        Training data for the final refit.
    w_full : array, optional
        Sample weights for the final refit.
    scoring : {"roc_auc", "log_loss", "brier"}
        Metric used for early stopping, pruning and candidate selection.
    n_trials : int
        Number of Optuna trials.
    max_rounds : int
        Upper bound on boosting rounds; early stopping normally ends sooner.
    patience : int
        Stop boosting when the validation metric has not improved for this
        many rounds.

    Returns
    -------
    TunedBooster
    """
    if scoring not in VALID_SCORINGS:
        raise ValueError(f"scoring must be one of {VALID_SCORINGS}, got '{scoring}'")

    train_fn = _TRAINERS[spec.library]

    def objective(trial):
        params = {**spec.fixed_params, **suggest_fn(trial)}
        scores, best_iters = [], []
        for i, (X_tr, y_tr, w_tr, X_va, y_va) in enumerate(folds):
            booster, n_rounds = train_fn(
                params,
                scoring,
                X_tr,
                y_tr,
                w_tr,
                X_va,
                y_va,
                max_rounds,
                patience,
                trial if i == 0 else None,  # prune on the first fold only
            )
            p_va = _predict_positive(spec.library, booster, X_va)
            scores.append(_score(scoring, y_va, p_va))
            best_iters.append(int(n_rounds))
        trial.set_user_attr("best_iterations", best_iters)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize" if scoring == "roc_auc" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=10),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=10, reduction_factor=3
        ),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = study.best_trial
    n_rounds = max(1, int(round(np.mean(best.user_attrs["best_iterations"]))))
    final_params = {**spec.fixed_params, **best.params}
    booster, _ = train_fn(
        final_params,
        scoring,
        X_full,
        y_full,
        w_full,
        None,
        None,
        n_rounds,
        patience,
        None,
    )

    return TunedBooster(
        library=spec.library,
        booster=booster,
        best_params=best.params,
        best_num_rounds=n_rounds,
        best_score=best.value,
        trials=study.trials_dataframe(),
    )


def _as_float(X):
    return np.asarray(X, dtype=np.float64)


def tune_native_gbm_cv(
    spec, suggest_fn, X, y, sample_weight=None, scoring="roc_auc", **kwargs
):
    """
    Tune a native GBM with a 3-fold stratified split of (X, y).

    Sample weights, if given, are applied to the training folds and the final
    refit only; validation folds are unweighted.
    """
    from sklearn.model_selection import StratifiedKFold

    X, y = _as_float(X), np.asarray(y)
    w = None if sample_weight is None else np.asarray(sample_weight)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
    folds = [
        (X[tr], y[tr], None if w is None else w[tr], X[va], y[va])
        for tr, va in skf.split(X, y)
    ]
    return tune_native_gbm(spec, suggest_fn, folds, X, y, w, scoring=scoring, **kwargs)


def tune_native_gbm_presplit(
    spec,
    suggest_fn,
    xtrain,
    ytrain,
    xtest,
    ytest,
    X_full,
    y_full,
    scoring="roc_auc",
    **kwargs,
):
    """
    Tune a native GBM on pre-computed (e.g. resampled) folds and refit on
    (X_full, y_full).
    """
    folds = [
        (_as_float(xtr), np.asarray(ytr), None, _as_float(xva), np.asarray(yva))
        for xtr, ytr, xva, yva in zip(xtrain, ytrain, xtest, ytest)
    ]
    return tune_native_gbm(
        spec,
        suggest_fn,
        folds,
        _as_float(X_full),
        np.asarray(y_full),
        scoring=scoring,
        **kwargs,
    )
