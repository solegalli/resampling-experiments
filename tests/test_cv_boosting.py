import numpy as np
import pytest
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification

from configs.ensemble_models import estimator_dict
from configs.hyperparams import hyperparam_ensemble_dict
from functions.cv import train_model
from functions.cv_boosting import (
    NativeGBM,
    TunedBooster,
    tune_native_gbm_cv,
)
from functions.cv_undersamplers import train_model_w_undersampling, undersample_data

LIBRARIES = ["cat", "lgbm", "xgb"]
N_TRIALS = 8
MAX_ROUNDS = 500


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=600, n_features=8, weights=[0.8, 0.2], random_state=10
    )
    return X, y


def _tune(library, X, y, **kwargs):
    return tune_native_gbm_cv(
        estimator_dict[library],
        hyperparam_ensemble_dict[f"{library}_params"],
        X,
        y,
        n_trials=N_TRIALS,
        max_rounds=MAX_ROUNDS,
        **kwargs,
    )


@pytest.fixture(scope="module", params=LIBRARIES)
def tuned(request, data):
    X, y = data
    return request.param, _tune(request.param, X, y)


def test_predict_proba_shape_and_range(tuned, data):
    _, model = tuned
    X, _ = data
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_early_stopping_sets_rounds(tuned):
    _, model = tuned
    assert 0 < model.best_num_rounds_ < MAX_ROUNDS


def test_final_model_uses_best_num_rounds(tuned):
    library, model = tuned
    if library == "lgbm":
        n_trees = model.booster.num_trees()
    elif library == "xgb":
        n_trees = model.booster.num_boosted_rounds()
    else:
        n_trees = model.booster.tree_count_
    assert n_trees == model.best_num_rounds_


def test_trials_recorded_with_pruning_reports(tuned):
    _, model = tuned
    trials = model.trials_
    assert len(trials) == N_TRIALS
    # per-fold best iterations are stored for completed trials
    complete = trials[trials["state"] == "COMPLETE"]
    assert complete["user_attrs_best_iterations"].map(len).eq(3).all()


def test_pruning_callback_reports_intermediate_values(data):
    # The pruning callback is attached to the first fold; it reports a value
    # per boosting round, which the SuccessiveHalvingPruner uses.
    import optuna

    X, y = data
    captured = {}
    original = optuna.create_study

    def spy(*args, **kwargs):
        study = original(*args, **kwargs)
        captured["study"] = study
        return study

    optuna.create_study = spy
    try:
        _tune("lgbm", X, y)
    finally:
        optuna.create_study = original

    trials = captured["study"].trials
    assert all(len(t.intermediate_values) > 0 for t in trials)
    assert isinstance(captured["study"].pruner, optuna.pruners.SuccessiveHalvingPruner)


@pytest.mark.parametrize(
    "scoring, direction",
    [("roc_auc", "maximize"), ("log_loss", "minimize"), ("brier", "minimize")],
)
def test_scoring_sets_direction(mocker, data, scoring, direction):
    import optuna

    X, y = data
    spy = mocker.spy(optuna, "create_study")
    _tune("lgbm", X, y, scoring=scoring)
    assert spy.call_args.kwargs["direction"] == direction


@pytest.mark.parametrize("library", ["lgbm", "xgb"])
def test_brier_and_log_loss_run(library, data):
    X, y = data
    for scoring in ("log_loss", "brier"):
        model = _tune(library, X, y, scoring=scoring)
        assert model.best_score_ > 0


def test_invalid_scoring_raises(data):
    X, y = data
    with pytest.raises(ValueError):
        _tune("lgbm", X, y, scoring="accuracy")


def test_invalid_library_raises():
    with pytest.raises(ValueError):
        NativeGBM("sklearn")


def test_sample_weights_only_on_training_data(mocker, data):
    import lightgbm as lgb

    X, y = data
    w = np.where(y == 1, 4, 1)
    spy = mocker.spy(lgb, "Dataset")
    _tune("lgbm", X, y, sample_weight=w)

    # validation Datasets are built with `reference=`; training ones are not
    train_calls = [c for c in spy.call_args_list if "reference" not in c.kwargs]
    valid_calls = [c for c in spy.call_args_list if "reference" in c.kwargs]
    assert valid_calls and all(c.kwargs.get("weight") is None for c in valid_calls)
    assert train_calls and all(c.kwargs.get("weight") is not None for c in train_calls)


def test_train_model_dispatches_to_native(mocker, data):
    X, y = data
    spy = mocker.patch("functions.cv.tune_native_gbm_cv", return_value="tuned")
    out = train_model(
        estimator_dict["xgb"], hyperparam_ensemble_dict["xgb_params"], X, y
    )
    assert out == "tuned"
    spy.assert_called_once()


def test_train_model_w_undersampling_native(data):
    X, y = data
    xtrainu, ytrainu, xtest, ytest, Xu, yu, _ = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    model = train_model_w_undersampling(
        estimator_dict["lgbm"],
        hyperparam_ensemble_dict["lgbm_params"],
        xtrainu,
        ytrainu,
        xtest,
        ytest,
        Xu,
        yu,
    )
    assert isinstance(model, TunedBooster)
    assert model.predict_proba(X).shape == (len(X), 2)
