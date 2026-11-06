import lightgbm as lgb
import numpy as np
import optuna
import pytest
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier

from configs.hyperparams import hyperparam_ensemble_dict
from functions.cv import train_model
from functions.cv_boosting import TunedBooster, _early_stop
from functions.cv_undersamplers import train_model_w_undersampling, undersample_data

X, y = make_classification(
    n_samples=600, n_features=8, weights=[0.8, 0.2], random_state=10
)


@pytest.fixture(scope="module", params=["cat", "lgbm", "xgb"])
def tuned(request):
    library = request.param
    return library, train_model(
        library, hyperparam_ensemble_dict[f"{library}_params"], X, y
    )


def test_predict_proba_shape_and_range(tuned):
    _, model = tuned
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_refit_uses_early_stopping_rounds(tuned):
    library, model = tuned
    assert 0 < model.best_num_rounds_ < 2000
    if library == "lgbm":
        n_trees = model.booster.num_trees()
    elif library == "xgb":
        n_trees = model.booster.num_boosted_rounds()
    else:
        n_trees = model.booster.tree_count_
    assert n_trees == model.best_num_rounds_


def test_trials_store_best_iteration_per_fold(tuned):
    _, model = tuned
    complete = model.trials_[model.trials_["state"] == "COMPLETE"]
    assert complete["user_attrs_best_iterations"].map(len).eq(3).all()


def test_pruning_callback_reports_each_round(mocker):
    spy = mocker.spy(optuna, "create_study")
    train_model("lgbm", hyperparam_ensemble_dict["lgbm_params"], X, y)
    study = spy.spy_return
    assert all(len(t.intermediate_values) > 0 for t in study.trials)


def test_sample_weights_only_on_training_data(mocker):
    spy = mocker.spy(lgb, "Dataset")
    train_model(
        "lgbm",
        hyperparam_ensemble_dict["lgbm_params"],
        X,
        y,
        sample_weight=np.where(y == 1, 4, 1),
    )
    # validation Datasets are built with `reference=`; training ones are not
    calls = spy.call_args_list
    valid = [c for c in calls if "reference" in c.kwargs]
    train = [c for c in calls if "reference" not in c.kwargs]
    assert all(c.kwargs.get("weight") is None for c in valid)
    assert all(c.kwargs["weight"] is not None for c in train)


def test_train_model_w_undersampling_native():
    xtrainu, ytrainu, xtest, ytest, Xu, yu, _ = undersample_data(
        RandomUnderSampler(random_state=10), X, y, scale=False
    )
    model = train_model_w_undersampling(
        "lgbm",
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


def test_early_stop_returns_best_round_before_patience_runs_out():
    # the 0.9 in round 6 comes after 3 rounds without improvement
    assert _early_stop([0.5, 0.6, 0.6, 0.6, 0.6, 0.9], patience=3) == (2, 0.6)


def test_adaboost_tuned_with_learning_rate_grid_and_early_stopping():
    model = train_model("ada", hyperparam_ensemble_dict["ada_params"], X, y)
    assert isinstance(model, AdaBoostClassifier)
    assert model.learning_rate in hyperparam_ensemble_dict["ada_params"]
    assert 0 < model.n_estimators < 1000
    assert len(model.estimators_) == model.n_estimators


def test_adaboost_with_undersampling():
    folds = undersample_data(RandomUnderSampler(random_state=10), X, y, scale=False)
    model = train_model_w_undersampling(
        "ada", hyperparam_ensemble_dict["ada_params"], *folds[:6]
    )
    assert model.predict_proba(X).shape == (len(X), 2)
