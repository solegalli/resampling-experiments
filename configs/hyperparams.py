from scipy import stats

# Search spaces for the natively trained boosters (Optuna, define-by-run).
# They mirror the previous sklearn-wrapper spaces, using native parameter
# names and avoiding invalid values:
#   - learning_rate is sampled on a log scale from 0.01: values near 0 never
#     trigger early stopping and would run to the maximum number of rounds.
#   - max_depth=None meant "library default" in the sklearn wrappers; that is
#     -1 (no limit) for LightGBM and 6 for XGBoost and CatBoost.
#   - column-sampling fractions start at 0.1 (0 is invalid); num_leaves >= 2.


def suggest_cat_params(trial):
    return {
        "depth": trial.suggest_categorical("depth", [1, 2, 3, 4, 6, 8]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 9
        ),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 9),
    }


def suggest_lgbm_params(trial):
    return {
        "max_depth": trial.suggest_categorical("max_depth", [1, 2, 3, 4, 8, -1]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 1023),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
    }


def suggest_xgb_params(trial):
    return {
        "max_depth": trial.suggest_categorical("max_depth", [1, 2, 3, 4, 6, 8]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
    }


hyperparam_ensemble_dict = {
    "rf_params": {
        "max_depth": (1, 2, 3, 4, 8, None),
        "min_samples_split": stats.randint(2, 20),
        "min_samples_leaf": stats.randint(2, 20),
        "max_features": ("log2", 0.25, "sqrt", 1.0),
    },
    "ada_params": {
        "learning_rate": stats.uniform(0, 1),
    },
    "gbm_params": {
        "max_depth": (1, 2, 3, 4, 8, None),
        "learning_rate": stats.uniform(0, 1),
        "min_samples_split": stats.randint(2, 20),
        "max_features": ("log2", 0.25, "sqrt", 1.0),
        "subsample": stats.uniform(0, 1),
    },
    "cat_params": suggest_cat_params,
    "lgbm_params": suggest_lgbm_params,
    "xgb_params": suggest_xgb_params,
}

hyperparam_special_dict = {
    "rus_params": {
        "learning_rate": stats.uniform(0, 10),
    },
    "easy_params": {
        "replacement": (True, False),
    },
    "brf_params": {
        "max_depth": (1, 2, 3, 4, 8, None),
        "min_samples_split": stats.randint(2, 20),
        "min_samples_leaf": stats.randint(2, 20),
        "max_features": ("log2", 0.25, "sqrt", 1.0),
    },
}


hyperparam_basic_dict = {
    "logit": {
        "clf__l1_ratio": stats.uniform(0, 1),
        "clf__C": stats.uniform(0, 1),
    },
    "knn": {
        "clf__n_neighbors": stats.randint(1, 10),
        "clf__weights": ("uniform", "distance"),
        "clf__p": (1, 2),
    },
    "tree": {
        "criterion": ("gini", "entropy", "log_loss"),
        "max_depth": (1, 2, 3, 4, 8, None),
        "min_samples_split": stats.randint(2, 20),
        "min_samples_leaf": stats.randint(2, 20),
        "max_features": ("log2", 0.25, "sqrt", 1.0),
    },
}
