from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from functions.cv_boosting import NativeGBM

# CatBoost, LightGBM and XGBoost are trained with their native APIs and tuned
# with Optuna plus early stopping (see functions/cv_boosting.py). The number of
# boosting rounds is set by early stopping, not tuned as a hyperparameter.
estimator_dict = {
    "rf": RandomForestClassifier(random_state=10, n_jobs=-1),
    "ada": AdaBoostClassifier(random_state=10),
    "gbm": GradientBoostingClassifier(random_state=10),
    "cat": NativeGBM(
        "cat",
        {"random_seed": 10, "verbose": False, "allow_writing_files": False},
    ),
    "lgbm": NativeGBM("lgbm", {"seed": 10, "verbosity": -1}),
    "xgb": NativeGBM("xgb", {"seed": 10, "verbosity": 0, "tree_method": "hist"}),
}
