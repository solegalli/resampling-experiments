from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# CatBoost, LightGBM and XGBoost are referenced by name and trained with their
# native APIs (see functions/cv_boosting.py).
estimator_dict = {
    "rf": RandomForestClassifier(random_state=10, n_jobs=-1),
    "ada": AdaBoostClassifier(random_state=10),
    "gbm": GradientBoostingClassifier(random_state=10),
    "cat": "cat",
    "lgbm": "lgbm",
    "xgb": "xgb",
}
