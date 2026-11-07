from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# CatBoost, LightGBM and XGBoost are referenced by name and trained with their
# native APIs (see functions/cv_boosting.py). The gradient boosting classifier
# stops adding trees when the loss on an internal validation split (10% of the
# training data) has not improved for 20 rounds.
estimator_dict = {
    "rf": RandomForestClassifier(random_state=10, n_jobs=-1),
    "ada": AdaBoostClassifier(random_state=10),
    "gbm": GradientBoostingClassifier(n_iter_no_change=20, random_state=10),
    "cat": "cat",
    "lgbm": "lgbm",
    "xgb": "xgb",
}
