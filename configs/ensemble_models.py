from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# AdaBoost, CatBoost, LightGBM and XGBoost are referenced by name and tuned
# with early stopping (see functions/cv_boosting.py). The gradient boosting
# classifier stops adding trees when the loss on an internal validation split
# has not improved for 3 rounds.
estimator_dict = {
    "rf": RandomForestClassifier(random_state=10, n_jobs=-1),
    "ada": "ada",
    "gbm": GradientBoostingClassifier(n_iter_no_change=3, random_state=10),
    "cat": "cat",
    "lgbm": "lgbm",
    "xgb": "xgb",
}
