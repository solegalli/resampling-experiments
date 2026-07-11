from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

rus = RUSBoostClassifier(
    estimator=DecisionTreeClassifier(random_state=10),
    learning_rate=1.0,
    sampling_strategy="auto",
    random_state=2909,
)

easy = EasyEnsembleClassifier(
    estimator=AdaBoostClassifier(random_state=10, n_estimators=10),
    sampling_strategy="auto",
    random_state=2909,
)

brf = BalancedRandomForestClassifier(
    criterion="gini",
    sampling_strategy="auto",
    random_state=2909,
    replacement=False,
    bootstrap=True,
)

estimator_dict = {
    "rusboost": rus,
    "easyEnsemble": easy,
    "balancedRF": brf,
}
