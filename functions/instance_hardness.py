import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling.base import BaseUnderSampler


class InstanceHardnessUnderSampler(BaseUnderSampler):
    """
    Undersampler that removes majority class instances based on instance
    hardness, inspired by Smith et al. (2014).

    For each majority class instance, a probability of belonging to class 1
    is estimated via cross-validated classifier scores. Majority class instances
    whose estimated probability of being class 1 exceeds a user-defined
    threshold are considered "hard" to classify and are removed from the
    training set.

    Parameters
    ----------
    estimator : sklearn-compatible classifier, default=RandomForestClassifier()
        The classifier used to estimate instance hardness scores.
        Must support predict_proba.
    threshold : float, default=0.7
        Majority class instances with a predicted probability of class 1
        greater than this threshold will be removed.
    cv : int, default=5
        Number of cross-validation folds used to estimate hardness scores.
    random_state : int or None, default=None
        Random state for reproducibility.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, weights=[0.8, 0.2],
    ...                            random_state=42)
    >>> sampler = InstanceHardnessUnderSampler(threshold=0.4, random_state=42)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    """

    def __init__(
        self,
        estimator=None,
        threshold=0.7,
        cv=5,
        random_state=None,
    ):
        super().__init__()
        self.estimator = estimator
        self.threshold = threshold
        self.cv = cv
        self.random_state = random_state

    def _fit_resample(self, X, y):
        # Determine majority/minority classes
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError(
                "InstanceHardnessUnderSampler only supports binary classification. "
                f"Found classes: {classes}"
            )

        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]

        # Default estimator
        estimator = (
            clone(self.estimator)
            if self.estimator is not None
            else RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        )

        # Ensure estimator has predict_proba
        if not hasattr(estimator, "predict_proba"):
            raise ValueError("The estimator must support predict_proba.")

        # Compute cross-validated hardness scores for majority class instances
        majority_mask = y == majority_class
        majority_indices = np.where(majority_mask)[0]

        hardness_scores = np.zeros(len(y))

        skf = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        # Index of class 1 in predict_proba output
        class_1_idx = list(estimator.classes_ if hasattr(estimator, "classes_")
                           else classes).index(1) if 1 in classes else 1

        for train_idx, test_idx in skf.split(X, y):
            fold_estimator = clone(estimator)
            fold_estimator.fit(X[train_idx], y[train_idx])

            # Determine which column corresponds to class 1
            fitted_classes = list(fold_estimator.classes_)
            if 1 in fitted_classes:
                col = fitted_classes.index(1)
            else:
                col = 1  # fallback

            proba = fold_estimator.predict_proba(X[test_idx])
            hardness_scores[test_idx] = proba[:, col]

        # Identify majority class instances to remove:
        # those with P(class=1) > threshold
        majority_to_remove = majority_indices[
            hardness_scores[majority_indices] > self.threshold
        ]

        # Build mask of instances to keep
        keep_mask = np.ones(len(y), dtype=bool)
        keep_mask[majority_to_remove] = False

        self.hardness_scores_ = hardness_scores
        self.removed_indices_ = majority_to_remove
        self.majority_class_ = majority_class
        self.minority_class_ = minority_class

        X_resampled = X[keep_mask]
        y_resampled = y[keep_mask]

        return X_resampled, y_resampled
