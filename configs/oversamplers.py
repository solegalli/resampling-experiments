"""Oversampling strategies for imbalanced datasets.

Each technique is provided in two variants:
- equal: oversample minority until minority == majority (sampling_strategy=1.0)
- half: oversample minority until minority == 0.5 * majority (sampling_strategy=0.5)

ROS is additionally provided with shrinkage (0.1 and 1) in both variants.
"""

from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler

# Random Over-Sampler
ros_equal = RandomOverSampler(sampling_strategy=1.0, random_state=0)
ros_half = RandomOverSampler(sampling_strategy=0.5, random_state=0)

ros_shrink01_equal = RandomOverSampler(
    sampling_strategy=1.0, random_state=0, shrinkage=0.1
)
ros_shrink01_half = RandomOverSampler(
    sampling_strategy=0.5, random_state=0, shrinkage=0.1
)

ros_shrink1_equal = RandomOverSampler(
    sampling_strategy=1.0, random_state=0, shrinkage=1
)
ros_shrink1_half = RandomOverSampler(sampling_strategy=0.5, random_state=0, shrinkage=1)

# SMOTE
smote_equal = SMOTE(sampling_strategy=1.0, random_state=0, k_neighbors=5)
smote_half = SMOTE(sampling_strategy=0.5, random_state=0, k_neighbors=5)

# ADASYN
adasyn_equal = ADASYN(sampling_strategy=1.0, random_state=0, n_neighbors=5)
adasyn_half = ADASYN(sampling_strategy=0.5, random_state=0, n_neighbors=5)

# Borderline SMOTE 1
bsmote1_equal = BorderlineSMOTE(
    sampling_strategy=1.0, random_state=0, k_neighbors=5, kind="borderline-1"
)
bsmote1_half = BorderlineSMOTE(
    sampling_strategy=0.5, random_state=0, k_neighbors=5, kind="borderline-1"
)

# Borderline SMOTE 2
bsmote2_equal = BorderlineSMOTE(
    sampling_strategy=1.0, random_state=0, k_neighbors=5, kind="borderline-2"
)
bsmote2_half = BorderlineSMOTE(
    sampling_strategy=0.5, random_state=0, k_neighbors=5, kind="borderline-2"
)
