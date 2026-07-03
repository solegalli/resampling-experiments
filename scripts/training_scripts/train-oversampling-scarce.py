"""Oversampling when the minority is scarce (#8/#18).

The undersampling work showed that on large imbalanced data, resampling does not
beat a properly class-corrected model, it just trains faster. Oversampling is
meant for the opposite regime: *small* datasets where the minority has only a
handful of examples, too few for the model to learn its shape. There the hope is
that synthesising minority points (SMOTE and relatives) genuinely adds
information a reweighted model cannot recover, because no amount of reweighting
invents the missing examples.

This tests that hope on four datasets with a scarce minority (24-104 positives):

  libras_move   360 rows, 24 positive
  spectrometer  531 rows, 45 positive
  ozone_level  2536 rows, 73 positive
  secom        1567 rows, ~104 positive

For each dataset and learner (Random Forest, XGBoost) it compares:

  baseline        no correction
  class_weight    balanced weights (the Elkan-equivalent correction)
  ros             random oversampling (duplicate the minority)
  smote           SMOTE
  borderline      BorderlineSMOTE
  adasyn          ADASYN

Every resampler is fit *inside* the cross-validation (imblearn Pipeline), so no
synthetic point ever leaks into a validation fold. Performance is threshold-free
(ROC-AUC and average precision) with dispersion from RepeatedStratifiedKFold
(5 folds x 3 repeats). The question: does synthesising minority points beat plain
class weights when the minority is genuinely scarce? If class weights already
match SMOTE here, the "oversampling helps small data" claim does not survive a
threshold-free comparison either.

We also report, per dataset, a *synthetic-quality* score: a Random Forest trained
to tell real minority points from SMOTE-synthetic ones (cross-validated AUC).
0.5 means synthetic points are indistinguishable from real (SMOTE is interpolating
inside the real manifold); near 1.0 means SMOTE is generating off-distribution
points, which is when it tends to hurt.
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    cross_validate,
)
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from functions.hard_data import load_hard_dataset

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

N_SPLITS, N_REPEATS = 5, 3
OUTPUT = REPO_ROOT / "models" / "oversampling-scarce"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_datasets():
    """Return {name: (X, y)} for the four scarce-minority datasets (full data)."""
    data = {}
    for name in ["libras_move", "spectrometer", "ozone_level"]:
        d = fetch_datasets()[name]
        y = np.where(d.target < 0, 0, 1)  # imblearn labels are -1/1
        data[name] = (np.asarray(d.data, dtype=float), y)
    Xtr, Xte, ytr, yte = load_hard_dataset("secom")
    X = np.vstack([np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float)])
    y = np.concatenate([ytr, yte])
    data["secom"] = (X, y)
    return data


def make_learner(name, seed, n_pos):
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed)
    return XGBClassifier(
        n_estimators=300,
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        eval_metric="logloss",
    )


def weighted_learner(name, seed, y):
    """Learner with the balanced class-weight correction built in."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, n_jobs=1, random_state=seed, class_weight="balanced"
        )
    n_pos = int((y == 1).sum())
    spw = (len(y) - n_pos) / max(n_pos, 1)
    return XGBClassifier(
        n_estimators=300,
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        eval_metric="logloss",
        scale_pos_weight=spw,
    )


def samplers(seed, n_pos):
    """Oversamplers, with k_neighbors kept safe for the scarcest fold."""
    k = min(5, max(1, int(n_pos * (N_SPLITS - 1) / N_SPLITS) - 1))
    return {
        "ros": RandomOverSampler(random_state=seed),
        "smote": SMOTE(random_state=seed, k_neighbors=k),
        "borderline": BorderlineSMOTE(random_state=seed, k_neighbors=k),
        "adasyn": ADASYN(random_state=seed, n_neighbors=k),
    }


def synthetic_quality(X, y, seed):
    """CV AUC of a classifier telling real minority from SMOTE-synthetic minority.

    0.5 = synthetic points indistinguishable from real; ->1.0 = off-distribution.
    """
    n_pos = int((y == 1).sum())
    k = min(5, max(1, n_pos - 1))
    Xr, yr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
    real = X[y == 1]
    synth = Xr[len(X) :]  # the appended synthetic minority points
    if len(synth) == 0:
        return float("nan")
    Xd = np.vstack([real, synth])
    yd = np.concatenate([np.zeros(len(real)), np.ones(len(synth))])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=seed)
    return float(np.mean(cross_val_score(clf, Xd, yd, scoring="roc_auc", cv=cv)))


def main():
    seed = 0
    datasets = load_datasets()
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed
    )
    scoring = ["roc_auc", "average_precision"]
    results = {"meta": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS}, "runs": []}

    for dname, (X, y) in datasets.items():
        n_pos = int((y == 1).sum())
        sq = synthetic_quality(X, y, seed)
        print(
            f"\n===== {dname}: {X.shape[0]} rows, {X.shape[1]} feats, "
            f"{n_pos} positive ({100 * y.mean():.1f}%), "
            f"SMOTE synthetic-quality AUC={sq:.3f} =====",
            flush=True,
        )
        results["meta"].setdefault("synthetic_quality", {})[dname] = sq

        for learner in ["rf", "xgb"]:
            estimators = {
                "baseline": make_learner(learner, seed, n_pos),
                "class_weight": weighted_learner(learner, seed, y),
            }
            for sname, samp in samplers(seed, n_pos).items():
                estimators[sname] = Pipeline(
                    [("samp", samp), ("clf", make_learner(learner, seed, n_pos))]
                )

            print(f"  --- {learner} ---", flush=True)
            print(f"  {'method':<13}{'roc_auc':>18}{'avg_precision':>20}", flush=True)
            for mname, est in estimators.items():
                try:
                    cvres = cross_validate(est, X, y, scoring=scoring, cv=cv, n_jobs=-1)
                    roc = cvres["test_roc_auc"]
                    ap = cvres["test_average_precision"]
                except Exception as exc:  # ADASYN can fail on a degenerate fold
                    print(f"  {mname:<13}  failed: {type(exc).__name__}", flush=True)
                    continue
                results["runs"].append(
                    {
                        "dataset": dname,
                        "learner": learner,
                        "method": mname,
                        "roc_auc": roc,
                        "average_precision": ap,
                    }
                )
                print(
                    f"  {mname:<13}{roc.mean():>10.4f}±{roc.std():<6.4f}"
                    f"{ap.mean():>12.4f}±{ap.std():<6.4f}",
                    flush=True,
                )

    with open(OUTPUT / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nsaved -> {OUTPUT / 'results.pkl'}\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
