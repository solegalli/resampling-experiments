"""
Does resampling reproduce on the Slowloris attack data?

Hasanin et al. (2019, J. Big Data) found that on their severely imbalanced
"SlowlorisBig" set, oversampling and undersampling improved performance (unlike
their first dataset, where, once uncertainty was shown, they did not). The
original SlowlorisBig is not published, so we rebuild a comparable
Slowloris-vs-benign task from the public CICIDS2017 set (UNB CIC; the Wednesday
working-hours capture labels 5,796 "DoS slowloris" flows against the benign
traffic).

The question: with threshold-free metrics and dispersion over seeds, does
oversampling or class-weighting beat the no-resampling baseline here, or does it
join the no-effect pile (our results, and the first SlowlorisBig dataset)?

Design: a fixed Random Forest and a fixed XGBoost, comparing
  baseline | class-weight | random oversampling | SMOTE | random undersampling
on the same stratified 70/30 splits, repeated over 5 seeds. Metrics are
threshold-free (ROC-AUC, average precision, Brier, mean predicted probability)
so nothing depends on the decision threshold. The benign majority is capped to
keep the run tractable while staying severely imbalanced.
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
import time
import warnings
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / ".data_cache" / "cicids_slowloris.pkl"
OUTPUT = REPO_ROOT / "models" / "cicids-slowloris"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_BENIGN = 200_000  # cap the majority for tractability; stays severely imbalanced
METHODS = ["baseline", "class_weight", "ros", "smote", "rus"]


def load_data():
    if not DATA.exists():
        raise SystemExit(
            f"{DATA} missing. Run scripts/data_prep/prepare_cicids_slowloris.py first."
        )
    with open(DATA, "rb") as f:
        d = pickle.load(f)
    X, y = d["X"], d["y"]
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg_keep = np.random.default_rng(0).choice(
        neg, size=min(N_BENIGN, neg.size), replace=False
    )
    idx = np.concatenate([pos, neg_keep])
    return X[idx], y[idx]


def make_clf(learner, seed, balanced, spw):
    if learner == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced" if balanced else None,
        )
    return XGBClassifier(
        n_estimators=300,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        eval_metric="logloss",
        scale_pos_weight=spw if spw else 1.0,
    )


def resample(method, X, y, seed):
    """Return (X_train, y_train, balanced_flag, scale_pos_weight)."""
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    if method == "baseline":
        return X, y, False, None
    if method == "class_weight":
        return X, y, True, n_neg / n_pos
    if method == "ros":
        Xr, yr = RandomOverSampler(random_state=seed).fit_resample(X, y)
    elif method == "smote":
        Xr, yr = SMOTE(random_state=seed).fit_resample(X, y)
    elif method == "rus":
        Xr, yr = RandomUnderSampler(random_state=seed).fit_resample(X, y)
    return Xr, yr, False, None


def main():
    X, y = load_data()
    print(
        f"Slowloris-vs-benign: {len(y):,} rows, {int(y.sum()):,} slowloris "
        f"({100 * y.mean():.3f}% positive), {X.shape[1]} features",
        flush=True,
    )
    results = {
        "meta": {"n": int(len(y)), "pos": int(y.sum()), "features": int(X.shape[1])},
        "runs": [],
    }

    for learner in ["rf", "xgb"]:
        print(f"\n##### {learner} #####", flush=True)
        print(
            f"{'method':<13}{'roc':>16}{'ap':>16}{'brier':>14}{'mean_p':>10}",
            flush=True,
        )
        for method in METHODS:
            roc, ap, brier, mp, ft = [], [], [], [], []
            for seed in SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.3, random_state=seed, stratify=y
                )
                Xu, yu, bal, spw = resample(method, Xtr, ytr, seed)
                clf = make_clf(learner, seed, bal, spw)
                t0 = time.perf_counter()
                clf.fit(Xu, yu)
                ft.append(time.perf_counter() - t0)
                prob = clf.predict_proba(Xte)[:, 1]
                roc.append(roc_auc_score(yte, prob))
                ap.append(average_precision_score(yte, prob))
                brier.append(brier_score_loss(yte, prob))
                mp.append(float(prob.mean()))
                results["runs"].append(
                    {
                        "learner": learner,
                        "method": method,
                        "seed": seed,
                        "roc": roc[-1],
                        "ap": ap[-1],
                        "brier": brier[-1],
                        "mean_prob": mp[-1],
                        "fit_time_s": ft[-1],
                    }
                )
            print(
                f"{method:<13}{np.mean(roc):>8.4f}±{np.std(roc):<6.4f}"
                f"{np.mean(ap):>8.4f}±{np.std(ap):<6.4f}"
                f"{np.mean(brier):>8.1e}    {np.mean(mp):>7.4f}",
                flush=True,
            )

    with open(OUTPUT / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nsaved -> {OUTPUT / 'results.pkl'}\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
