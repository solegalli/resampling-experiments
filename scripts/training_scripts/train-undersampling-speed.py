"""
Undersampling for speed: does dropping majority-class rows let us train faster
without losing performance, and do class weights restore calibration?

First experiment from the "when should we actually reach for resampling"
discussion (issue #8). The motivation is not that resampling improves a model
(we have established it does not), but that on a large, severely imbalanced
dataset undersampling the majority can cut training time sharply while keeping
the signal, as long as we put the original class prior back with class weights
so the probabilities stay calibrated.

For each dataset we run two curves, repeated over several seeds, with a fixed
Random Forest so the only thing changing is the training data:

Part A -- original-distribution learning curve
    Train on growing stratified subsamples that keep the real prior. Shows how
    much data is needed, and that the binding constraint is the number of
    positives (which only grows with total size).

Part B -- undersample the majority
    Keep every positive, vary the negatives per positive, and train with and
    without prior-restoring class weights. Shows how few rows (and how little
    time) are needed to match the full-data model, and whether the weights
    recover the calibration that undersampling destroys.

Datasets span the imbalance/size spectrum so we can see where the speed payoff
appears: creditcard (~0.17% positive, huge) and protein_homo (~0.9%, big) are
the strong cases, diabetes130 (~11%) is moderate, isolet (~7.7%, small,
high-dimensional) is the small-data contrast where there is no speed case.

Metrics are threshold-free (ROC-AUC, average precision, Brier, mean predicted
probability), so none of this depends on the decision threshold. Results are
saved per dataset and the run is resumable (a finished dataset is skipped).
"""

import os

# Keep BLAS from oversubscribing against the forest's own parallelism, and make
# the fit-time measurements comparable across runs.
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
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from functions.hard_data import load_hard_dataset
from functions.imbalanced_data import _fetch_openml

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

N_ESTIMATORS = 200
SEEDS = [0, 1, 2, 3, 4]
SPLIT_STATE = 0  # the test split is fixed; seeds vary the subsample and the model
OUTPUT = REPO_ROOT / "models" / "undersampling-speed"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Part A: fractions of the training set, original (stratified) distribution.
FRACTIONS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
# Part B: negatives kept per positive (ratios above the dataset's own ratio are
# skipped, since they would just be the full set).
RATIOS = [1, 2, 5, 10, 25, 50, 100, 200]


def _to_xy(X, y):
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y).astype(int)
    return X, y


def load_creditcard_full():
    X, y = _fetch_openml("creditcard", 1597)
    y = pd.to_numeric(y).astype(int).to_numpy()
    X = X.apply(pd.to_numeric).reset_index(drop=True).to_numpy()
    return train_test_split(X, y, test_size=0.3, random_state=SPLIT_STATE)


def load_diabetes130():
    X_train, X_test, y_train, y_test = load_hard_dataset("diabetes130")
    X_train, y_train = _to_xy(X_train, y_train)
    X_test, y_test = _to_xy(X_test, y_test)
    return X_train, X_test, y_train, y_test


def load_imblearn(name):
    from imblearn.datasets import fetch_datasets

    d = fetch_datasets()[name]
    X, y = _to_xy(d.data, (np.asarray(d.target) == 1))
    return train_test_split(X, y, test_size=0.3, random_state=SPLIT_STATE, stratify=y)


LOADERS = {
    "isolet": lambda: load_imblearn("isolet"),
    "diabetes130": load_diabetes130,
    "protein_homo": lambda: load_imblearn("protein_homo"),
    "creditcard": load_creditcard_full,
}


def fit_and_eval(X_train, y_train, X_test, y_test, seed, class_weight=None):
    """Fit a fixed Random Forest, timing only the fit, and score it threshold-free."""
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        n_jobs=-1,
        random_state=seed,
        class_weight=class_weight,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    prob = clf.predict_proba(X_test)[:, 1]
    return {
        "n_train": int(len(y_train)),
        "n_pos": int(y_train.sum()),
        "n_neg": int(len(y_train) - y_train.sum()),
        "roc": float(roc_auc_score(y_test, prob)),
        "ap": float(average_precision_score(y_test, prob)),
        "brier": float(brier_score_loss(y_test, prob)),
        "mean_prob": float(prob.mean()),
        "fit_time_s": float(fit_time),
    }


def stratified_subsample(X, y, fraction, seed):
    if fraction >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=fraction, random_state=seed, stratify=y
    )
    return X_sub, y_sub


def undersample_majority(X, y, neg_per_pos, seed):
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_neg_keep = min(neg_per_pos * len(pos_idx), len(neg_idx))
    neg_keep = np.random.default_rng(seed).choice(
        neg_idx, size=n_neg_keep, replace=False
    )
    keep = np.concatenate([pos_idx, neg_keep])
    return X[keep], y[keep]


def run_dataset(name):
    X_train, X_test, y_train, y_test = LOADERS[name]()
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    print(
        f"\n##### {name}: train={len(y_train):,} (pos={n_pos}, neg={n_neg:,}, "
        f"{100 * y_train.mean():.3f}% positive) | test={len(y_test):,} "
        f"(pos={int(y_test.sum())}, prior={y_test.mean():.5f})",
        flush=True,
    )
    results = {
        "meta": {
            "dataset": name,
            "n_pos_full": n_pos,
            "n_neg_full": n_neg,
            "n_features": int(X_train.shape[1]),
            "test_prior": float(y_test.mean()),
            "n_estimators": N_ESTIMATORS,
            "seeds": SEEDS,
        },
        "A_learning_curve": [],
        "B_undersample": [],
    }

    for frac in FRACTIONS:
        for seed in SEEDS:
            Xs, ys = stratified_subsample(X_train, y_train, frac, seed)
            r = fit_and_eval(Xs, ys, X_test, y_test, seed)
            r.update({"fraction": frac, "seed": seed})
            results["A_learning_curve"].append(r)
        sub = [r for r in results["A_learning_curve"] if r["fraction"] == frac]
        print(
            f"  A frac={frac:<4} n={sub[0]['n_train']:>8,} pos={sub[0]['n_pos']:>5} "
            f"roc={np.mean([r['roc'] for r in sub]):.4f} "
            f"ap={np.mean([r['ap'] for r in sub]):.4f} "
            f"fit={np.mean([r['fit_time_s'] for r in sub]):.2f}s",
            flush=True,
        )

    for ratio in RATIOS:
        Xu0, _ = undersample_majority(X_train, y_train, ratio, SEEDS[0])
        if Xu0.shape[0] >= len(y_train):
            continue  # would be the full set; Part A frac=1.0 is the reference
        for label, make_cw in [("none", None), ("prior", "prior")]:
            for seed in SEEDS:
                Xu, yu = undersample_majority(X_train, y_train, ratio, seed)
                if make_cw == "prior":
                    cw = {0: n_neg / int((yu == 0).sum()), 1: 1.0}
                else:
                    cw = None
                r = fit_and_eval(Xu, yu, X_test, y_test, seed, class_weight=cw)
                r.update({"neg_per_pos": ratio, "weights": label, "seed": seed})
                results["B_undersample"].append(r)
            sub = [
                r
                for r in results["B_undersample"]
                if r["neg_per_pos"] == ratio and r["weights"] == label
            ]
            print(
                f"  B ratio={ratio:<4} {label:<5} n={sub[0]['n_train']:>8,} "
                f"roc={np.mean([r['roc'] for r in sub]):.4f} "
                f"ap={np.mean([r['ap'] for r in sub]):.4f} "
                f"brier={np.mean([r['brier'] for r in sub]):.2e} "
                f"mean_p={np.mean([r['mean_prob'] for r in sub]):.4f} "
                f"fit={np.mean([r['fit_time_s'] for r in sub]):.2f}s",
                flush=True,
            )

    with open(OUTPUT / f"{name}_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"  saved -> {name}_results.pkl", flush=True)


def main():
    for name in LOADERS:
        out = OUTPUT / f"{name}_results.pkl"
        if out.exists():
            print(f"skip {name} (already done)", flush=True)
            continue
        run_dataset(name)
    print("\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
