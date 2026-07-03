"""
Undersampling on the FULL credit-card dataset (re-run for #18/#19/#20).

The earlier experiments loaded a 20k-majority subsample, which is itself random
undersampling and hid the effect. On the whole 284,807-row dataset (~0.17%
positive), a naive model underfits the tiny minority; some class correction is
needed. This compares the ways of correcting:

  baseline            no correction (full data)
  class_weight        balanced sample weights (full data)
  rus_<r>             random undersampling to r negatives per positive
  rus_<r>_wt          rus + prior-restoring weights (recover the original prior)

on Random Forest and XGBoost, over 3 seeds, reporting threshold-free metrics
(ROC-AUC, average precision), calibration (Brier, mean predicted probability) and
training time. The question: does undersampling beat a properly class-corrected
baseline (Elkan says no), how much time does it save, and do the prior-restoring
weights recover calibration.
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
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from functions.imbalanced_data import load_imbalanced_dataset

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

SEEDS = [0, 1, 2]
RATIOS = [1, 10, 100]  # negatives per positive for the undersampling variants
OUTPUT = REPO_ROOT / "models" / "undersampling-creditcard-full"
OUTPUT.mkdir(parents=True, exist_ok=True)


def make(learner, seed):
    if learner == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    return XGBClassifier(
        n_estimators=300,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        eval_metric="logloss",
    )


def evaluate(clf, X_test, y_test):
    p = clf.predict_proba(X_test)[:, 1]
    return {
        "roc": roc_auc_score(y_test, p),
        "ap": average_precision_score(y_test, p),
        "brier": brier_score_loss(y_test, p),
        "mean_prob": p.mean(),
    }


def main():
    Xtr, Xte, ytr, yte = load_imbalanced_dataset("creditcard")
    n_neg_full = int((ytr == 0).sum())
    print(
        f"full creditcard: train={len(ytr):,} ({100 * ytr.mean():.3f}% pos) "
        f"test={len(yte):,} (pos={int(yte.sum())})",
        flush=True,
    )

    methods = ["baseline", "class_weight"]
    for r in RATIOS:
        methods += [f"rus_{r}", f"rus_{r}_wt"]

    results = {
        "meta": {
            "n_train": len(ytr),
            "n_neg_full": n_neg_full,
            "test_prior": float(yte.mean()),
        },
        "runs": [],
    }

    for learner in ["rf", "xgb"]:
        print(f"\n##### {learner} #####", flush=True)
        print(
            f"{'method':<12}{'roc':>16}{'ap':>16}{'brier':>11}{'mean_p':>9}{'fit_s':>8}",
            flush=True,
        )
        for method in methods:
            roc, ap, brier, mp, ft = [], [], [], [], []
            for seed in SEEDS:
                Xu, yu, sw = Xtr, ytr, None
                if method == "class_weight":
                    sw = compute_sample_weight("balanced", ytr)
                elif method.startswith("rus_"):
                    r = int(method.split("_")[1])
                    Xu, yu = RandomUnderSampler(
                        sampling_strategy=1.0 / r, random_state=seed
                    ).fit_resample(Xtr, ytr)
                    if method.endswith("_wt"):
                        w0 = n_neg_full / int((yu == 0).sum())
                        sw = np.where(yu == 0, w0, 1.0)
                clf = make(learner, seed)
                t0 = time.perf_counter()
                clf.fit(Xu, yu, sample_weight=sw)
                ft.append(time.perf_counter() - t0)
                m = evaluate(clf, Xte, yte)
                roc.append(m["roc"])
                ap.append(m["ap"])
                brier.append(m["brier"])
                mp.append(m["mean_prob"])
                results["runs"].append(
                    {
                        "learner": learner,
                        "method": method,
                        "seed": seed,
                        **m,
                        "fit_time_s": ft[-1],
                    }
                )
            print(
                f"{method:<12}{np.mean(roc):>8.4f}±{np.std(roc):<6.4f}"
                f"{np.mean(ap):>8.4f}±{np.std(ap):<6.4f}"
                f"{np.mean(brier):>10.1e}{np.mean(mp):>9.4f}{np.mean(ft):>8.1f}",
                flush=True,
            )

    with open(OUTPUT / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nsaved -> {OUTPUT / 'results.pkl'}\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
