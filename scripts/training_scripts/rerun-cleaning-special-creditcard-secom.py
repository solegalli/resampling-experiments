"""Special ensembles and cleaning undersamplers on the changed datasets (#18/#20).

Completes the full-creditcard / out-of-sample-secom re-run. Notebook 14 already
covered the plain undersampling story on full creditcard (baseline / class_weight
/ RandomUnderSampler). This adds the two remaining method families on the two
datasets whose loaders changed:

  - special ensembles that build resampling into the model
    (BalancedRandomForest, RUSBoost, EasyEnsemble), and
  - the cleaning undersamplers (TomekLinks, ENN, RENN, AllKNN, NCR, OSS, CNN)
    plus NearMiss.

against a Random Forest baseline and the class-weight correction, with
threshold-free metrics (ROC-AUC, average precision), calibration (Brier, mean
predicted probability) and training time, over three seeds.

Tractability. The neighbour-based cleaning undersamplers need a k-NN query per
sample; at 199k rows and ~30 dimensions that degrades to brute force and does not
finish, which is the same reason the earlier scripts restricted the full
credit-card set. So on creditcard we run only the row-reducers (RandomUnderSampler,
NearMiss) plus the special ensembles (which undersample internally and stay fast);
on secom (1096 rows) we run the whole cleaning suite. Any sampler exceeding a wall
-clock budget is skipped and logged, so nothing is silently dropped.

The question is the same as everywhere else: on threshold-free metrics, do these
resampling-based methods beat a class-weighted Random Forest, or is class
correction still all that is needed.
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
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.under_sampling import (
    AllKNN,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from functions.hard_data import load_hard_dataset
from functions.imbalanced_data import load_imbalanced_dataset

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

SEEDS = [0, 1, 2]
SAMPLER_BUDGET_S = 900  # skip a sampler on a dataset if resampling exceeds this
OUTPUT = REPO_ROOT / "models" / "cleaning-special-rerun"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Cleaning / row undersamplers, built per seed (a few take a random_state).
CLEANERS = {
    "tomek": lambda s: TomekLinks(),
    "enn": lambda s: EditedNearestNeighbours(),
    "renn": lambda s: RepeatedEditedNearestNeighbours(),
    "allknn": lambda s: AllKNN(),
    "ncr": lambda s: NeighbourhoodCleaningRule(),
    "oss": lambda s: OneSidedSelection(random_state=s),
    "cnn": lambda s: CondensedNearestNeighbour(random_state=s),
    "nearmiss1": lambda s: NearMiss(version=1),
    "nearmiss2": lambda s: NearMiss(version=2),
    "rus": lambda s: RandomUnderSampler(random_state=s),
}
ROW_REDUCERS = ["rus", "nearmiss1", "nearmiss2"]  # tractable on full creditcard

# Special ensembles, built per seed.
SPECIAL = {
    "balanced_rf": lambda s: BalancedRandomForestClassifier(
        n_estimators=200,
        n_jobs=1,
        random_state=s,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
    ),
    "rusboost": lambda s: RUSBoostClassifier(n_estimators=200, random_state=s),
    "easyensemble": lambda s: EasyEnsembleClassifier(
        n_estimators=20, n_jobs=1, random_state=s
    ),
}


def rf(seed, weighted=False):
    return RandomForestClassifier(
        n_estimators=200,
        n_jobs=1,
        random_state=seed,
        class_weight="balanced" if weighted else None,
    )


def evaluate(clf, Xte, yte):
    p = clf.predict_proba(Xte)[:, 1]
    return {
        "roc": roc_auc_score(yte, p),
        "ap": average_precision_score(yte, p),
        "brier": brier_score_loss(yte, p),
        "mean_prob": float(p.mean()),
    }


def run_dataset(name, loader, cleaners, results):
    Xtr, Xte, ytr, yte = loader(name)
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)
    ytr = np.asarray(ytr)
    yte = np.asarray(yte)
    print(
        f"\n===== {name}: train={len(ytr):,} ({100 * ytr.mean():.3f}% pos) "
        f"feats={Xtr.shape[1]} test_pos={int(yte.sum())} =====",
        flush=True,
    )
    header = (
        f"{'method':<16}{'roc':>16}{'ap':>16}{'brier':>11}{'mean_p':>9}{'fit_s':>8}"
    )

    def record(method, family, metrics, fits):
        results["runs"].append(
            {"dataset": name, "method": method, "family": family, **metrics}
        )
        m = metrics
        print(
            f"{method:<16}{m['roc_m']:>8.4f}±{m['roc_s']:<6.4f}"
            f"{m['ap_m']:>8.4f}±{m['ap_s']:<6.4f}"
            f"{m['brier_m']:>10.1e}{m['mean_prob_m']:>9.4f}{np.mean(fits):>8.1f}",
            flush=True,
        )

    def agg(dicts, fits):
        arr = {k: np.array([d[k] for d in dicts]) for k in dicts[0]}
        out = {}
        for k, v in arr.items():
            out[f"{k}_m"] = float(v.mean())
            out[f"{k}_s"] = float(v.std())
        out["fit_s"] = float(np.mean(fits))
        return out

    print(header, flush=True)

    # baseline + class weights (Random Forest)
    for method, weighted in [("baseline", False), ("class_weight", True)]:
        ms, fits = [], []
        for seed in SEEDS:
            clf = rf(seed, weighted=weighted)
            t0 = time.perf_counter()
            sw = compute_sample_weight("balanced", ytr) if weighted else None
            clf.fit(Xtr, ytr, sample_weight=sw)
            fits.append(time.perf_counter() - t0)
            ms.append(evaluate(clf, Xte, yte))
        record(method, "reference", agg(ms, fits), fits)

    # special ensembles
    for method, build in SPECIAL.items():
        ms, fits = [], []
        for seed in SEEDS:
            clf = build(seed)
            t0 = time.perf_counter()
            clf.fit(Xtr, ytr)
            fits.append(time.perf_counter() - t0)
            ms.append(evaluate(clf, Xte, yte))
        record(method, "special_ensemble", agg(ms, fits), fits)

    # cleaning / row undersamplers -> Random Forest
    for method in cleaners:
        ms, fits, skipped = [], [], False
        for seed in SEEDS:
            sampler = CLEANERS[method](seed)
            t0 = time.perf_counter()
            try:
                Xu, yu = sampler.fit_resample(Xtr, ytr)
            except Exception as exc:
                print(f"{method:<16}  failed: {type(exc).__name__}", flush=True)
                skipped = True
                break
            samp_s = time.perf_counter() - t0
            if samp_s > SAMPLER_BUDGET_S:
                print(
                    f"{method:<16}  skipped: resampling took {samp_s:.0f}s "
                    f"> {SAMPLER_BUDGET_S}s budget",
                    flush=True,
                )
                skipped = True
                break
            clf = rf(seed)
            t1 = time.perf_counter()
            clf.fit(Xu, yu)
            fits.append(samp_s + time.perf_counter() - t1)
            ms.append(evaluate(clf, Xte, yte))
        if not skipped:
            record(method, "undersampler", agg(ms, fits), fits)


def main():
    results = {"runs": []}
    # creditcard first (the slow one), secom second
    run_dataset("creditcard", load_imbalanced_dataset, ROW_REDUCERS, results)
    run_dataset("secom", load_hard_dataset, list(CLEANERS), results)
    with open(OUTPUT / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nsaved -> {OUTPUT / 'results.pkl'}\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
