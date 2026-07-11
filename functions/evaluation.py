import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_class(y, prob):
    """
    Predict binary classes using an optimal threshold derived from the precision-recall curve.

    The best threshold is selected by maximizing the F1 score across all
    thresholds returned by the precision-recall curve.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True binary labels.
    prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns
    -------
    preds : array of shape (n_samples,)
        Binary predictions obtained by applying the best threshold.
    best_threshold : float
        Threshold value that maximizes the F1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y, prob)

    # Calculate F1 scores and get threshold that gives max F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # Apply new threshold
    preds = (prob >= best_threshold).astype(int)
    return preds, best_threshold


def calculate_classif_metrics(y, prob):
    """
    Get the best value of various classification metrics.

    Returns the maximum of Matthews correlation coefficient, balanced accuracy
    and geometric mean over all decision thresholds (each unique predicted
    probability, with samples scored positive when ``prob >= threshold``).

    This is a vectorized equivalent of looping over ``np.unique(prob)`` and
    calling the scikit-learn / imbalanced-learn scorers at each threshold: the
    confusion-matrix counts at every threshold are obtained from cumulative sums
    of the probability-sorted labels, which is O(n log n) instead of O(n^2) and
    matters on large test sets where boosting models produce many distinct
    probabilities. The returned maxima are numerically identical to the loop.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype="float64")
    n_pos = int(y.sum())
    n_neg = y.shape[0] - n_pos

    # Sort by descending probability so that, as the threshold is lowered, each
    # sample is added to the positive predictions exactly once.
    order = np.argsort(-prob, kind="mergesort")
    y_sorted = y[order]
    prob_sorted = prob[order]

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    # Evaluate only at distinct probabilities: keep the last index of each run of
    # equal values, where every sample with prob >= that value has been counted.
    last_of_run = np.r_[np.diff(prob_sorted) != 0, True]
    tp = tp_cum[last_of_run].astype("float64")
    fp = fp_cum[last_of_run].astype("float64")
    fn = n_pos - tp
    tn = n_neg - fp

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(tp), where=denom > 0)

    tpr = tp / n_pos if n_pos else np.zeros_like(tp)
    tnr = tn / n_neg if n_neg else np.zeros_like(tn)
    ba = (tpr + tnr) / 2
    g_mean = np.sqrt(tpr * tnr)

    return float(mcc.max()), float(ba.max()), float(g_mean.max())


def evaluate_model_on_test_set(search, X, y):
    """
    Evaluate a fitted model on the test set using 5 bootstrap samples (60%,
    with replacement) to estimate dispersion.

    Returns a flat list:
        [mean_roc_auc, std_roc_auc,
         mean_avg_precision, std_avg_precision,
         mean_precision, std_precision,
         mean_recall, std_recall,
         mean_f1, std_f1,
         mean_mcc, std_mcc,
         mean_ba, std_ba,
         mean_brier, std_brier,
         mean_gmean, std_gmean,
         mean_threshold, std_threshold]

    Precision, recall, and threshold are computed at the F1-optimal threshold
    per bootstrap sample. Other threshold dependent metrics are obtained using
    the threshold that maximizes them.
    """
    # To obtain a dispersion value, we take bootstrapped samples
    # of a portion of the test set.
    X = np.asarray(X)
    y = np.asarray(y)
    n = int(0.6 * len(X))

    roc, ap, precision, recall, f1score, mcc, ba, brier, gmean, thresh = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for seed in range(1, 6):
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=True)
        xs, ys = X[idx], y[idx]
        prob = search.predict_proba(xs)[:, 1]
        preds, t = predict_class(ys, prob)

        roc.append(roc_auc_score(ys, prob))
        ap.append(average_precision_score(ys, prob))
        precision.append(precision_score(ys, preds))
        recall.append(recall_score(ys, preds))
        f1score.append(f1_score(ys, preds))
        mcc_val, ba_val, gmean_val = calculate_classif_metrics(ys, prob)
        mcc.append(mcc_val)
        ba.append(ba_val)
        gmean.append(gmean_val)
        brier.append(brier_score_loss(ys, prob))
        thresh.append(t)

    results = [
        np.mean(roc),
        np.std(roc),
        np.mean(ap),
        np.std(ap),
        np.mean(precision),
        np.std(precision),
        np.mean(recall),
        np.std(recall),
        np.mean(f1score),
        np.std(f1score),
        np.mean(mcc),
        np.std(mcc),
        np.mean(ba),
        np.std(ba),
        np.mean(brier),
        np.std(brier),
        np.mean(gmean),
        np.std(gmean),
        np.mean(thresh),
        np.std(thresh),
    ]


    return results, {
        "roc": roc,
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1score,
        "mcc": mcc,
        "ba": ba,
        "brier": brier,
        "gmean": gmean,
        "thresh": thresh,
    }

