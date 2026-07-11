import pandas as pd


def create_df(scores_dict, dataset, models):
    """
    Create a DataFrame of evaluation scores for a given dataset and set of models.

    Parameters
    ----------
    scores_dict : dict
        Nested dictionary where keys are dataset names and values are
        dictionaries of scores per model, as produced by the evaluation functions.
    dataset : str
        The dataset key to look up in scores_dict.
    models : list of str
        List of model names to include as rows in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with models as rows and evaluation metrics as columns.
        Missing values are filled with 0.
    """
    df = pd.DataFrame(
        scores_dict[dataset],
        index=[
            "roc",
            "roc_std",
            "ap",
            "ap_std",
            "precision",
            "precision_std",
            "recall",
            "recall_std",
            "f1_score",
            "f1_std",
            "mcc",
            "mcc_std",
            "ba",
            "ba_std",
            "brier",
            "brier_std",
            "gmean",
            "gmean_std",
            "thresh",
            "tresh_std",
        ],
    )
    df = df.T
    return df.loc[models].fillna(0)


def create_fold_df(folds_dict, dataset, metric):
    """
    Create a DataFrame of per-fold metric values for a given dataset and metric.

    Parameters
    ----------
    folds_dict : dict
        Nested dictionary where keys are dataset names and values are
        dictionaries keyed by model name, each mapping metric names to a list
        of per-fold (bootstrap sample) values, as produced by
        `evaluate_model_on_test_set` and saved as `results_folds` by the
        evaluation scripts.
    dataset : str
        The dataset key to look up in folds_dict.
    metric : str
        Name of the metric to extract (e.g. 'roc', 'ap', 'precision',
        'recall', 'f1', 'mcc', 'ba', 'brier', 'gmean', 'thresh').

    Returns
    -------
    pd.DataFrame
        DataFrame where each row is a bootstrap fold and each column is a
        model.
    """
    return pd.DataFrame(
        {model: values[metric] for model, values in folds_dict[dataset].items()}
    )


def best_performance_summary(
    scores_dict,
    datasets,
    metric,
    metric_std,
    factor = 3,
):
    """
    Build a styled summary DataFrame comparing baseline models against their
    best cost-sensitive learning (CSL) or resampled variant for a given metric.

    For each dataset and base model, finds the CSL/resampled variant with the highest
    metric value, computes the difference, and applies conditional highlighting:
    - Orange: the single highest metric value across all models for that dataset
      (in either the baseline or CSL/resampled column).
    - Red: the CSL improvement exceeds `factor` times the standard deviation of
      both the baseline and the CSL variant.
    - Green: the CSL improvement exceeds 1 times the standard deviation of
      both the baseline and the CSL variant, but not `factor` times.
    - Yellow: the CSL improvement is positive but does not exceed 1 standard
      deviation of both models.

    Parameters
    ----------
    scores_dict : dict
        Nested dict keyed by dataset name, then model name, containing
        evaluation score arrays as produced by the training pipeline.
    datasets : list of str
        Ordered list of dataset names to include in the summary.
    metric : str
        Name of the metric column to compare (e.g. 'roc', 'ap', 'brier').
    metric_std : str
        Name of the corresponding standard deviation column (e.g. 'roc_std').
    factor : int
        Multiplier applied to the baseline std for the red threshold. Default is 3,
        meaning red appears when the improvement exceeds 3 * std.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled DataFrame with one row per (dataset, base model) combination.
    """

    base_models = ["rf", "ada", "gbm", "cat", "lgbm", "xgb"]

    rows = []
    for data in datasets:
        all_models = list(scores_dict[data].keys())
        df = create_df(scores_dict, data, all_models)
        for model in base_models:
            base_val = df.loc[model, metric]
            base_std = df.loc[model, metric_std]
            variants = [m for m in all_models if m.startswith(model + "_")]
            best_variant = max(variants, key=lambda v: df.loc[v, metric])
            best_val = df.loc[best_variant, metric]
            best_std = df.loc[best_variant, metric_std]
            diff = best_val - base_val
            rows.append(
                {
                    "dataset": data,
                    "model": model,
                    metric: base_val,
                    metric_std: base_std,
                    "best_csl_variant": best_variant,
                    f"best_csl_{metric}": best_val,
                    f"best_csl_{metric_std}": best_std,
                    f"{metric}_diff": diff,
                }
            )

    result = pd.DataFrame(rows)

    cols = list(result.columns)
    base_idx = cols.index(metric)
    csl_idx = cols.index(f"best_csl_{metric}")
    diff_idx = cols.index(f"{metric}_diff")

    dataset_max = (
        result.groupby("dataset")[[metric, f"best_csl_{metric}"]].max().max(axis=1)
    )

    def style_row(row):
        styles = [""] * len(row)
        diff = row[f"{metric}_diff"]
        std = row[metric_std]
        std_csl = row[f"best_csl_{metric_std}"]
        if diff > factor * std and diff > factor * std_csl:
            styles[diff_idx] = "background-color: red"
        elif diff > std and diff > std_csl:
            styles[diff_idx] = "background-color: lightgreen"
        elif diff > 0:
            styles[diff_idx] = "background-color: yellow"
        best = dataset_max[row["dataset"]]
        if row[metric] == best:
            styles[base_idx] = "background-color: orange"
        elif row[f"best_csl_{metric}"] == best:
            styles[csl_idx] = "background-color: orange"
        return styles

    return result.style.apply(style_row, axis=1)
