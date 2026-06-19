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


def best_performance_summary(
    scores_dict, datasets, metric, metric_std,
):
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
        std_w = row[f"best_csl_{metric_std}"]
        if diff > std and diff > std_w:
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
