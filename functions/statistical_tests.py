import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

def apply_friedman_test(df):
    """
    Applies Friedman's test to compare models based on bootstrap metric values.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame where columns are models and rows are metric values
        for different bootstrap samples.

    Returns
    -------
    statistic : float
        The Friedman test statistic.
    pvalue : float
        The p-value of the test.
    """
    return friedmanchisquare(*[df[col] for col in df.columns])

def apply_nemenyi_test(df):
    """
    Applies Nemenyi test to compare models after a significant Friedman test.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame where columns are models and rows are metric values
        for different bootstrap samples.

    Returns
    -------
    p_values : pandas.DataFrame
        The p-values of the pairwise Nemenyi tests.
    """
    return sp.posthoc_nemenyi_friedman(df)
