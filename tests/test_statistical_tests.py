import pandas as pd
import pytest
from functions.statistical_tests import apply_friedman_test, apply_nemenyi_test

def test_apply_friedman_test():
    data = {
        'model1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'model2': [0.2, 0.3, 0.4, 0.5, 0.6],
        'model3': [0.3, 0.4, 0.5, 0.6, 0.7]
    }
    df = pd.DataFrame(data)
    statistic, pvalue = apply_friedman_test(df)
    assert isinstance(statistic, float)
    assert isinstance(pvalue, float)
    assert statistic >= 0
    assert 0 <= pvalue <= 1

def test_apply_nemenyi_test():
    data = {
        'model1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'model2': [0.2, 0.3, 0.4, 0.5, 0.6],
        'model3': [0.3, 0.4, 0.5, 0.6, 0.7]
    }
    df = pd.DataFrame(data)
    p_values = apply_nemenyi_test(df)
    assert isinstance(p_values, pd.DataFrame)
    assert list(p_values.columns) == ['model1', 'model2', 'model3']
    assert list(p_values.index) == ['model1', 'model2', 'model3']
    assert p_values.loc['model1', 'model2'] >= 0
    assert p_values.loc['model1', 'model2'] <= 1
