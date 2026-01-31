"""Statistical tests and evaluation utilities for model comparison."""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test comparing two classifiers with continuity correction.

    Returns (chi2_statistic, p_value).
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B correct

    if b + c == 0:
        return 0.0, 1.0

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


def pairwise_mcnemar(y_true, model_predictions):
    """
    Run McNemar's test for all pairs of models.

    Args:
        y_true: true labels
        model_predictions: dict of model_name -> y_pred array

    Returns a DataFrame of p-values (symmetric matrix).
    """
    model_names = list(model_predictions.keys())
    n = len(model_names)
    matrix = pd.DataFrame(
        np.ones((n, n)),
        index=model_names,
        columns=model_names
    )

    for i in range(n):
        for j in range(i + 1, n):
            _, p_val = mcnemar_test(
                y_true,
                model_predictions[model_names[i]],
                model_predictions[model_names[j]]
            )
            matrix.iloc[i, j] = p_val
            matrix.iloc[j, i] = p_val

    return matrix
