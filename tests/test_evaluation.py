"""Unit tests for src.evaluation module."""

import numpy as np
import pytest

from src.evaluation import mcnemar_test, pairwise_mcnemar


class TestMcNemarTest:
    def test_identical_predictions(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        chi2, p = mcnemar_test(y_true, y_pred, y_pred)
        assert chi2 == 0.0
        assert p == 1.0

    def test_different_predictions(self):
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
        y_pred_a = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1])  # perfect
        y_pred_b = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0])  # bad
        chi2, p = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert chi2 > 0
        assert p < 1.0

    def test_symmetric(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred_a = np.array([0, 1, 0, 0])
        y_pred_b = np.array([0, 0, 1, 0])
        chi2_ab, p_ab = mcnemar_test(y_true, y_pred_a, y_pred_b)
        chi2_ba, p_ba = mcnemar_test(y_true, y_pred_b, y_pred_a)
        assert chi2_ab == chi2_ba
        assert p_ab == p_ba

    def test_returns_tuple(self):
        y_true = np.array([0, 1])
        result = mcnemar_test(y_true, y_true, y_true)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestPairwiseMcNemar:
    def test_matrix_shape(self):
        y_true = np.array([0, 1, 1, 0])
        preds = {
            'A': np.array([0, 1, 1, 0]),
            'B': np.array([0, 1, 0, 0]),
            'C': np.array([1, 1, 1, 0]),
        }
        matrix = pairwise_mcnemar(y_true, preds)
        assert matrix.shape == (3, 3)
        assert list(matrix.index) == ['A', 'B', 'C']

    def test_diagonal_is_one(self):
        y_true = np.array([0, 1])
        preds = {'A': np.array([0, 1]), 'B': np.array([0, 1])}
        matrix = pairwise_mcnemar(y_true, preds)
        assert matrix.loc['A', 'A'] == 1.0
        assert matrix.loc['B', 'B'] == 1.0

    def test_symmetric_matrix(self):
        y_true = np.array([0, 1, 1, 0, 1])
        preds = {
            'A': np.array([0, 1, 0, 0, 1]),
            'B': np.array([0, 0, 1, 0, 1]),
        }
        matrix = pairwise_mcnemar(y_true, preds)
        assert matrix.loc['A', 'B'] == matrix.loc['B', 'A']
