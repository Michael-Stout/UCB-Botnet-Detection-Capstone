"""Visualization helpers for botnet detection analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    auc, confusion_matrix, precision_recall_curve, roc_curve,
)


def apply_log_scale_with_stats(ax, data_series, axis='x', feature_name='Feature', logger=None, log_fn=None):
    """
    Apply log or symlog scale based on whether data has non-positive values.
    Optionally logs stats via log_fn(message, logger).
    """
    data_min = data_series.min()

    if data_min <= 0:
        scale = 'symlog'
        kwargs = {'linthresh': 1e-2}
    else:
        scale = 'log'
        kwargs = {}

    if axis == 'x':
        ax.set_xscale(scale, **kwargs)
    else:
        ax.set_yscale(scale, **kwargs)

    if log_fn and logger:
        stats = data_series.describe().to_dict()
        log_fn(f"Applied {scale} scale for {feature_name}. Stats: {stats}", logger, level='info')


def plot_gains_chart(y_true, y_prob, model_name, save_path=None):
    """Plot a cumulative gains chart."""
    sort_idx = np.argsort(y_prob)[::-1]
    y_sorted = np.array(y_true)[sort_idx]

    cum_pos = np.cumsum(y_sorted)
    total_pos = cum_pos[-1] if len(cum_pos) > 0 else 0

    if total_pos == 0:
        return

    gains = cum_pos / total_pos
    fraction = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fraction, gains, label=model_name, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    ax.set_xlabel('Fraction of Samples')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Gains Chart — {model_name}')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Botnet'],
                yticklabels=['Normal', 'Botnet'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_prob, model_name, save_path=None):
    """Plot ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {model_name}')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_pr_curve(y_true, y_prob, model_name, save_path=None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2, label=model_name)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {model_name}')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_calibration_curve(y_true, y_prob, model_name, save_path=None):
    """Plot calibration curve."""
    fig, ax = plt.subplots(figsize=(7, 5))
    CalibrationDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
    ax.set_title(f'Calibration Curve — {model_name}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
