"""Model training and prediction utilities for botnet detection."""

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, get_model_grids


def get_models_and_grids(random_state=RANDOM_STATE):
    """Return the dict of model names -> {model, param_grid}."""
    return get_model_grids()


def train_models(X_train, y_train, models_and_grids, cv=5, scoring='f1', logger=None, log_fn=None):
    """
    Train all models using GridSearchCV.

    Returns:
        results: dict of model_name -> metrics dict
        trained_models: dict of model_name -> fitted pipeline
    """
    results = {}
    trained_models = {}

    for model_name, config in models_and_grids.items():
        if log_fn and logger:
            log_fn(f"Training {model_name}...", logger, level='info')

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', config['model'])
        ])

        grid_search = GridSearchCV(
            pipeline,
            config['param_grid'],
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        start = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start

        best_model = grid_search.best_estimator_
        trained_models[model_name] = best_model

        results[model_name] = {
            'best_params': grid_search.best_params_,
            'train_time_sec': train_time,
            'cv_f1': grid_search.best_score_,
        }

        if log_fn and logger:
            log_fn(
                f"  {model_name}: CV F1={grid_search.best_score_:.4f}, "
                f"Time={train_time:.2f}s, Params={grid_search.best_params_}",
                logger, level='info'
            )

    return results, trained_models


def evaluate_model(y_test, y_pred, y_prob=None):
    """
    Compute standard classification metrics.

    Returns a dict with accuracy, precision, recall, f1, specificity,
    and optionally roc_auc, log_loss, mAP.
    """
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'test_accuracy': acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
        'specificity': specificity,
        'confusion_matrix': str(cm).replace('\n', ' '),
    }

    if y_prob is not None:
        try:
            metrics['log_loss'] = log_loss(y_test, y_prob)
        except ValueError:
            metrics['log_loss'] = None
        metrics['mAP'] = average_precision_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        metrics['test_roc_auc'] = auc(fpr, tpr)
    else:
        metrics['log_loss'] = None
        metrics['mAP'] = None
        metrics['test_roc_auc'] = None

    return metrics


def predict_all(trained_models, X_test):
    """Generate predictions for all trained models. Returns dict of model_name -> y_pred."""
    return {name: model.predict(X_test) for name, model in trained_models.items()}
