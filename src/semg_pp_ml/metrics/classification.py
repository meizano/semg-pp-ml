"""
Classification metrics for sEMG gesture recognition.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_auc_score, log_loss
)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, Any]:
    """Compute Specificity (True Negative Rate) per class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    specificity_per_class = []

    for i in range(n_classes):
        tn = np.sum(cm) - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp + 1e-12)
        specificity_per_class.append(float(specificity))

    return {
        'specificity_per_class': specificity_per_class,
        'specificity_mean': float(np.mean(specificity_per_class)),
    }


def compute_auc_roc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> Dict[str, Any]:
    """Compute AUC-ROC using One-vs-Rest for multi-class."""
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        return {'auc_roc_weighted': float(auc)}
    except ValueError:
        # Not all classes present in y_true
        return {'auc_roc_weighted': None}


def compute_logloss(y_true_onehot: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Compute Log Loss / Cross-Entropy."""
    try:
        ll = log_loss(y_true_onehot, y_prob)
        return {'log_loss': float(ll)}
    except ValueError:
        return {'log_loss': None}


def compute_confusion_matrix_detail(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, Any]:
    """Compute TP, FP, FN, TN per class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    detail = {}

    for i in range(n_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - cm[i, i])
        fn = int(cm[i, :].sum() - cm[i, i])
        tn = int(np.sum(cm) - tp - fp - fn)
        detail[f'class_{i}'] = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

    return {
        'confusion_matrix': cm.tolist(),
        'per_class': detail,
    }


def compute_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: Optional[list] = None) -> Dict[str, Any]:
    """Compute full classification report as dict."""
    return classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)


def compute_all_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    y_true_onehot: np.ndarray,
    n_classes: int,
    class_names: Optional[list] = None
) -> Dict[str, Any]:
    """Compute all classification metrics at once."""
    metrics = {}

    metrics['accuracy'] = compute_accuracy(y_true, y_pred)
    metrics['f1_score'] = compute_f1(y_true, y_pred)
    metrics['precision'] = compute_precision(y_true, y_pred)
    metrics['recall'] = compute_recall(y_true, y_pred)

    spec = compute_specificity(y_true, y_pred, n_classes)
    metrics.update(spec)

    auc = compute_auc_roc(y_true, y_prob, n_classes)
    metrics.update(auc)

    ll = compute_logloss(y_true_onehot, y_prob)
    metrics.update(ll)

    cm_detail = compute_confusion_matrix_detail(y_true, y_pred, n_classes)
    metrics['confusion_matrix'] = cm_detail['confusion_matrix']
    metrics['confusion_matrix_detail'] = cm_detail['per_class']

    if class_names:
        metrics['classification_report'] = compute_classification_report(y_true, y_pred, class_names)

    return metrics