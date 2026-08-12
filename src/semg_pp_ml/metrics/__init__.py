"""
Metrics package for sEMG classification and regression.
"""
from .classification import (
    compute_accuracy,
    compute_f1,
    compute_precision,
    compute_recall,
    compute_specificity,
    compute_auc_roc,
    compute_logloss,
    compute_confusion_matrix_detail,
    compute_classification_report,
    compute_all_classification_metrics,
)
from .regression import (
    compute_rmse,
    compute_mae,
    compute_nrmse,
    compute_r2,
    compute_all_regression_metrics,
)

__all__ = [
    # Classification
    "compute_accuracy",
    "compute_f1",
    "compute_precision",
    "compute_recall",
    "compute_specificity",
    "compute_auc_roc",
    "compute_logloss",
    "compute_confusion_matrix_detail",
    "compute_classification_report",
    "compute_all_classification_metrics",
    # Regression
    "compute_rmse",
    "compute_mae",
    "compute_nrmse",
    "compute_r2",
    "compute_all_regression_metrics",
]