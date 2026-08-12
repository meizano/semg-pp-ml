"""
Regression metrics for sEMG (if doing regression tasks).
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.nan_to_num(np.array(y_true), nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(np.array(y_pred), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.nan_to_num(np.array(y_true), nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(np.array(y_pred), nan=0.0, posinf=0.0, neginf=0.0)
    return float(mean_absolute_error(y_true, y_pred))


def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse_val = compute_rmse(y_true, y_pred)
    y_range = np.max(y_true) - np.min(y_true)
    if y_range == 0:
        return 0.0
    return rmse_val / float(y_range)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.nan_to_num(np.array(y_true), nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(np.array(y_pred), nan=0.0, posinf=0.0, neginf=0.0)
    return float(r2_score(y_true, y_pred))


def compute_all_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'nrmse': compute_nrmse(y_true, y_pred),
        'r2': compute_r2(y_true, y_pred),
    }