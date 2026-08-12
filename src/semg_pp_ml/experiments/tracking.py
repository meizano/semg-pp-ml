"""
MLflow integration for experiment tracking.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class MLflowTracker:
    """Wrapper for MLflow experiment tracking."""

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        if not HAS_MLFLOW:
            print("[MLflow] Warning: mlflow not installed. Tracking disabled.")
            return
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.active_run = None

    def get_or_create_experiment(self, name: str) -> str:
        if not HAS_MLFLOW:
            return ""
        exp = mlflow.get_experiment_by_name(name)
        if exp:
            return exp.experiment_id
        return mlflow.create_experiment(name)

    def start_run(
        self,
        experiment_name: str,
        run_name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        if not HAS_MLFLOW:
            return ""
        exp_id = self.get_or_create_experiment(experiment_name)
        self.active_run = mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags)
        return self.active_run.info.run_id

    def end_run(self, status: str = "FINISHED"):
        if HAS_MLFLOW and self.active_run:
            mlflow.end_run(status=status)
            self.active_run = None

    def log_preprocessing(self, config: Dict, metrics: Dict):
        if not HAS_MLFLOW or not self.active_run:
            return
        for k, v in config.items():
            if isinstance(v, (str, int, float, bool)):
                mlflow.log_param(f"prep_{k}", v)
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                mlflow.log_metric(f"prep_{k}", v)

    def log_training(
        self,
        model_type: str,
        hyperparams: Dict,
        metrics: Dict,
        model=None,
        model_path: Optional[str] = None
    ):
        if not HAS_MLFLOW or not self.active_run:
            return
        for k, v in hyperparams.items():
            if isinstance(v, (str, int, float, bool)):
                mlflow.log_param(f"model_{k}", v)
        mlflow.log_param("model_type", model_type)
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                mlflow.log_metric(k, v)


def auto_log_training(model, X_train, y_train, X_val, y_val, **kwargs):
    if HAS_MLFLOW:
        mlflow.pytorch.autolog()
        mlflow.sklearn.autolog()
    return model.fit(X_train, y_train, validation_data=(X_val, y_val), **kwargs)