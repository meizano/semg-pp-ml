"""
Single experiment runner for sEMG classification.

Two entry points share one training/evaluation path:

- ``run_single_experiment``: preprocess + train + evaluate one (preprocessing, model) pair.
- ``train_and_evaluate``: train + evaluate one model on already-prepared data. The grid
  search calls this so each preprocessing method is computed only once.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config import ExperimentConfig
from ..metrics.classification import compute_all_classification_metrics
from ..models.registry import (
    build_model, count_params, is_classical_model, merge_hyperparameters, save_model,
)
from ..preprocessing.pipeline import PreprocessingPipeline
from ..visualization.confusion import plot_confusion_matrix
from ..visualization.training import plot_training_curves


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    model_type: str
    prep_method: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    training_time: float
    inference_time: float
    epochs_run: int
    n_params: int
    trainable_params: int
    history: Optional[Dict] = None
    model_path: Optional[str] = None
    status: str = "OK"
    error: Optional[str] = None

    @classmethod
    def failed(cls, config: ExperimentConfig, model_type: str, prep_method: str, error: str) -> "ExperimentResult":
        return cls(
            model_type=model_type, prep_method=prep_method,
            config=config.model_dump(mode="json"), metrics={},
            training_time=0.0, inference_time=0.0, epochs_run=0,
            n_params=0, trainable_params=0, status="FAILED", error=error,
        )


def prepare_dataset_for_prep(config: ExperimentConfig, prep_method: str) -> Dict[str, Any]:
    """Run preprocessing and windowing once for ``prep_method`` (uses the on-disk cache)."""
    config = config.model_copy(deep=True)
    config.preprocessing.method = prep_method
    return PreprocessingPipeline(config).prepare_for_training(method=prep_method)


def train_and_evaluate(
    config: ExperimentConfig,
    prep_data: Dict[str, Any],
    model_type: str,
    *,
    save_model_flag: bool = True,
    skip_visualization: bool = False,
    output_dir: Optional[Path] = None,
    verbose: int = 1,
) -> ExperimentResult:
    """Train ``model_type`` on prepared data and evaluate it on the test split.

    Never raises for model/training errors: they are returned as ``status="FAILED"``
    so one bad model does not stop a grid search.
    """
    prep_method = prep_data["method"]
    output_dir = Path(output_dir or config.output_dir)
    try:
        X_tr, y_tr = prep_data["X_tr"], prep_data["y_tr"]
        X_te, y_te = prep_data["X_te"], prep_data["y_te"]
        le, n_classes = prep_data["le"], prep_data["n_classes"]
        classical = is_classical_model(model_type)

        # Build
        input_shape = (config.signal.window_size, config.dataset.n_channels)
        model_params = merge_hyperparameters(model_type, config.model.model_dump())
        model = build_model(model_type, input_shape, n_classes, config.training.lr, model_params)
        n_params, trainable_params, _ = count_params(model)

        # Train
        t0 = time.perf_counter()
        if classical:
            X_tr_2d = X_tr.reshape(X_tr.shape[0], -1)
            model.fit(X_tr_2d, np.argmax(y_tr, axis=1))
            train_time = time.perf_counter() - t0
            history = None
            epochs_run = 1
            avg_epoch_time = train_time
            avg_throughput = len(X_tr_2d) / max(train_time, 1e-9)
        else:
            history = model.fit(
                X_tr, y_tr,
                validation_split=config.training.validation_split,
                epochs=config.training.epochs,
                batch_size=config.training.batch_size,
                verbose=verbose,
            )
            train_time = time.perf_counter() - t0
            epochs_run = len(history.history["loss"])
            avg_epoch_time = train_time / max(1, epochs_run)
            avg_throughput = len(X_tr) / max(avg_epoch_time, 1e-9)

        # Evaluate
        X_te_model = X_te.reshape(X_te.shape[0], -1) if classical else X_te
        t0_inf = time.perf_counter()
        y_prob = model.predict_proba(X_te_model)
        t_inf = time.perf_counter() - t0_inf
        latency_ms = (t_inf / max(len(X_te), 1)) * 1000.0

        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(y_te, axis=1)
        class_names = [f"Class-{c}" for c in le.classes_]
        metrics = compute_all_classification_metrics(
            y_true, y_pred, y_prob, y_te, n_classes, class_names=class_names
        )
        metrics.update({f"prep_{k}": v for k, v in prep_data["preprocessing_result"].metrics.items()})
        metrics.update({
            "model_type": model_type,
            "prep_method": prep_method,
            "model_params_total": int(n_params),
            "model_params_trainable": int(trainable_params),
            "inference_latency_ms": round(latency_ms, 4),
            "epochs_run": epochs_run,
            "avg_throughput_samples_per_sec": round(avg_throughput, 1),
            "avg_epoch_time_sec": round(avg_epoch_time, 2),
        })

        # Save model
        model_path = None
        if save_model_flag:
            ext = ".pkl" if classical else ".pt"
            model_path = str(output_dir / f"model_{model_type}_{prep_method}{ext}")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_model(model, model_path, model_type)

        # Visualise
        if not skip_visualization:
            viz_dir = output_dir / "visualizations" / f"{model_type}_{prep_method}"
            viz_dir.mkdir(parents=True, exist_ok=True)
            if history is not None:
                plot_training_curves(history, viz_dir, model_type, prep_method)
            plot_confusion_matrix(y_true, y_pred, class_names, viz_dir, model_type, prep_method)

        return ExperimentResult(
            model_type=model_type,
            prep_method=prep_method,
            config=config.model_dump(mode="json"),
            metrics=metrics,
            training_time=train_time,
            inference_time=t_inf,
            epochs_run=epochs_run,
            n_params=n_params,
            trainable_params=trainable_params,
            history=history.history if history is not None else None,
            model_path=model_path,
            status="OK",
        )
    except Exception as e:  # noqa: BLE001 - a failed model must not abort the whole run
        if verbose:
            traceback.print_exc()
        return ExperimentResult.failed(config, model_type, prep_method, str(e))


def run_single_experiment(
    config: ExperimentConfig,
    prep_method: Optional[str] = None,
    model_type: Optional[str] = None,
    save_model_flag: bool = True,
    skip_visualization: bool = False,
    output_dir: Optional[Path] = None,
) -> ExperimentResult:
    """Run one preprocessing + training + evaluation experiment."""
    config = config.model_copy(deep=True)
    prep_method = prep_method or config.preprocessing.method
    model_type = model_type or config.model.type
    config.preprocessing.method = prep_method
    output_dir = Path(output_dir or config.output_dir)

    print(f"\n{'=' * 70}\n[EXPERIMENT] {prep_method.upper()} + {model_type.upper()}\n{'=' * 70}")
    try:
        prep_data = prepare_dataset_for_prep(config, prep_method)
    except FileNotFoundError as e:  # missing data: a clear message is enough, no traceback
        print(f"[ERROR] {e}")
        return ExperimentResult.failed(config, model_type, prep_method, str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return ExperimentResult.failed(config, model_type, prep_method, f"preprocessing failed: {e}")

    result = train_and_evaluate(
        config, prep_data, model_type,
        save_model_flag=save_model_flag,
        skip_visualization=skip_visualization,
        output_dir=output_dir,
        verbose=1,
    )
    if result.status == "OK":
        m = result.metrics
        print(f"\n[RESULT] {model_type} + {prep_method}")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  F1 Score:  {m['f1_score']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  Latency:   {m['inference_latency_ms']:.2f} ms")
    return result


def save_result_json(result: ExperimentResult, path: Path) -> None:
    """Write a compact JSON summary of one result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": result.model_type,
            "prep_method": result.prep_method,
            "metrics": result.metrics,
            "training_time": result.training_time,
            "inference_time": result.inference_time,
            "epochs_run": result.epochs_run,
            "n_params": result.n_params,
            "status": result.status,
            "error": result.error,
        }, f, indent=2, default=str)
