"""
Single experiment runner for sEMG classification.
"""
from __future__ import annotations

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict



from ..config import ExperimentConfig, load_config
from ..data.loader import prepare_dataset
from ..preprocessing.pipeline import PreprocessingPipeline
from ..models.registry import (
    build_model, is_classical_model, count_params, save_model,
    merge_hyperparameters, get_default_hyperparameters
)
from ..metrics.classification import compute_all_classification_metrics
from ..visualization.training import plot_training_curves, analyze_overfitting
from ..visualization.confusion import plot_confusion_matrix
from ..utils.timer import Timer





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


def run_single_experiment(
    config: ExperimentConfig,
    prep_method: Optional[str] = None,
    model_type: Optional[str] = None,
    save_model_flag: bool = True,
    skip_visualization: bool = False,
    output_dir: Optional[Path] = None
) -> ExperimentResult:
    """
    Run a single preprocessing + training experiment.
    """
    prep_method = prep_method or config.preprocessing.method
    model_type = model_type or config.model.__dict__.get('type', 'svm')

    log = {}
    output_dir = output_dir or config.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config.preprocessing.method = prep_method

    print(f"\n{'='*70}")
    print(f"[EXPERIMENT] {prep_method.upper()} + {model_type.upper()}")
    print(f"{'='*70}")

    try:
        # 1. Preprocessing pipeline
        with Timer("Preprocessing", log):
            pipeline = PreprocessingPipeline(config)
            prep_result = pipeline.prepare_for_training(method=prep_method)

        X_tr, y_tr = prep_result['X_tr'], prep_result['y_tr']
        X_te, y_te = prep_result['X_te'], prep_result['y_te']
        le, n_classes = prep_result['le'], prep_result['n_classes']

        prep_metrics = {f"prep_{k}": v for k, v in prep_result['preprocessing_result'].metrics.items()}

        # 2. Build model
        with Timer("Build Model", log):
            input_shape = (config.signal.window_size, config.dataset.n_channels)
            model_params = merge_hyperparameters(model_type, config.model.model_dump())
            model = build_model(model_type, input_shape, n_classes, config.training.lr, model_params)

            n_params, trainable_params, _ = count_params(model)

        # 3. Train model
        with Timer("Training", log):
            if is_classical_model(model_type):
                X_tr_2d = X_tr.reshape(X_tr.shape[0], -1)
                y_tr_1d = np.argmax(y_tr, axis=1)

                t0 = time.perf_counter()
                model.fit(X_tr_2d, y_tr_1d)
                train_time = time.perf_counter() - t0

                epochs_run = 1
                avg_throughput = len(X_tr_2d) / train_time
                avg_epoch_time = train_time
                history = None
            else:
                t0 = time.perf_counter()
                history = model.fit(
                    X_tr, y_tr,
                    validation_split=config.training.validation_split,
                    epochs=config.training.epochs,
                    batch_size=config.training.batch_size,
                    verbose=1
                )
                train_time = time.perf_counter() - t0
                epochs_run = len(history.history['loss'])
                avg_epoch_time = train_time / max(1, epochs_run)
                avg_throughput = len(X_tr) / max(1, avg_epoch_time)

        # 4. Evaluate
        with Timer("Evaluation", log):
            t0_inf = time.perf_counter()

            if is_classical_model(model_type):
                X_te_2d = X_te.reshape(X_te.shape[0], -1)
                y_prob = model.predict_proba(X_te_2d)
            else:
                y_prob = model.predict_proba(X_te)

            t_inf = time.perf_counter() - t0_inf
            latency_ms = (t_inf / len(X_te)) * 1000.0

            y_pred = np.argmax(y_prob, axis=1)
            y_true = np.argmax(y_te, axis=1)

            metrics = compute_all_classification_metrics(
                y_true, y_pred, y_prob, y_te, n_classes,
                class_names=[f"Class-{c}" for c in le.classes_]
            )

            metrics.update({
                'model_type': model_type,
                'prep_method': prep_method,
                'model_params_total': int(n_params),
                'model_params_trainable': int(trainable_params),
                'inference_latency_ms': round(latency_ms, 4),
                'epochs_run': epochs_run,
                'avg_throughput_samples_per_sec': round(avg_throughput, 1),
                'avg_epoch_time_sec': round(avg_epoch_time, 2),
            })
            metrics.update(prep_metrics)

        # 5. Save model
        model_path = None
        if save_model_flag:
            ext = ".pkl" if is_classical_model(model_type) else ".pt"
            model_path = str(output_dir / f"model_{model_type}_{prep_method}{ext}")
            save_model(model, model_path, model_type)

        # 6. Visualizations
        if not skip_visualization:
            viz_dir = output_dir / "visualizations" / f"{model_type}_{prep_method}"
            viz_dir.mkdir(parents=True, exist_ok=True)

            if not is_classical_model(model_type) and history:
                plot_training_curves(history, viz_dir, model_type, prep_method)

            plot_confusion_matrix(
                y_true, y_pred,
                [f"Class-{c}" for c in le.classes_],
                viz_dir, model_type, prep_method
            )

        # 7. Prepare result
        result = ExperimentResult(
            model_type=model_type,
            prep_method=prep_method,
            config=config.model_dump(mode='json'),
            metrics=metrics,
            training_time=log.get("Training", 0),
            inference_time=t_inf,
            epochs_run=epochs_run,
            n_params=n_params,
            trainable_params=trainable_params,
            history=history.history if history else None,
            model_path=model_path,
            status="OK"
        )

        print(f"\n[RESULT] {model_type} + {prep_method}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Latency:   {metrics['inference_latency_ms']:.2f} ms")

        return result

    except Exception as e:
        print(f"[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return ExperimentResult(
            model_type=model_type,
            prep_method=prep_method,
            config=config.model_dump(mode='json'),
            metrics={},
            training_time=0,
            inference_time=0,
            epochs_run=0,
            n_params=0,
            trainable_params=0,
            status="FAILED",
            error=str(e)
        )


def run_experiment_cli(
    config_path: str,
    prep_method: str,
    model_type: str,
    output_dir: str,
    save_model: bool = True,
    no_viz: bool = False
) -> None:
    """CLI entry point for single experiment."""
    config = load_config(config_path)
    config.preprocessing.method = prep_method

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = run_single_experiment(
        config, prep_method, model_type,
        save_model_flag=save_model,
        skip_visualization=no_viz,
        output_dir=output_path
    )

    with open(output_path / f"result_{model_type}_{prep_method}.json", 'w') as f:
        json.dump({
            'model_type': result.model_type,
            'prep_method': result.prep_method,
            'metrics': result.metrics,
            'training_time': result.training_time,
            'inference_time': result.inference_time,
            'epochs_run': result.epochs_run,
            'n_params': result.n_params,
            'status': result.status,
            'error': result.error,
        }, f, indent=2)

    if result.status == "OK":
        print(f"\n[OK] Experiment complete. Results saved to {output_path}")
    else:
        print(f"\n[FAILED] Experiment failed: {result.error}")
        exit(1)