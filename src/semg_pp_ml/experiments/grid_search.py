"""
Optimized Grid Search for sEMG experiments.
Preprocessing is done once per method, then all models are tested.
"""
from __future__ import annotations

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..config import ExperimentConfig, GridSearchConfig
from ..preprocessing.pipeline import PreprocessingPipeline
from ..models.registry import is_classical_model, merge_hyperparameters
from .runner import run_single_experiment, ExperimentResult


@dataclass
class GridSearchResult:
    """Container for grid search results."""
    results: List[ExperimentResult]
    summary_df: pd.DataFrame
    best_by_metric: Dict[str, ExperimentResult]
    total_time: float
    config: GridSearchConfig


def prepare_dataset_for_prep(
    config: ExperimentConfig,
    prep_method: str
) -> Dict[str, Any]:
    """
    Prepare dataset for a specific preprocessing method.
    Done ONCE per preprocessing method.
    """
    config.preprocessing.method = prep_method
    pipeline = PreprocessingPipeline(config)
    return pipeline.prepare_for_training(method=prep_method)


def run_model_on_prepped_data(
    config: ExperimentConfig,
    prep_data: Dict[str, Any],
    model_type: str,
    skip_visualization: bool = False
) -> ExperimentResult:
    """Run a single model on pre-prepared data."""
    # Create a modified config for this model
    model_config = ExperimentConfig(**config.model_dump(mode='json'))
    model_config.preprocessing.method = prep_data['method']

    # Override model hyperparameters
    model_params = merge_hyperparameters(model_type, config.model.model_dump())
    model_config.model = type('obj', (object,), model_params)()

    # Run experiment with prepped data
    # We need to modify run_single_experiment to accept prepped data
    return _run_model_with_prepped_data(model_config, prep_data, model_type, skip_visualization)


def _run_model_with_prepped_data(
    config: ExperimentConfig,
    prep_data: Dict[str, Any],
    model_type: str,
    skip_visualization: bool
) -> ExperimentResult:
    """Internal: run model with pre-prepared dataset."""
    from ..models.registry import build_model, count_params, save_model
    from ..metrics.classification import compute_all_classification_metrics
    from ..visualization.training import plot_training_curves
    from ..visualization.confusion import plot_confusion_matrix
    from ..utils.timer import Timer


    log = {}
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr = prep_data['X_tr'], prep_data['y_tr']
    X_te, y_te = prep_data['X_te'], prep_data['y_te']
    le, n_classes = prep_data['le'], prep_data['n_classes']
    prep_method = prep_data['method']

    try:
        # Build model
        input_shape = (config.signal.window_size, config.dataset.n_channels)
        model_params = merge_hyperparameters(model_type, config.model.model_dump())
        model = build_model(model_type, input_shape, n_classes, config.training.lr, model_params)
        n_params, trainable_params, _ = count_params(model)

        # Train
        if is_classical_model(model_type):
            X_tr_2d = X_tr.reshape(X_tr.shape[0], -1)
            y_tr_1d = np.argmax(y_tr, axis=1)

            t0 = time.perf_counter()
            model.fit(X_tr_2d, y_tr_1d)
            train_time = time.perf_counter() - t0

            epochs_run = 1
            avg_throughput = len(X_tr_2d) / train_time
            history = None
        else:
            t0 = time.perf_counter()
            history = model.fit(
                X_tr, y_tr,
                validation_split=config.training.validation_split,
                epochs=config.training.epochs,
                batch_size=config.training.batch_size,
                verbose=0
            )
            train_time = time.perf_counter() - t0
            epochs_run = len(history.history['loss'])
            avg_epoch_time = train_time / max(1, epochs_run)
            avg_throughput = len(X_tr) / max(1, avg_epoch_time)

        # Evaluate
        t0_inf = time.perf_counter()
        if is_classical_model(model_type):
            X_te_2d = X_te.reshape(X_te.shape[0], -1)
            y_prob = model.predict_proba(X_te_2d)
        else:
            y_prob = model.predict(X_te, batch_size=config.training.batch_size, verbose=0)
        t_inf = time.perf_counter() - t0_inf
        latency_ms = (t_inf / len(X_te)) * 1000.0

        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(y_te, axis=1)

        metrics = compute_all_classification_metrics(
            y_true, y_pred, y_prob, y_te, n_classes,
            class_names=[f"Class-{c}" for c in le.classes_]
        )

        # Add preprocessing metrics
        prep_metrics = {f"prep_{k}": v for k, v in prep_data['preprocessing_result'].metrics.items()}
        metrics.update(prep_metrics)

        metrics.update({
            'model_type': model_type,
            'prep_method': prep_method,
            'model_params_total': int(n_params),
            'model_params_trainable': int(trainable_params),
            'inference_latency_ms': round(latency_ms, 4),
            'epochs_run': epochs_run,
            'avg_throughput_samples_per_sec': round(avg_throughput, 1),
        })

        # Visualizations
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

        return ExperimentResult(
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
            status="OK"
        )

    except Exception as e:
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


def run_grid_search_optimized(
    config: ExperimentConfig,
    grid_config: Optional[GridSearchConfig] = None,
    skip_visualization: bool = False,
    parallel: bool = False,
    max_workers: int = 2
) -> GridSearchResult:
    """
    Run optimized grid search:
    1. For each preprocessing method, prepare data ONCE
    2. For each model, train on the prepped data
    3. Skip excluded (prep, model) pairs
    """
    grid_config = grid_config or GridSearchConfig()

    print(f"\n{'#'*70}")
    print(f"# GRID SEARCH OPTIMIZED")
    print(f"# Preprocessing: {grid_config.preps}")
    print(f"# Models:        {grid_config.models}")
    print(f"# Excluded:      {grid_config.exclude_pairs}")
    print(f"# Parallel:      {parallel} (workers={max_workers})")
    print(f"{'#'*70}")

    start_time = time.perf_counter()
    results = []

    # Convert exclude pairs to set for fast lookup
    excluded = set(tuple(pair) for pair in grid_config.exclude_pairs)

    for prep in grid_config.preps:
        # Check if any models are valid for this prep
        valid_models = [m for m in grid_config.models if (prep, m) not in excluded]
        if not valid_models:
            print(f"\n[SKIP PREP] All models excluded for '{prep}'")
            continue

        print(f"\n{'='*70}")
        print(f"[PREP] Preparing data for: {prep.upper()}")
        print(f"{'='*70}")

        try:
            # Prepare data ONCE per preprocessing method
            prep_data = prepare_dataset_for_prep(config, prep)
            print(f"  Train: {prep_data['X_tr'].shape}, Test: {prep_data['X_te'].shape}")

            # Run all valid models on this prepped data
            if parallel and len(valid_models) > 1:
                with ProcessPoolExecutor(max_workers=min(max_workers, len(valid_models))) as executor:
                    futures = {
                        executor.submit(
                            _run_model_with_prepped_data, config, prep_data, mdl, skip_visualization
                        ): mdl for mdl in valid_models
                    }
                    for future in as_completed(futures):
                        mdl = futures[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            print(f"[ERROR] {mdl} on {prep}: {e}")
                            results.append(ExperimentResult(
                                model_type=mdl, prep_method=prep,
                                config=config.model_dump(mode='json'),
                                metrics={}, training_time=0, inference_time=0,
                                epochs_run=0, n_params=0, trainable_params=0,
                                status="FAILED", error=str(e)
                            ))
            else:
                for mdl in valid_models:
                    print(f"\n  [MODEL] {mdl.upper()} on {prep.upper()}")
                    result = _run_model_with_prepped_data(config, prep_data, mdl, skip_visualization)
                    results.append(result)

        except Exception as e:
            print(f"[ERROR PREP] Failed to prepare data for '{prep}': {e}")
            for mdl in valid_models:
                results.append(ExperimentResult(
                    model_type=mdl, prep_method=prep,
                    config=config.model_dump(mode='json'),
                    metrics={}, training_time=0, inference_time=0,
                    epochs_run=0, n_params=0, trainable_params=0,
                    status="FAILED", error=f"Prep failed: {e}"
                ))

    total_time = time.perf_counter() - start_time

    # Create summary DataFrame
    summary_rows = []
    for r in results:
        row = {
            'prep': r.prep_method,
            'model': r.model_type,
            'status': r.status,
        }
        if r.status == "OK":
            row.update(r.metrics)
        else:
            row['error'] = r.error
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)

    # Find best by key metrics
    best_by_metric = {}
    if not df[df['status'] == 'OK'].empty:
        ok_df = df[df['status'] == 'OK']
        for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auc_roc_weighted']:
            if metric in ok_df.columns:
                best_idx = ok_df[metric].idxmax()
                best_by_metric[metric] = results[best_idx]

    # Save summary
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "grid_search_summary.csv", index=False)

    print(f"\n{'#'*70}")
    print(f"# GRID SEARCH COMPLETE in {total_time:.1f}s")
    print(f"# Results saved to: {output_dir / 'grid_search_summary.csv'}")
    print(f"{'#'*70}")

    return GridSearchResult(
        results=results,
        summary_df=df,
        best_by_metric=best_by_metric,
        total_time=total_time,
        config=grid_config
    )


def run_grid_search_cli(
    config_path: str,
    grid_config_path: Optional[str] = None,
    output_dir: str = "outputs/grid_search",
    skip_viz: bool = False,
    parallel: bool = False,
    workers: int = 2
) -> None:
    """CLI entry point for grid search."""
    config = load_config(config_path)
    grid_config = GridSearchConfig() if not grid_config_path else load_config(grid_config_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = run_grid_search_optimized(
        config, grid_config, skip_visualization=skip_viz,
        parallel=parallel, max_workers=workers
    )

    # Save detailed results
    import json
    with open(output_path / "grid_search_results.json", 'w') as f:
        json.dump({
            'total_time': result.total_time,
            'config': result.config.model_dump(mode='json'),
            'best_by_metric': {
                k: {'prep': v.prep_method, 'model': v.model_type, 'metrics': v.metrics}
                for k, v in result.best_by_metric.items()
            }
        }, f, indent=2)

    print(f"\n[OK] Grid search complete. Results in {output_path}")


if __name__ == "__main__":
    import argparse
    from ..config import load_config

    parser = argparse.ArgumentParser(description="Run grid search")
    parser.add_argument("--config", default="config/default.yaml", help="Main config")
    parser.add_argument("--grid-config", help="Grid search config")
    parser.add_argument("--output", default="outputs/grid_search", help="Output directory")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualizations")
    parser.add_argument("--parallel", action="store_true", help="Parallel execution")
    parser.add_argument("--workers", type=int, default=2, help="Max workers")
    args = parser.parse_args()

    run_grid_search_cli(args.config, args.grid_config, args.output, args.skip_viz, args.parallel, args.workers)