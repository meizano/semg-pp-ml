"""
Grid search over preprocessing methods x models.

Preprocessing and windowing run once per method; every valid model is then trained
on that prepared data (see ``runner.train_and_evaluate``).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import ExperimentConfig, GridSearchConfig, load_grid_config, load_config
from .runner import (
    ExperimentResult,
    prepare_dataset_for_prep,
    train_and_evaluate,
)

BEST_METRICS = ("accuracy", "f1_score", "precision", "recall", "auc_roc_weighted")


@dataclass
class GridSearchResult:
    """Container for grid search results."""
    results: List[ExperimentResult]
    summary_df: pd.DataFrame
    best_by_metric: Dict[str, ExperimentResult]
    total_time: float
    config: GridSearchConfig


def _summary_frame(results: List[ExperimentResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row: Dict[str, Any] = {"prep": r.prep_method, "model": r.model_type, "status": r.status}
        if r.status == "OK":
            row.update(r.metrics)
        else:
            row["error"] = r.error
        rows.append(row)
    return pd.DataFrame(rows)


def _best_by_metric(results: List[ExperimentResult]) -> Dict[str, ExperimentResult]:
    ok = [r for r in results if r.status == "OK"]
    best: Dict[str, ExperimentResult] = {}
    for metric in BEST_METRICS:
        candidates = [r for r in ok if metric in r.metrics]
        if candidates:
            best[metric] = max(candidates, key=lambda r: r.metrics[metric])
    return best


def run_grid_search_optimized(
    config: ExperimentConfig,
    grid_config: Optional[GridSearchConfig] = None,
    skip_visualization: bool = False,
    parallel: bool = False,
    max_workers: int = 2,
) -> GridSearchResult:
    """Run every (preprocessing, model) pair that is not excluded.

    Failures are recorded per pair (``status == "FAILED"``) and do not stop the search.
    The summary is written to ``<output_dir>/grid_search_summary.csv``.
    """
    grid_config = grid_config or GridSearchConfig()
    config = config.model_copy(deep=True)
    excluded = {tuple(pair) for pair in grid_config.exclude_pairs}
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}\n# GRID SEARCH\n# Preprocessing: {grid_config.preps}\n"
          f"# Models:        {grid_config.models}\n# Excluded:      {grid_config.exclude_pairs}\n"
          f"# Parallel:      {parallel} (workers={max_workers})\n{'#' * 70}")

    start = time.perf_counter()
    results: List[ExperimentResult] = []

    for prep in grid_config.preps:
        models = [m for m in grid_config.models if (prep, m) not in excluded]
        if not models:
            print(f"\n[SKIP PREP] All models excluded for '{prep}'")
            continue

        print(f"\n{'=' * 70}\n[PREP] {prep.upper()}\n{'=' * 70}")
        try:
            prep_data = prepare_dataset_for_prep(config, prep)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR PREP] Failed to prepare data for '{prep}': {e}")
            results += [ExperimentResult.failed(config, m, prep, f"Prep failed: {e}") for m in models]
            continue
        print(f"  Train: {prep_data['X_tr'].shape}, Test: {prep_data['X_te'].shape}")

        if parallel and len(models) > 1:
            with ProcessPoolExecutor(max_workers=min(max_workers, len(models))) as pool:
                futures = {
                    pool.submit(train_and_evaluate, config, prep_data, m,
                                skip_visualization=skip_visualization, verbose=0): m
                    for m in models
                }
                for future in as_completed(futures):
                    m = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:  # noqa: BLE001 - e.g. worker crash
                        results.append(ExperimentResult.failed(config, m, prep, str(e)))
        else:
            for m in models:
                print(f"\n  [MODEL] {m.upper()} on {prep.upper()}")
                results.append(train_and_evaluate(
                    config, prep_data, m, skip_visualization=skip_visualization, verbose=0
                ))

    total_time = time.perf_counter() - start
    df = _summary_frame(results)
    summary_path = output_dir / "grid_search_summary.csv"
    df.to_csv(summary_path, index=False)

    print(f"\n{'#' * 70}\n# GRID SEARCH COMPLETE in {total_time:.1f}s\n# Summary: {summary_path}\n{'#' * 70}")
    return GridSearchResult(
        results=results,
        summary_df=df,
        best_by_metric=_best_by_metric(results),
        total_time=total_time,
        config=grid_config,
    )


def run_grid_search_cli(
    config_path: Optional[str] = None,
    grid_config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    skip_viz: bool = False,
    parallel: bool = False,
    workers: int = 2,
) -> GridSearchResult:
    """Load configs, run the grid search and write a JSON summary of the best results."""
    config = load_config(config_path)
    if output_dir:
        config.output_dir = Path(output_dir)
    grid_config = load_grid_config(grid_config_path)

    result = run_grid_search_optimized(
        config, grid_config, skip_visualization=skip_viz, parallel=parallel, max_workers=workers
    )

    out = save_grid_json(result, config.output_dir)
    print(f"\n[OK] Grid search complete. Results in {out.parent}")
    return result


def save_grid_json(result: GridSearchResult, output_dir: Path) -> Path:
    """Write grid_search_results.json: totals, settings and the best run per metric."""
    import json
    out = Path(output_dir) / "grid_search_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "total_time": result.total_time,
            "config": result.config.model_dump(mode="json"),
            "best_by_metric": {
                k: {"prep": v.prep_method, "model": v.model_type, "metrics": v.metrics}
                for k, v in result.best_by_metric.items()
            },
        }, f, indent=2, default=str)
    return out
