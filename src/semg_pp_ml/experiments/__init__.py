"""
Experiments package for running and tracking sEMG experiments.
"""
from .runner import (
    run_single_experiment,
    ExperimentResult,
    run_experiment_cli,
)
from .grid_search import (
    run_grid_search_optimized,
    GridSearchResult,
    run_grid_search_cli,
    prepare_dataset_for_prep,
)
from .tracking import (
    MLflowTracker,
    auto_log_training,
)

__all__ = [
    # Runner
    "run_single_experiment",
    "ExperimentResult",
    "run_experiment_cli",
    # Grid Search
    "run_grid_search_optimized",
    "GridSearchResult",
    "run_grid_search_cli",
    "prepare_dataset_for_prep",
    # Tracking
    "MLflowTracker",
    "auto_log_training",
]