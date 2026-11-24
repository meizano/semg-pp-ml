"""
sEMG Preprocessing and Machine Learning Platform for NinaPro DB8.

Public names are imported lazily on first access, so ``import semg_pp_ml`` and the
``semg`` CLI start quickly and do not load torch unless a model is used.
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.2.0"

_EXPORTS = {
    # config
    "ExperimentConfig": ".config", "DatasetConfig": ".config", "SignalConfig": ".config",
    "PreprocessingConfig": ".config", "TrainingConfig": ".config",
    "ModelHyperparameters": ".config", "GridSearchConfig": ".config",
    "load_config": ".config", "load_grid_config": ".config", "save_config": ".config",
    "create_default_configs": ".config", "apply_overrides": ".config",
    # data
    "load_mat_files_from_folder": ".data", "sliding_window": ".data", "prepare_dataset": ".data",
    # preprocessing
    "apply_preprocessing": ".preprocessing", "PreprocessingPipeline": ".preprocessing",
    "PreprocessingResult": ".preprocessing", "compute_snr": ".preprocessing",
    "compute_psd_metrics": ".preprocessing", "compute_preprocessing_rmse": ".preprocessing",
    # models
    "build_model": ".models", "get_available_models": ".models", "get_model_categories": ".models",
    "get_model_info": ".models", "convert_to_onnx": ".models",
    "generate_arduino_sketch": ".models", "save_pytorch_model": ".models",
    # metrics / experiments / utils
    "compute_all_classification_metrics": ".metrics",
    "run_single_experiment": ".experiments", "run_grid_search_optimized": ".experiments",
    "ExperimentResult": ".experiments", "GridSearchResult": ".experiments",
    "get_system_info": ".utils", "print_system_info": ".utils",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value  # cache for next access
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
