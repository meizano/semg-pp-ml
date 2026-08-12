"""
sEMG Preprocessing and Machine Learning Platform for NinaPro DB8.
"""
__version__ = "0.1.0"

from .config import (
    ExperimentConfig,
    DatasetConfig,
    SignalConfig,
    PreprocessingConfig,
    TrainingConfig,
    ModelHyperparameters,
    GridSearchConfig,
    load_config,
    save_config,
    create_default_configs,
)
from .data import (
    load_mat_files_from_folder,
    sliding_window,
    prepare_dataset,
)
from .preprocessing import (
    apply_preprocessing,
    PreprocessingPipeline,
    PreprocessingResult,
    compute_snr,
    compute_psd_metrics,
    compute_preprocessing_rmse,
)
from .models import (
    build_model,
    get_available_models,
    get_model_categories,
    get_model_info,
    convert_to_onnx,
    generate_arduino_sketch,
    save_pytorch_model,
)
from .metrics import compute_all_classification_metrics
from .experiments import (
    run_single_experiment,
    run_grid_search_optimized,
    ExperimentResult,
    GridSearchResult,
)
from .utils import get_system_info, print_system_info

__all__ = [
    # Config
    "ExperimentConfig",
    "DatasetConfig",
    "SignalConfig",
    "PreprocessingConfig",
    "TrainingConfig",
    "ModelHyperparameters",
    "GridSearchConfig",
    "load_config",
    "save_config",
    "create_default_configs",
    # Data
    "load_mat_files_from_folder",
    "sliding_window",
    "prepare_dataset",
    # Preprocessing
    "apply_preprocessing",
    "PreprocessingPipeline",
    "PreprocessingResult",
    "compute_snr",
    "compute_psd_metrics",
    "compute_preprocessing_rmse",
    # Models
    "build_model",
    "get_available_models",
    "get_model_categories",
    "get_model_info",
    "convert_to_onnx",
    "generate_arduino_sketch",
    "save_pytorch_model",
    # Metrics
    "compute_all_classification_metrics",
    # Experiments
    "run_single_experiment",
    "run_grid_search_optimized",
    "ExperimentResult",
    "GridSearchResult",
    # Utils
    "get_system_info",
    "print_system_info",
]