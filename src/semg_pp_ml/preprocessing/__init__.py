"""
Preprocessing package for sEMG signals.
"""
from .filters import (
    apply_preprocessing,
    get_filter,
    FILTER_REGISTRY,
    raw_filter,
    butterworth_filter,
    kalman_filter,
    butterworth_kalman_filter,
    vmd_filter,
    wavelet_filter,
    notch_filter,
    KalmanFilter1D,
)
from .metrics import (
    compute_snr,
    compute_psd_metrics,
    compute_preprocessing_rmse,
    compute_all_preprocessing_metrics,
)
from .pipeline import (
    PreprocessingPipeline,
    PreprocessingResult,
    run_preprocessing_cli,
)

__all__ = [
    # Filters
    "apply_preprocessing",
    "get_filter",
    "FILTER_REGISTRY",
    "raw_filter",
    "butterworth_filter",
    "kalman_filter",
    "butterworth_kalman_filter",
    "vmd_filter",
    "wavelet_filter",
    "notch_filter",
    "KalmanFilter1D",
    # Metrics
    "compute_snr",
    "compute_psd_metrics",
    "compute_preprocessing_rmse",
    "compute_all_preprocessing_metrics",
    # Pipeline
    "PreprocessingPipeline",
    "PreprocessingResult",
    "run_preprocessing_cli",
]