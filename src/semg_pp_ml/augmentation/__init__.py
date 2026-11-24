"""Data augmentation and artificial sEMG generation (numpy/scipy only)."""

from .synthetic import SyntheticConfig, ClassModel, generate_synthetic_semg, make_class_models
from .transforms import (
    TRANSFORMS,
    add_gaussian_noise,
    augment_dataset,
    baseline_wander,
    channel_dropout,
    crosstalk,
    electrode_shift,
    mixup_same_class,
    powerline,
    random_amplitude_scale,
    time_warp,
)

__all__ = [
    "SyntheticConfig",
    "ClassModel",
    "generate_synthetic_semg",
    "make_class_models",
    "TRANSFORMS",
    "add_gaussian_noise",
    "augment_dataset",
    "baseline_wander",
    "channel_dropout",
    "crosstalk",
    "electrode_shift",
    "mixup_same_class",
    "powerline",
    "random_amplitude_scale",
    "time_warp",
]
