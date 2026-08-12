"""
Visualization package for sEMG signals, training, and results.
"""
from .signals import (
    plot_signal_comparison,
    plot_preprocessing_metrics,
    plot_psd_comparison,
)
from .training import (
    plot_training_curves,
    analyze_overfitting,
)
from .confusion import (
    plot_confusion_matrix,
)

__all__ = [
    # Signals
    "plot_signal_comparison",
    "plot_preprocessing_metrics",
    "plot_psd_comparison",
    # Training
    "plot_training_curves",
    "analyze_overfitting",
    # Confusion/Results
    "plot_confusion_matrix",
]