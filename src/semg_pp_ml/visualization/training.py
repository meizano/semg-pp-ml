"""
Training visualization for deep learning models.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_training_curves(
    history: Any,
    output_dir: str | Path,
    model_name: str,
    prep_name: str,
    title: Optional[str] = None
) -> str:
    """Plot training and validation accuracy/loss curves."""
    if not HAS_MATPLOTLIB:
        return ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_dict = history.history if hasattr(history, 'history') else history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(history_dict.get('accuracy', []), label='Train', linewidth=2)
    ax.plot(history_dict.get('val_accuracy', []), label='Validation', linewidth=2)
    ax.set_title('Model Accuracy', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.plot(history_dict.get('loss', []), label='Train', linewidth=2)
    ax.plot(history_dict.get('val_loss', []), label='Validation', linewidth=2)
    ax.set_title('Model Loss (Cross-Entropy)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    fname = output_dir / f"training_curves_{model_name}_{prep_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return str(fname)


def analyze_overfitting(history: Any) -> str:
    """Analyze training history for overfitting/underfitting."""
    history_dict = history.history if hasattr(history, 'history') else history
    train_loss = history_dict.get('loss', [])
    val_loss = history_dict.get('val_loss', [])

    if len(train_loss) < 5:
        return "Insufficient epochs for analysis"

    train_trend = np.mean(train_loss[-5:]) - np.mean(train_loss[-10:-5]) if len(train_loss) >= 10 else train_loss[-1] - train_loss[0]
    val_trend = np.mean(val_loss[-5:]) - np.mean(val_loss[-10:-5]) if len(val_loss) >= 10 else val_loss[-1] - val_loss[0]

    if val_trend > 0.05 and train_trend < -0.01:
        return "OVERFITTING detected: Validation loss increasing while training loss decreasing"
    elif train_trend > -0.01 and val_trend > -0.01:
        return "UNDERFITTING suspected: Both losses stagnant or high"
    else:
        return "GOOD generalization: Validation loss follows training loss"