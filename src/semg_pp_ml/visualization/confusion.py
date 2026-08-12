"""
Confusion matrix visualization.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from sklearn.metrics import confusion_matrix

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str | Path,
    model_name: str,
    prep_name: str,
    title: Optional[str] = None,
    normalize: bool = False,
    cmap: str = 'Blues'
) -> str:
    """Plot confusion matrix."""
    if not HAS_MATPLOTLIB:
        return ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title or 'Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    suffix = "_norm" if normalize else ""
    fname = output_dir / f"confusion_matrix_{model_name}_{prep_name}{suffix}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return str(fname)