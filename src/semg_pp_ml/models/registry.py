"""
Model Registry - Unified interface for all model types.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Callable, List, Optional
from pathlib import Path
import numpy as np



from .classical import CLASSICAL_MODELS, CLASSICAL_MODEL_NAMES, ClassicalModelWrapper
from .deep_learning import DEEP_LEARNING_MODELS, DEEP_LEARNING_NAMES
from .transformer import TRANSFORMER_MODELS, TRANSFORMER_NAMES
from .graph import GRAPH_MODELS, GRAPH_NAMES
from .embedded import EMBEDDED_MODELS, EMBEDDED_NAMES


# Combine all model registries
MODEL_REGISTRY: Dict[str, Callable] = {}
MODEL_REGISTRY.update(CLASSICAL_MODELS)
MODEL_REGISTRY.update(DEEP_LEARNING_MODELS)
MODEL_REGISTRY.update(TRANSFORMER_MODELS)
MODEL_REGISTRY.update(GRAPH_MODELS)
MODEL_REGISTRY.update(EMBEDDED_MODELS)

MODEL_NAMES: Dict[str, str] = {}
MODEL_NAMES.update(CLASSICAL_MODEL_NAMES)
MODEL_NAMES.update(DEEP_LEARNING_NAMES)
MODEL_NAMES.update(TRANSFORMER_NAMES)
MODEL_NAMES.update(GRAPH_NAMES)
MODEL_NAMES.update(EMBEDDED_NAMES)

# Model categories for UI grouping
MODEL_CATEGORIES = {
    'Classical ML': ['svm', 'rf', 'knn', 'lda', 'nb', 'mlp_sklearn'],
    'Deep Learning': ['mlp', 'cnn1d', 'cnn2d', 'rnn', 'lstm', 'bilstm', 'gru', 'gru_lstm', 'cnn_rnn', 'cnn_lstm', 'tcn'],
    'Transformer': ['transformer', 'cnn_transformer', 'mafn', 'msemg'],
    'Graph': ['gcn', 'gat'],
    'Embedded': ['tiny_mlp', 'tiny_cnn'],
}

# Models that need 2D input (flattened)
FLATTEN_MODELS = {'svm', 'rf', 'knn', 'lda', 'nb', 'mlp_sklearn'}

# Models that support probability prediction
PROBABILISTIC_MODELS = {
    'svm', 'rf', 'knn', 'lda', 'nb', 'mlp_sklearn',
    'mlp', 'cnn1d', 'cnn2d', 'rnn', 'lstm', 'bilstm', 'gru', 'gru_lstm',
    'cnn_rnn', 'cnn_lstm', 'tcn', 'transformer', 'cnn_transformer', 'mafn', 'msemg', 'gcn', 'gat',
    'tiny_mlp', 'tiny_cnn'
}


def get_model_categories() -> Dict[str, List[str]]:
    """Get model categories for UI."""
    return MODEL_CATEGORIES


def get_model_names() -> Dict[str, str]:
    """Get display names for all models."""
    return MODEL_NAMES


def get_available_models() -> List[str]:
    """Get list of all available model keys."""
    return list(MODEL_REGISTRY.keys())


def is_classical_model(model_type: str) -> bool:
    """Check if model is classical ML (needs flattened input)."""
    return model_type in FLATTEN_MODELS


def is_deep_learning_model(model_type: str) -> bool:
    """Check if model is deep learning (Keras)."""
    return model_type in DEEP_LEARNING_MODELS or model_type in TRANSFORMER_MODELS or model_type in GRAPH_MODELS or model_type in EMBEDDED_MODELS


def supports_probability(model_type: str) -> bool:
    """Check if model supports predict_proba."""
    return model_type in PROBABILISTIC_MODELS


def build_model(
    model_type: str,
    input_shape: Tuple,
    n_classes: int,
    lr: float,
    params: Dict[str, Any]
) -> Any:
    """Build model from registry."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    builder = MODEL_REGISTRY[model_type]
    if is_classical_model(model_type):
        return builder(params)
    else:
        return builder(input_shape, n_classes, lr, params)


def count_params(model: Any) -> Tuple[int, int, int]:
    """Count total, trainable, and non-trainable parameters."""
    if hasattr(model, 'model'):  # ClassicalModelWrapper
        return 0, 0, 0
    elif hasattr(model, 'count_params'):  # PyTorchModelWrapper
        return model.count_params()
    return 0, 0, 0


def save_model(model: Any, path: str | Path, model_type: str) -> None:
    """Save model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_classical_model(model_type):
        import joblib
        joblib.dump(model, path.with_suffix('.pkl'))
    else:
        if hasattr(model, 'save'):
            model.save(path.with_suffix('.pt'))
        else:
            import torch
            torch.save(model, path.with_suffix('.pt'))


def load_model(path: str | Path, model_type: str) -> Any:
    """Load model from disk."""
    path = Path(path)

    if is_classical_model(model_type):
        import joblib
        return joblib.load(path)
    else:
        from .deep_learning import PyTorchModelWrapper
        pt_path = path.with_suffix('.pt')
        if not pt_path.exists():
            pt_path = path  # fallback
        return PyTorchModelWrapper.load(pt_path, MODEL_REGISTRY[model_type])


def get_model_info(model_type: str) -> Dict[str, Any]:
    """Get metadata about a model type."""
    return {
        'type': model_type,
        'name': MODEL_NAMES.get(model_type, model_type),
        'category': next((cat for cat, models in MODEL_CATEGORIES.items() if model_type in models), 'Unknown'),
        'is_classical': is_classical_model(model_type),
        'is_dl': is_deep_learning_model(model_type),
        'supports_proba': supports_probability(model_type),
    }


# Default hyperparameters per model
DEFAULT_HYPERPARAMETERS = {
    # Classical
    'svm': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True},
    'rf': {'n_estimators': 100, 'max_depth': None, 'random_state': 42, 'n_jobs': -1},
    'knn': {'n_neighbors': 5, 'weights': 'uniform', 'n_jobs': -1},
    'lda': {'solver': 'svd'},
    'nb': {},
    'mlp_sklearn': {'hidden_layer_sizes': (256, 128), 'max_iter': 500, 'random_state': 42},

    # Deep Learning
    'mlp': {'mlp_units_1': 256, 'mlp_units_2': 128},
    'cnn1d': {'cnn_filters_1': 64, 'cnn_filters_2': 128, 'cnn_filters_3': 256, 'cnn_kernel': 3, 'pool_size': 2},
    'cnn2d': {'cnn_filters_1': 64, 'cnn_filters_2': 128, 'cnn_kernel': 3, 'pool_size': 2},
    'rnn': {'rnn_units_1': 128, 'rnn_units_2': 64},
    'lstm': {'rnn_units_1': 128, 'rnn_units_2': 64},
    'bilstm': {'rnn_units_1': 128, 'rnn_units_2': 64},
    'gru': {'rnn_units_1': 128, 'rnn_units_2': 64},
    'gru_lstm': {'rnn_units_1': 128, 'rnn_units_2': 64},
    'cnn_rnn': {'cnn_filters_1': 64, 'rnn_units_2': 64, 'cnn_kernel': 3, 'pool_size': 2},
    'cnn_lstm': {'cnn_filters_1': 64, 'cnn_filters_2': 128, 'rnn_units_2': 64, 'cnn_kernel': 3, 'pool_size': 2},
    'tcn': {'tcn_filters': 64, 'tcn_kernel_size': 3, 'tcn_dilations': [1, 2, 4, 8, 16]},

    # Transformer
    'transformer': {'transformer_heads': 4, 'transformer_ff_dim': 128, 'attention_key_dim': 64},
    'cnn_transformer': {'cnn_filters_1': 64, 'cnn_filters_2': 128, 'transformer_heads': 4, 'transformer_ff_dim': 128},
    'mafn': {'cnn_filters_1': 64, 'transformer_heads': 4},
    'msemg': {'cnn_filters_1': 64, 'rnn_units_2': 64},

    # Graph
    'gcn': {'dense_units': 128},
    'gat': {'dense_units': 128, 'transformer_heads': 4},

    # Embedded
    'tiny_mlp': {'tiny_mlp_units': [48, 24]},
    'tiny_cnn': {'tiny_cnn_filters': [16, 8], 'tiny_cnn_kernels': [5, 3]},
}


def get_default_hyperparameters(model_type: str) -> Dict[str, Any]:
    """Get default hyperparameters for a model."""
    return DEFAULT_HYPERPARAMETERS.get(model_type, {}).copy()


def merge_hyperparameters(model_type: str, user_params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user hyperparameters with defaults."""
    defaults = get_default_hyperparameters(model_type)
    merged = defaults.copy()
    merged.update(user_params)
    return merged