"""
Embedded/TinyML models (PyTorch) for edge deployment (e.g., nRF52840).
"""
from __future__ import annotations

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from .deep_learning import PyTorchModelWrapper


class PyTorchTinyMLP(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int, params: Dict[str, Any]):
        super().__init__()
        in_features = input_shape[0] * input_shape[1]
        units = params.get('tiny_mlp_units', [48, 24])

        layers = [nn.Flatten()]
        prev_u = in_features
        for i, u in enumerate(units):
            layers.append(nn.Linear(prev_u, u))
            layers.append(nn.ReLU())
            prev_u = u
        layers.append(nn.Linear(prev_u, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PyTorchTinyCNN(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int, params: Dict[str, Any]):
        super().__init__()
        n_channels = input_shape[1]
        filters = params.get('tiny_cnn_filters', [16, 8])
        kernels = params.get('tiny_cnn_kernels', [5, 3])

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, filters[0], kernel_size=kernels[0], padding=kernels[0]//2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], padding=kernels[1]//2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[1], 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_tiny_mlp(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchTinyMLP(input_shape, n_classes, params)
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'tiny_mlp')


def create_tiny_cnn(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchTinyCNN(input_shape, n_classes, params)
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'tiny_cnn')


EMBEDDED_MODELS = {
    'tiny_mlp': create_tiny_mlp,
    'tiny_cnn': create_tiny_cnn,
}

EMBEDDED_NAMES = {
    'tiny_mlp': 'Tiny MLP (nRF52840)',
    'tiny_cnn': 'Tiny CNN (nRF52840)',
}
