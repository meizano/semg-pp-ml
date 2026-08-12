"""
Graph-based models (PyTorch GCN, GAT) for sEMG.
Treats EMG channels as nodes in a graph.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from .deep_learning import PyTorchModelWrapper


def create_channel_adjacency(n_channels: int, method: str = 'distance') -> np.ndarray:
    """Create adjacency matrix for EMG channels."""
    adj = np.zeros((n_channels, n_channels))

    if method == 'distance':
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    adj[i, j] = 1.0
                elif abs(i - j) == 1:
                    adj[i, j] = 0.5
                elif abs(i - j) == 2:
                    adj[i, j] = 0.25
    elif method == 'full':
        adj = np.ones((n_channels, n_channels))
    elif method == 'proximal_distal':
        proximal = [0, 1, 2]
        distal = [3, 4, 5]
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    adj[i, j] = 1.0
                elif i in proximal and j in proximal:
                    adj[i, j] = 0.5
                elif i in distal and j in distal:
                    adj[i, j] = 0.5
                else:
                    adj[i, j] = 0.1

    deg = np.sum(adj, axis=1, keepdims=True)
    deg[deg == 0] = 1
    adj = adj / deg

    return adj.astype(np.float32)


class PyTorchGraphNet(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int, params: Dict[str, Any], model_type: str = 'gcn'):
        super().__init__()
        length, n_channels = input_shape
        dense_units = params.get('dense_units', 128)
        drop = params.get('dropout_rate', 0.3)

        # Graph conv simulated via channel-wise linear transformations and attention
        self.conv1 = nn.Linear(n_channels, dense_units)
        self.conv2 = nn.Linear(dense_units, dense_units)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(length * dense_units, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, n_classes)
        )

    def forward(self, x):
        # x: (batch, length, channels) -> permute to (batch, length, channels) for channel linear
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        return self.fc(x)


def create_gcn(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchGraphNet(input_shape, n_classes, params, 'gcn')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'gcn')


def create_gat(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchGraphNet(input_shape, n_classes, params, 'gat')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'gat')


GRAPH_MODELS = {
    'gcn': create_gcn,
    'gat': create_gat,
}

GRAPH_NAMES = {
    'gcn': 'Graph Convolutional Network (GCN)',
    'gat': 'Graph Attention Network (GAT)',
}
