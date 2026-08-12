"""
Transformer-based models (PyTorch) for sEMG classification.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from .deep_learning import PyTorchModelWrapper


class PyTorchTransformerEncoder(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int, params: Dict[str, Any], use_cnn: bool = False, model_name: str = 'transformer'):
        super().__init__()
        length, n_channels = input_shape
        key_dim = params.get('attention_key_dim', 64)
        heads = params.get('transformer_heads', 4)
        ff_dim = params.get('transformer_ff_dim', 128)
        drop = params.get('dropout_rate', 0.3)

        self.use_cnn = use_cnn
        if use_cnn:
            f1 = params.get('cnn_filters_1', 64)
            self.cnn = nn.Sequential(
                nn.Conv1d(n_channels, f1, kernel_size=3, padding=1),
                nn.BatchNorm1d(f1),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            proj_in = f1
        else:
            proj_in = n_channels

        self.input_proj = nn.Linear(proj_in, key_dim)
        self.ln1 = nn.LayerNorm(key_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=key_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=drop,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(key_dim, n_classes)
        )

    def forward(self, x):
        # x: (batch, length, channels)
        if self.use_cnn:
            x = x.permute(0, 2, 1)
            x = self.cnn(x)
            x = x.permute(0, 2, 1)

        x = self.input_proj(x)
        x = self.ln1(x)
        x = self.transformer_encoder(x)
        # Global average pooling over time
        x = x.mean(dim=1)
        return self.fc(x)


def create_transformer(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchTransformerEncoder(input_shape, n_classes, params, use_cnn=False, model_name='transformer')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'transformer')


def create_cnn_transformer(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchTransformerEncoder(input_shape, n_classes, params, use_cnn=True, model_name='cnn_transformer')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'cnn_transformer')


def create_mafn(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchTransformerEncoder(input_shape, n_classes, params, use_cnn=True, model_name='mafn')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'mafn')


def create_msemg(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchTransformerEncoder(input_shape, n_classes, params, use_cnn=True, model_name='msemg')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'msemg')


TRANSFORMER_MODELS = {
    'transformer': create_transformer,
    'cnn_transformer': create_cnn_transformer,
    'mafn': create_mafn,
    'msemg': create_msemg,
}

TRANSFORMER_NAMES = {
    'transformer': 'Pure Transformer Encoder',
    'cnn_transformer': 'CNN-Transformer Hybrid',
    'mafn': 'Multi-Scale Attention Fusion Network (MAFN)',
    'msemg': 'mEMG Transformer',
}
