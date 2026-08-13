"""
Deep Learning models (PyTorch) for sEMG classification.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


class PyTorchModelWrapper:
    """Wrapper to make PyTorch models compatible with sklearn/Keras training/prediction interface."""
    def __init__(self, net: nn.Module, input_shape: Tuple[int, int], n_classes: int, lr: float, model_type: str = "dl"):
        self.net = net
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.lr = lr
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, epochs: int = 50, batch_size: int = 32, callbacks: List = None, verbose: int = 1):
        if y.ndim > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y.astype(np.int64)

        val_size = int(len(X) * validation_split)
        if val_size > 0:
            X_tr, X_val = X[:-val_size], X[-val_size:]
            y_tr, y_val = y_labels[:-val_size], y_labels[-val_size:]
        else:
            X_tr, X_val = X, X[:batch_size]
            y_tr, y_val = y_labels, y_labels[:batch_size]

        is_conv = any(c in self.model_type for c in ['cnn', 'tcn', 'gcn', 'gat'])
        if is_conv and X_tr.ndim == 3:
            X_tr_t = torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1)
            X_val_t = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
        else:
            X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)

        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_weights = None

        epoch_times = []
        throughputs = []

        import time
        for epoch in range(epochs):
            t0 = time.perf_counter()
            self.net.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

            train_loss = running_loss / total
            train_acc = correct / total

            self.net.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.net(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()

            val_loss = val_loss / val_total if val_total > 0 else train_loss
            val_acc = val_correct / val_total if val_total > 0 else train_acc

            elapsed = time.perf_counter() - t0
            epoch_times.append(elapsed)
            throughput = len(X_tr) / (elapsed + 1e-10)
            throughputs.append(throughput)

            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)

            if verbose > 0 and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose > 0:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if best_weights is not None:
            self.net.load_state_dict(best_weights)

        class HistoryObj:
            pass
        hist = HistoryObj()
        hist.history = self.history
        return hist

    def predict_proba(self, X: np.ndarray, batch_size: int = 32, verbose: int = 0) -> np.ndarray:
        self.net.eval()
        is_conv = any(c in self.model_type for c in ['cnn', 'tcn', 'gcn', 'gat'])
        if is_conv and X.ndim == 3:
            X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        else:
            X_t = torch.tensor(X, dtype=torch.float32)

        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        probs_list = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.net(batch_X)
                if outputs.ndim == 1:
                    outputs = outputs.unsqueeze(0)
                probs = softmax(outputs)
                if probs.ndim == 1:
                    probs = probs.unsqueeze(0)
                probs_list.append(probs.cpu().numpy())

        res = np.vstack(probs_list)
        if res.ndim == 1:
            res = res.reshape(1, -1)
        return res

    def predict(self, X: np.ndarray, batch_size: int = 32, verbose: int = 0) -> np.ndarray:
        probs = self.predict_proba(X, batch_size, verbose)
        return np.argmax(probs, axis=1)

    def count_params(self) -> Tuple[int, int, int]:
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return total, trainable, total - trainable

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.net.state_dict(),
            'input_shape': self.input_shape,
            'n_classes': self.n_classes,
            'lr': self.lr,
            'model_type': self.model_type,
        }, path)

    @classmethod
    def load(cls, path: str | Path, net_builder_func) -> PyTorchModelWrapper:
        data = torch.load(path, map_location='cpu')
        net = net_builder_func(data['input_shape'], data['n_classes'], data['lr'], {})
        net.load_state_dict(data['state_dict'])
        wrapper = cls(net, data['input_shape'], data['n_classes'], data['lr'], data['model_type'])
        return wrapper


# =============================================================================
# PyTorch Net Architectures
# =============================================================================
class PyTorchMLP(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int, params: Dict[str, Any]):
        super().__init__()
        in_features = input_shape[0] * input_shape[1]
        u1 = params.get('mlp_units_1', 256)
        u2 = params.get('mlp_units_2', 128)
        drop = params.get('dropout_rate', 0.3)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, u1),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(u1, u2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(u2, n_classes)
        )
    def forward(self, x):
        return self.net(x)


class PyTorchCNN1D(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int, params: Dict[str, Any]):
        super().__init__()
        n_channels = input_shape[1]
        f1 = params.get('cnn_filters_1', 64)
        f2 = params.get('cnn_filters_2', 128)
        f3 = params.get('cnn_filters_3', 256)
        kernel = params.get('cnn_kernel', 3)
        pool = params.get('pool_size', 2)
        dense_u = params.get('dense_units', 128)
        drop = params.get('dropout_rate', 0.3)

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, f1, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm1d(f1),
            nn.ReLU(),
            nn.MaxPool1d(pool),
            nn.Conv1d(f1, f2, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm1d(f2),
            nn.ReLU(),
            nn.MaxPool1d(pool),
            nn.Conv1d(f2, f3, kernel_size=kernel, padding=kernel//2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f3, dense_u),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dense_u, n_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class PyTorchCNN2D(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int, params: Dict[str, Any]):
        super().__init__()
        length, n_channels = input_shape
        f1 = params.get('cnn_filters_1', 64)
        f2 = params.get('cnn_filters_2', 128)
        kernel = params.get('cnn_kernel', 3)
        pool = params.get('pool_size', 2)
        dense_u = params.get('dense_units', 128)
        drop = params.get('dropout_rate', 0.3)

        self.conv = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(f1, f2, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(pool),
        )
        # Calculate flattened dim
        dummy = torch.zeros(1, 1, length, n_channels)
        dummy_out = self.conv(dummy)
        flat_dim = dummy_out.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, dense_u),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dense_u, n_classes)
        )
    def forward(self, x):
        # x is (batch, length, channels) -> (batch, 1, length, channels)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        return self.classifier(x)


class PyTorchRNN(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int, params: Dict[str, Any], rnn_type: str = 'rnn'):
        super().__init__()
        length, n_channels = input_shape
        u1 = params.get('rnn_units_1', 128)
        u2 = params.get('rnn_units_2', 64)
        dense_u = params.get('dense_units', 128)
        drop = params.get('dropout_rate', 0.3)
        is_bi = 'bi' in rnn_type

        rnn_cls = nn.RNN if 'rnn' in rnn_type and 'lstm' not in rnn_type and 'gru' not in rnn_type else (nn.LSTM if 'lstm' in rnn_type else nn.GRU)

        self.rnn1 = rnn_cls(n_channels, u1, batch_first=True, bidirectional=is_bi)
        mult = 2 if is_bi else 1
        self.dropout = nn.Dropout(drop)
        self.rnn2 = rnn_cls(u1 * mult, u2, batch_first=True, bidirectional=is_bi)
        self.fc = nn.Sequential(
            nn.Linear(u2 * mult, dense_u),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dense_u, n_classes)
        )

    def forward(self, x):
        # x is (batch, length, channels)
        out, _ = self.rnn1(x)
        out = self.dropout(out)
        out, _ = self.rnn2(out)
        # take last time step
        out = out[:, -1, :]
        return self.fc(out)


class PyTorchHybrid(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int, params: Dict[str, Any], mode: str = 'cnn_lstm'):
        super().__init__()
        length, n_channels = input_shape
        f1 = params.get('cnn_filters_1', 64)
        f2 = params.get('cnn_filters_2', 128)
        kernel = params.get('cnn_kernel', 3)
        pool = params.get('pool_size', 2)
        u2 = params.get('rnn_units_2', 64)
        dense_u = params.get('dense_units', 128)
        drop = params.get('dropout_rate', 0.3)

        self.mode = mode
        self.conv1 = nn.Conv1d(n_channels, f1, kernel_size=kernel, padding=kernel//2)
        self.bn1 = nn.BatchNorm1d(f1)
        self.pool = nn.MaxPool1d(pool)

        if 'cnn_lstm' in mode or 'gru_lstm' in mode:
            self.conv2 = nn.Conv1d(f1, f2, kernel_size=kernel, padding=kernel//2)
            self.bn2 = nn.BatchNorm1d(f2)
            rnn_in_dim = f2
        else:
            rnn_in_dim = f1

        self.rnn = nn.LSTM(rnn_in_dim, u2, batch_first=True)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Sequential(
            nn.Linear(u2, dense_u),
            nn.ReLU(),
            nn.Linear(dense_u, n_classes)
        )

    def forward(self, x):
        # x: (batch, length, channels) -> permute to (batch, channels, length) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        if hasattr(self, 'conv2'):
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # permute back for RNN: (batch, length, channels)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class PyTorchTCN(nn.Module):
    def __init__(self, input_shape: Tuple, n_classes: int, params: Dict[str, Any]):
        super().__init__()
        n_channels = input_shape[1]
        filters = params.get('tcn_filters', 64)
        kernel = params.get('tcn_kernel_size', 3)
        dilations = params.get('tcn_dilations', [1, 2, 4, 8, 16])
        drop = params.get('dropout_rate', 0.3)

        layers = []
        in_c = n_channels
        for d in dilations:
            layers.append(nn.Conv1d(in_c, filters, kernel_size=kernel, padding=d*(kernel-1)//2, dilation=d))
            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())
            in_c = filters

        self.convs = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(filters, n_classes)
        )

    def forward(self, x):
        # (batch, length, channels) -> (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.convs(x)
        x = self.gap(x)
        return self.fc(x)


# =============================================================================
# Builder Functions returning PyTorchModelWrapper
# =============================================================================
def create_mlp(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchMLP(input_shape, n_classes, params)
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'mlp')

def create_cnn1d(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchCNN1D(input_shape, n_classes, params)
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'cnn1d')

def create_cnn2d(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchCNN2D(input_shape, n_classes, params)
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'cnn2d')

def create_rnn(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchRNN(input_shape, n_classes, params, 'rnn')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'rnn')

def create_lstm(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchRNN(input_shape, n_classes, params, 'lstm')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'lstm')

def create_bilstm(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchRNN(input_shape, n_classes, params, 'bilstm')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'bilstm')

def create_gru(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchRNN(input_shape, n_classes, params, 'gru')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'gru')

def create_gru_lstm(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchHybrid(input_shape, n_classes, params, 'gru_lstm')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'gru_lstm')

def create_cnn_rnn(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchHybrid(input_shape, n_classes, params, 'cnn_rnn')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'cnn_rnn')

def create_cnn_lstm(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchHybrid(input_shape, n_classes, params, 'cnn_lstm')
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'cnn_lstm')

def create_tcn(input_shape: Tuple, n_classes: int, lr: float, params: Dict[str, Any]) -> PyTorchModelWrapper:
    net = PyTorchTCN(input_shape, n_classes, params)
    return PyTorchModelWrapper(net, input_shape, n_classes, lr, 'tcn')


DEEP_LEARNING_MODELS = {
    'mlp': create_mlp,
    'cnn1d': create_cnn1d,
    'cnn2d': create_cnn2d,
    'rnn': create_rnn,
    'lstm': create_lstm,
    'bilstm': create_bilstm,
    'gru': create_gru,
    'gru_lstm': create_gru_lstm,
    'cnn_rnn': create_cnn_rnn,
    'cnn_lstm': create_cnn_lstm,
    'tcn': create_tcn,
}

DEEP_LEARNING_NAMES = {
    'mlp': 'Multi-Layer Perceptron',
    'cnn1d': 'CNN 1D',
    'cnn2d': 'CNN 2D (Pseudo-2D)',
    'rnn': 'Simple RNN',
    'lstm': 'LSTM',
    'bilstm': 'Bidirectional LSTM',
    'gru': 'GRU',
    'gru_lstm': 'GRU-LSTM Hybrid',
    'cnn_rnn': 'CNN-RNN Hybrid',
    'cnn_lstm': 'CNN-LSTM Hybrid',
    'tcn': 'Temporal Convolutional Network',
}
