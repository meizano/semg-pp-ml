# Models

All model names below are accepted by `--model` and `model.type`.

## Which models need PyTorch?

| Group | Models | Extra needed |
|---|---|---|
| Classical | `svm`, `rf`, `knn`, `lda`, `nb`, `mlp_sklearn` | none (core install) |
| Deep learning | `mlp`, `cnn1d`, `cnn2d`, `rnn`, `lstm`, `bilstm`, `gru`, `gru_lstm`, `cnn_rnn`, `cnn_lstm`, `tcn` | `dl` |
| Transformer | `transformer`, `cnn_transformer`, `mafn`, `msemg` | `dl` |
| Graph | `gcn`, `gat` | `dl` |
| Embedded (TinyML) | `tiny_mlp`, `tiny_cnn` | `dl` |

Classical models run without PyTorch. Install the `dl` extra for everything else:

```bash
pip install -e ".[dl]"
```

## How inputs differ

- **Classical models** receive each window flattened to one vector of `window_size x n_channels` numbers.
- **Deep and graph models** receive the window as `(window_size, n_channels)`. Graph models treat channels as nodes.

## Choosing a model

| If you want | Try first |
|---|---|
| A fast baseline | `svm` or `rf` |
| A small model for a microcontroller | `tiny_cnn` (then export with the TinyML page) |
| Good accuracy on raw windows | `cnn1d` |
| Long temporal patterns | `lstm`, `bilstm`, `tcn` |
| The strongest published architectures | `cnn_transformer`, `mafn` |

Run a grid search to compare on your data: [Grid search](grid-search.md).

## Hyperparameters

Set them in the config under `model`, or with `--set`:

```bash
semg train --model svm --set model.svm.C=10 --set model.svm.gamma=0.01
semg train --model cnn1d --epochs 30 --set model.cnn_filters_1=32
```

The full list with defaults is in [Configuration](configuration.md#model).

## Metrics and saved models

Every model is evaluated with the same metrics (accuracy, precision, recall, F1, AUC, per-class scores, latency). Classical models are saved as `.pkl` and deep models as `.pt`. See [Results and outputs](outputs.md).

## Exporting to microcontrollers

The `tiny_*` models and the TinyML Export page can convert a trained model to ONNX and generate an Arduino sketch. Use the [web app](web-app.md) TinyML page for this. Check the exported model's accuracy again after conversion.
