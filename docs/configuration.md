# Configuration

Settings are read from a YAML file, can be overridden on the command line, and are validated when loaded. This page lists every setting and its default.

## Where settings come from

From lowest to highest priority:

1. Built-in defaults (defined in `src/semg_pp_ml/config/models.py`).
2. The YAML file: `config/default.yaml` if it exists, or the file you pass with `--config`.
3. Convenience flags such as `--prep`, `--model`, `--epochs`, `--data-dir`, `--output`.
4. `--set KEY=VALUE` overrides, applied last.

## Starter files

```bash
semg init-config --dir config
```

creates `config/default.yaml`, one file per preprocessing method in `config/preprocessing/`, and one per model in `config/models/`. Copy one of these, edit it, and pass it with `--config`.

`config/grid_search.yaml` (shipped with the repository) is used by `semg grid --grid-config`.

## Strict checking

- **Unknown keys are errors.** A typo such as `training.epohcs` stops the command with `unknown config key`. This applies to YAML files and `--set`.
- **Values are validated.** Wrong types and unknown method names are rejected with a message that names the key.
- **Paths are expanded.** `~` is expanded, and relative paths are resolved against the current folder.

## Overrides (`--set`)

```bash
semg train --set training.epochs=5 --set model.svm.C=3 --set preprocessing.method=vmd
```

- Format: `dotted.key=value`. Repeat `--set` for several values.
- The value is read as YAML: `1e-3` is a number, `[1, 2]` is a list, `true` is a boolean, `null` is empty.
- Keys inside free-form dictionaries can be set even if they are not listed here, for example `model.svm.C` or `tags.owner`.

## Reference

### `dataset`

| Key | Type | Default | Description |
|---|---|---|---|
| `root` | path | `data/ninapro_db8` | Dataset folder. Relative paths are resolved against the current folder. |
| `train_dir` | path | `train` | Training folder, relative to `root` (or absolute). |
| `eval_dir` | path | `evaluation` | Optional test folder, relative to `root`. |
| `subjects` | list of int | `1..10` | Subject numbers to load. |
| `channels` | list of int | `[3, 4, 5, 11, 12, 13]` | 0-based electrode columns to keep. |
| `n_channels` | int | `6` | Number of channels in `channels`. Sets the model input size. |
| `classes` | list of int | `[0, 1, 2, 3, 4, 5]` | Labels to keep. |
| `n_classes` | int | `6` | Number of classes. |

### `signal`

| Key | Default | Description |
|---|---|---|
| `fs` | `2000` | Sampling rate in Hz. |
| `window_size` | `400` | Window length in samples (200 ms at 2000 Hz). |
| `overlap` | `100` | Overlap between windows in samples. Step = `window_size - overlap`. |

### `preprocessing`

| Key | Default | Description |
|---|---|---|
| `method` | `butterworth_kalman` | One of `raw`, `butterworth`, `kalman`, `butterworth_kalman`, `vmd`, `wavelet`, `notch`. |
| `butterworth.order` | `4` | Filter order. |
| `butterworth.low_hz` / `high_hz` | `10.0` / `400.0` | Band-pass edges in Hz. |
| `kalman.q` / `kalman.r` | `1e-5` / `0.05` | Kalman process and measurement noise. |
| `vmd.alpha` | `2000` | VMD bandwidth constraint. |
| `vmd.k` | `3` | Number of modes. |
| `vmd.tau`, `dc`, `init`, `tol` | `0`, `0`, `1`, `1e-7` | Other VMD parameters. |
| `wavelet.wavelet` | `db4` | Wavelet family (any PyWavelets name). |
| `wavelet.level` | `4` | Decomposition level. |
| `wavelet.threshold_mode` | `soft` | `soft` or `hard`. |
| `notch.freq` | `50.0` | Mains frequency to remove (use `60.0` where needed). |
| `notch.quality_factor` | `30.0` | Notch sharpness. |

### `training`

| Key | Default | Description |
|---|---|---|
| `epochs` | `100` | Maximum epochs (deep models). |
| `batch_size` | `64` | Batch size (deep models). |
| `lr` | `0.001` | Learning rate. |
| `validation_split` | `0.2` | Fraction of training windows held out for validation (deep models). |
| `test_size` | `0.2` | Test fraction when there is no evaluation folder. |
| `random_state` | `42` | Seed for the train/test split. |
| `early_stopping`, `patience` | `true`, `15` | **Stored but not yet used by the training code.** See the known gaps below. |

### `model`

| Key | Default | Description |
|---|---|---|
| `type` | `cnn1d` | Model to train. See [Models](models.md). |
| `svm` | `{C: 1.0, kernel: rbf, gamma: scale, probability: true}` | SVM parameters (passed to scikit-learn). |
| `rf` | `{n_estimators: 100, max_depth: null, random_state: 42, n_jobs: -1}` | Random forest parameters. |
| `knn` | `{n_neighbors: 5, weights: uniform, n_jobs: -1}` | k-NN parameters. |
| `lda` | `{solver: svd}` | LDA parameters. |
| `nb` | `{}` | Naive Bayes parameters. |
| `mlp_sklearn` | `{hidden_layer_sizes: [256, 128], max_iter: 500, random_state: 42}` | scikit-learn MLP. |
| `activation`, `dropout_rate`, `dense_units` | `relu`, `0.3`, `128` | Shared deep-learning settings. |
| `cnn_filters_1..3`, `cnn_kernel`, `pool_size` | `64`, `128`, `256`, `3`, `2` | CNN architecture. |
| `rnn_units_1`, `rnn_units_2` | `128`, `64` | Recurrent layer sizes. |
| `tcn_filters`, `tcn_kernel_size`, `tcn_dilations` | `64`, `3`, `[1, 2, 4, 8, 16]` | Temporal convolution network. |
| `transformer_heads`, `transformer_ff_dim`, `attention_key_dim` | `4`, `128`, `64` | Transformer block. |
| `mlp_units_1`, `mlp_units_2` | `256`, `128` | Keras-style MLP sizes. |
| `tiny_mlp_units`, `tiny_cnn_filters`, `tiny_cnn_kernels` | `[48, 24]`, `[16, 8]`, `[5, 3]` | Small models for microcontrollers. |

### Top level

| Key | Default | Description |
|---|---|---|
| `output_dir` | `outputs` | Where results, models, plots and the cache are written. |
| `name`, `description`, `tags`, `experiment_name` | `default`, empty, empty, `semg_experiment` | Labels stored with results. |

## Grid search settings (`config/grid_search.yaml`)

```yaml
preps:                      # preprocessing methods to try
  - "raw"
  - "butterworth"
models:                     # models to train on each method
  - "svm"
  - "cnn1d"
exclude_pairs:              # (method, model) pairs to skip
  - ["raw", "svm"]
  - ["raw", "knn"]
```

Excluded pairs are skipped without running. A method whose models are all excluded is not preprocessed at all.

## Known gaps

- `training.early_stopping` and `training.patience` are read from the config but the training code does not use them yet. Deep models train for the full `epochs`.
- Settings that describe the NinaPro layout in the old README (for example `../ninapro/DB8`) are no longer used. The default root is `data/ninapro_db8`.
