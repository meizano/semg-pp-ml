# CLI reference

The command is `semg`. It is installed with the project (`pip install -e .`). If your shell cannot find it, run `python -m semg_pp_ml.cli` instead. Every example below works with either form.

```text
semg [--version] COMMAND [options]
```

## Commands at a glance

| Command | Purpose |
|---|---|
| `semg info` | Show CPU, RAM and GPU information |
| `semg check-data` | Check that the dataset folder contains usable `.mat` files |
| `semg init-config` | Write starter YAML config files |
| `semg train` | Run one preprocessing + model experiment |
| `semg grid` | Run many preprocessing x model experiments |
| `semg augment` | Generate synthetic sEMG windows, or augment existing ones |

Get help for any command with `semg <command> --help`.

## Shared options

These work on `check-data`, `train` and `grid`:

| Option | Meaning |
|---|---|
| `--config FILE` | YAML config to load. Default: `config/default.yaml` if it exists, else built-in defaults. |
| `--data-dir DIR` | Dataset root (sets `dataset.root`). |
| `--output DIR` | Output folder (sets `output_dir`). Default: `outputs`. |
| `--set KEY=VALUE` | Override any config value. Repeat the flag for several values. See [Configuration](configuration.md#overrides---set). |

## `semg info`

```bash
semg info
```

Prints CPU model and cores, total RAM, and detected GPUs. Useful to confirm that PyTorch can see your GPU.

## `semg check-data`

```bash
semg check-data --data-dir data/ninapro_db8
```

Reports the resolved training folder, whether it exists, which subjects were found, how many `.mat` files there are, and whether the EMG key is present. Exit code `0` if files were found, `1` if not.

## `semg init-config`

```bash
semg init-config --dir config
```

Writes starter files: `config/default.yaml`, `config/preprocessing/*.yaml` (one per method) and `config/models/*.yaml`. Existing files are overwritten, so run it in an empty folder or a new one if you have edits.

## `semg train`

Runs one experiment: preprocessing, training, evaluation, saving.

```bash
semg train --prep butterworth --model svm
semg train --prep vmd --model cnn1d --epochs 20 --batch-size 32
semg train --prep raw --model rf --set model.rf.n_estimators=300 --no-viz
```

| Option | Meaning |
|---|---|
| `--prep METHOD` | Preprocessing method. Sets `preprocessing.method`. |
| `--model NAME` | Model type. Sets `model.type`. See [Models](models.md). |
| `--epochs N` | Training epochs (deep models). Sets `training.epochs`. |
| `--batch-size N` | Batch size (deep models). Sets `training.batch_size`. |
| `--no-save` | Do not save the trained model file. |
| `--no-viz` | Skip plots. Faster. |

Without `--prep` or `--model`, the values from the config are used (defaults: `butterworth_kalman` and `cnn1d`).

Prints the metrics and writes `result_<model>_<prep>.json` to the output folder. See [Results and outputs](outputs.md).

## `semg grid`

Runs every combination of preprocessing methods and models, except the excluded pairs.

```bash
semg grid --grid-config config/grid_search.yaml
semg grid --preps raw butterworth --models svm rf knn --no-viz
semg grid --preps vmd --models cnn1d lstm --parallel --workers 2
```

| Option | Meaning |
|---|---|
| `--grid-config FILE` | YAML with `preps`, `models` and `exclude_pairs`. Without it, built-in defaults are used. |
| `--preps M [M ...]` | Override the preprocessing methods. |
| `--models M [M ...]` | Override the models. |
| `--parallel` | Train models in parallel processes (classical models benefit most). |
| `--workers N` | Number of parallel workers. Default 2. |
| `--no-viz` | Skip plots. |

See [Grid search](grid-search.md) for details.

## `semg augment`

Generates or augments windows. All options are passed through to the augmentation tool.

```bash
# 200 synthetic windows per class for 6 classes
semg augment --synthetic --n-classes 6 --per-class 200 --out artificial.npz

# two augmented copies of an existing dataset
semg augment --input windows.npz --copies 2 --out augmented.npz
```

Full option list: [Augmentation and synthetic data](augmentation.md).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | An experiment failed (`train`), no data was found (`check-data`), or every grid run failed |
| `2` | Usage or configuration error (unknown option, unknown config key, missing config file, bad override) |

## Legacy `run_cli.py`

The old script still works and maps to the new commands:

```bash
python run_cli.py --mode info
python run_cli.py --mode single --prep wavelet --model cnn1d --epochs 20   # -> semg train
python run_cli.py --mode grid                                              # -> semg grid
```

New scripts should use `semg` directly.
