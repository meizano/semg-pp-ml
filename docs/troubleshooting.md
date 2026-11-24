# Troubleshooting

Find your symptom below. Each entry gives the likely cause and the fix.

## Installation

### `semg` is not recognized

Cause: the folder that contains `semg.exe` is not on your PATH. This happens when you install with `pip install --user` on Windows, and pip prints a warning such as "The script semg.exe is installed in ... which is not on PATH".

Fix, in order of preference:

1. Use a virtual environment (see [Getting started](getting-started.md)). After activation, `semg` is on the PATH.
2. Or run the command as a module, which always works: `python -m semg_pp_ml.cli --version`.
3. Or add the printed `Scripts` folder to your PATH.

### `No module named torch`

Cause: you ran a deep-learning model without the `dl` extra.

Fix: `pip install -e ".[dl]"`. Classical models (`svm`, `rf`, `knn`, `lda`, `nb`, `mlp_sklearn`) do not need PyTorch.

### `No module named streamlit`

Fix: `pip install -e ".[ui]"`.

### PowerShell says running scripts is disabled

Cause: the execution policy blocks `Activate.ps1`.

Fix: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. See the [Windows guide](windows_guide.md).

## Data

### `Dataset folder not found: ...`

Cause: the path is wrong. Relative paths are resolved from the folder you run the command in.

Fix: run `semg check-data`, then either use `--data-dir` with the correct path, or set `dataset.root` in the config. An absolute path is the safest choice.

### `No .mat files found`

Cause: the folder exists but has no `S<n>_*.mat` files, or the subject numbers do not match the configured `dataset.subjects`.

Fix: check the file names (they must start with `S` and a number) and the `subjects` setting. See [Data](data.md).

### `v7.3 HDF5 format not supported`

Cause: the `.mat` file was saved in MATLAB's newer format, which the loader does not read.

Fix: re-save the file in MATLAB with `save('file.mat', 'emg', 'stimulus', '-v7')`, or convert it in Python.

### Training runs but accuracy is near chance

Possible causes:

- Labels are not what you expect. Check `dataset.classes`. Labels not in the list are dropped, so you may be training on fewer gestures than you think.
- Channels are wrong. Check `dataset.channels` (0-based) and `n_channels`.
- A window size that is too short for the signal. Try `signal.window_size` of 300 to 500 samples.

## Configuration

### `unknown config key 'training.epohcs'`

Cause: a typo, or a key that does not exist. The platform refuses unknown keys so that typos cannot silently change results.

Fix: correct the key. Keys are listed in [Configuration](configuration.md).

### `override must look like key.path=value`

Cause: a `--set` argument has no `=` sign.

Fix: use `--set training.epochs=5`.

### Config changes have no effect

Cause: a value is being overridden later in the chain. The order is: YAML file, then convenience flags (`--prep`, `--model`, ...), then `--set`. So `--set` always wins. Check the order in [Configuration](configuration.md#where-settings-come-from).

Fix: remove the conflicting `--set` or flag, or edit the file you pass with `--config`.

## Running experiments

### A model shows `FAILED` in the grid search

Cause: the model could not be built or trained with the current settings. The error message is in the `error` column of `grid_search_summary.csv`.

Fix: run that single combination with `semg train --prep <method> --model <model>` to see the full traceback. Other models in the grid still run.

### Out of memory

Causes and fixes:

- Fewer subjects or channels: `--set dataset.subjects=[1,2,3]`.
- A smaller batch size for deep models: `--batch-size 16`.
- No parallel runs: remove `--parallel` or use `--workers 1`.
- Many subjects at once: each preprocessing method keeps the whole filtered training set in memory. Start with fewer subjects.

### Preprocessing is very slow

`vmd` is the slowest method. Run it once; the result is cached in `outputs/preprocessing_cache`. Later runs with the same settings are fast.

### No plots are written

Cause: `--no-viz` was used, or the output folder is not writable.

### Deep model training is very slow

Check `semg info` to see whether a GPU is detected. Without one, reduce `training.epochs` and the model size, or use a smaller subset of subjects.

## Web app

### Pages show the wrong dataset folder

Each page has its own dataset-folder box in the sidebar. Set it there. The default is `data/ninapro_db8`, relative to the folder you started Streamlit from.

## REST API

### `401 Unauthorized`

Cause: the access token is missing, expired or wrong. Access tokens expire.

Fix: log in again, or use the refresh endpoint. Send the header as `Authorization: Bearer <token>`.

### Login works but other requests fail

Check that you send the token on every request, and that the server was started with the `api` extra installed.

## Still stuck?

- Run `semg info` and `semg check-data` and note the output.
- Run the failing command with `--no-viz` and without `--set` overrides to rule them out.
- Open an issue at [github.com/meizano/semg-pp-ml](https://github.com/meizano/semg-pp-ml/issues) with the command, the full error and the output of `semg info`.
