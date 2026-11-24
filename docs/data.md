# Data

The platform reads NinaPro DB8 data as MATLAB `.mat` files. This page describes the expected layout and file contents, and what happens when something is missing.

## Folder layout

By default the dataset root is `data/ninapro_db8` (relative to where you run the command). Put the training files in `train/` below it:

```text
data/ninapro_db8/
├── train/                  <- dataset.train_dir (default "train")
│   ├── S1_A1_E1.mat
│   ├── S1_A1_E2.mat
│   ├── S2_A1_E1.mat
│   └── ...
└── evaluation/             <- dataset.eval_dir (optional, default "evaluation")
    ├── S1_A1_E3.mat
    └── ...
```

Rules:

- **File names must start with `S<subject number>_`**, for example `S3_A1_E2.mat`. The loader selects files by subject with the pattern `S<n>_*.mat`.
- **`train_dir` and `eval_dir` are relative to `dataset.root`.** An absolute path also works.
- **Flat layout also works.** If `train/` does not exist but the root folder directly contains `S*.mat` files, the root is used. So `semg check-data --data-dir data/ninapro_db8` works for either layout.
- **Evaluation folder is optional.** If it is missing or has no `.mat` files, the platform splits the training data into train and test sets (stratified, 80/20).

Check the layout at any time:

```bash
semg check-data --data-dir path/to/ninapro_db8
```

## File contents

Each `.mat` file must contain these variables (MATLAB v5/v7 format):

| Variable | Shape | Used for | Required |
|---|---|---|---|
| `emg` | `(n_samples, n_electrodes)` | the sEMG signal, one column per electrode | yes (`EMG` is also accepted) |
| `stimulus` | `(n_samples,)` or `(n_samples, 1)` | the gesture label of each sample | yes, unless `restimulus` is present |
| `restimulus` | same as `stimulus` | a re-labelled version of `stimulus`, used if `stimulus` is missing | no |
| `glove` | `(n_samples, n_dof)` | CyberGlove joint angles (regression targets) | no |

Labels follow the NinaPro convention: `0` is rest, and positive integers are gestures. Only labels listed in `dataset.classes` are kept.

Files in MATLAB v7.3 (HDF5) format are **not** supported by the loader. They are skipped with the message `v7.3 HDF5 format not supported`. Re-save them from MATLAB with `save(file, '-v7')` or `-v6`, or convert them in Python.

## Channels, subjects and classes

These are set in the configuration (see [Configuration](configuration.md)):

| Key | Default | Meaning |
|---|---|---|
| `dataset.subjects` | `1..10` | which subjects to load |
| `dataset.channels` | `[3, 4, 5, 11, 12, 13]` | 0-based electrode columns of `emg` to keep |
| `dataset.n_channels` | `6` | must equal the number of channels you keep |
| `dataset.classes` | `[0, 1, 2, 3, 4, 5]` | labels to keep (windows with other labels are dropped) |

> If your recordings contain more gestures than the default `classes`, add them, and update `n_classes`. Otherwise those gestures are dropped without an error.

Keep `n_channels` equal to `len(channels)`. The model input shape is built from `n_channels`.

## Missing or wrong data

| Situation | What happens |
|---|---|
| Folder does not exist | Error: `Dataset folder not found: ...`. Fix `dataset.root` or `--data-dir`. |
| Folder exists but has no files for the chosen subjects | Error: `No usable S<subject>_*.mat files ...`. |
| An evaluation folder exists but has no files for the chosen subjects | Warning, then the platform falls back to the train/test split. |
| A file has no `emg` or `stimulus` key | That file is skipped with a `[SKIP]` message. |

The platform does not generate fake data when real data is missing. If you need artificial windows for testing, use [Augmentation and synthetic data](augmentation.md).
