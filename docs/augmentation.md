# Augmentation and synthetic data

The `semg_pp_ml.augmentation` package does two things:

1. **Augments** existing windows: creates modified copies that keep their labels, to make training more robust.
2. **Generates** artificial sEMG windows from a simple physiological model, for testing or for adding training data.

It uses only numpy and scipy, so it runs without PyTorch.

> **Limit.** The synthetic generator is a simplified model inspired by published sEMG simulators. It is not validated against real recordings. Use it to prototype, test code, or add data cautiously. Check any model trained on it against real NinaPro data before drawing conclusions.

## Input format

All functions work on **windows**:

- `X`: array of shape `(n_windows, n_samples, n_channels)`, for example `(1000, 400, 6)`.
- `y`: integer labels of shape `(n_windows,)`.

To augment NinaPro data, first cut windows with `semg_pp_ml.data.loader.sliding_window`, then augment them.

## Augmentations

| Name | What it simulates | Main parameter |
|---|---|---|
| `noise` | sensor noise | `snr_db` (default 20) |
| `amplitude` | electrode and skin changes | `sigma` (log-normal gain) |
| `time_warp` | slightly faster or slower contractions | `sigma` (speed variation) |
| `electrode_shift` | electrodes moved around the muscle | `max_shift` (channels) |
| `crosstalk` | signal leaking into neighbouring electrodes | `max_alpha` |
| `channel_dropout` | a loose or failed electrode | `p` (probability) |
| `baseline_wander` | slow drift from movement | `amp` |
| `powerline` | mains hum at 50 or 60 Hz | `mains_hz`, `amp` |
| `mixup` (same class) | blending two windows of the same gesture | `alpha` |

By default `augment_dataset` applies `noise`, `amplitude`, `time_warp`, `electrode_shift` and `crosstalk`, then same-class mixup.

## Augment a dataset from the command line

Save your windows as an `.npz` file with arrays `X` and `y`, then:

```bash
semg augment --input windows.npz --copies 2 --out augmented.npz
```

This keeps the originals and adds two augmented copies of every window. Options:

| Option | Default | Meaning |
|---|---|---|
| `--input FILE` | | `.npz` with `X` and `y` |
| `--copies N` | `1` | augmented copies per window |
| `--ops NAME ...` | defaults above | choose the augmentations |
| `--no-mixup` | off | skip same-class mixup |
| `--fs HZ` | `2000` | sampling rate, used by baseline wander and powerline |
| `--seed N` | `0` | random seed |
| `--out FILE` | required | output `.npz` |

Example with specific operations:

```bash
semg augment --input windows.npz --copies 3 --ops noise time_warp powerline --out aug.npz
```

## Generate artificial data

```bash
semg augment --synthetic --n-classes 6 --per-class 200 --channels 8 --out artificial.npz
```

| Option | Default | Meaning |
|---|---|---|
| `--synthetic` | | generate instead of augmenting |
| `--n-classes N` | `6` | number of gesture classes |
| `--per-class N` | `200` | windows per class |
| `--channels N` | `8` | electrodes |
| `--fs HZ` | `2000` | sampling rate |

Each class gets its own activation level and spatial pattern, so classes differ in force and in which electrodes respond. Higher-force classes have larger signals; this is checked by a test.

## Use it in Python

```python
import numpy as np
from semg_pp_ml.augmentation import augment_dataset, generate_synthetic_semg, SyntheticConfig

# augment existing windows
X_aug, y_aug = augment_dataset(X, y, n_copies=2, fs=2000, rng=0)

# generate artificial windows
cfg = SyntheticConfig(fs=2000, window_ms=200, n_channels=8, seed=1)
X_syn, y_syn, models = generate_synthetic_semg(n_classes=6, windows_per_class=100, cfg=cfg)
```

Everything is reproducible: pass the same `seed` or `rng` and you get the same output.

## Tips

- Augment **only the training data**. Augmenting the test set makes results look better than they are.
- Start with mild settings (for example `noise` at 20 dB) and increase them if the model overfits.
- Compare a model trained with and without augmentation on the same test set before keeping it.
