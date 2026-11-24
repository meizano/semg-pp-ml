# How it works

This page explains what the platform does to your data, step by step. Knowing this helps you choose settings and read the results correctly.

## The pipeline at a glance

```text
.mat files -> load -> filter -> standardise -> window -> split -> train -> evaluate -> save
 (raw EMG)    (per subject)  (preprocessing.method)  (StandardScaler)  (400/300)  (80/20 or eval folder)  (model)  (metrics)  (JSON, model, plots)
```

Each step lives in `src/semg_pp_ml/`:

| Step | Module | Notes |
|---|---|---|
| Load | `data/loader.py` | Reads `S<subject>_*.mat` from the training folder and keeps the selected `channels`. |
| Filter | `preprocessing/filters.py` | One of seven methods, chosen by `preprocessing.method`. |
| Standardise | `preprocessing/pipeline.py` | `StandardScaler` per channel, fitted on the filtered training signal. |
| Window | `data/loader.py` | Sliding windows of `window_size` samples, moved by `window_size - overlap`. |
| Split | `data/loader.py`, `pipeline.py` | Uses an `evaluation` folder if it has `.mat` files, otherwise a stratified split. |
| Train | `experiments/runner.py` | Classical models get flattened windows. Deep models get `(window, channels)` tensors. |
| Evaluate | `metrics/classification.py` | Accuracy, precision, recall, F1, AUC, per-class scores and latency. |
| Save | `experiments/runner.py` | Result JSON, model file and plots. See [Results and outputs](outputs.md). |

## 1. Filtering

Each method removes a different kind of noise. Use `raw` as a baseline, then compare.

| Method | What it does | Main settings |
|---|---|---|
| `raw` | No filtering | none |
| `butterworth` | Band-pass filter that keeps the useful EMG band | `low_hz`, `high_hz`, `order` |
| `kalman` | Kalman filter that smooths the signal | `q`, `r` |
| `butterworth_kalman` | Band-pass, then Kalman (default) | both sets |
| `vmd` | Variational mode decomposition, keeps the relevant modes | `alpha`, `k`, ... |
| `wavelet` | Wavelet denoising with thresholding | `wavelet`, `level`, `threshold_mode` |
| `notch` | Removes mains hum (50 or 60 Hz) | `freq`, `quality_factor` |

`vmd` is the slowest method. Preprocessing results are cached (see below), so each method runs once per settings combination.

## 2. Standardisation

After filtering, each channel is scaled to zero mean and unit variance with a `StandardScaler`. The scaler is saved with the cache, so the same scaling can be applied to other data.

> **Caveat.** In the current code the scaler is fitted on all training windows before the train/test split, so the test windows influence the scaling. This is a small, common form of leakage. For a strict evaluation, use an `evaluation` folder with different sessions.

## 3. Windowing and labels

- A window is `signal.window_size` samples long. The default of 400 samples is 200 ms at 2000 Hz.
- The next window starts `window_size - overlap` samples later. The default step is 300 samples.
- Each window takes the **majority label** of its samples. Windows whose label is not in `dataset.classes` are dropped.

## 4. Train and test split

- If the `dataset.eval_dir` folder (default `evaluation`) exists and contains `.mat` files, those files form the test set and the training folder forms the training set.
- Otherwise the windows are split stratified by class. `training.test_size` (default 0.2) goes to the test set. `training.random_state` (default 42) makes the split reproducible.

## 5. Training and evaluation

- Classical models (`svm`, `rf`, `knn`, `lda`, `nb`, `mlp_sklearn`) are fitted once on the flattened windows.
- Deep models train for `training.epochs` epochs with `training.batch_size`. They hold out `training.validation_split` of the training data for validation. The training curve is plotted.
- The test set is used only for the final metrics.
- Each result also records the parameter count, training time, inference latency and throughput.

## 6. Caching

Preprocessing is slow, especially VMD, so results are cached in `outputs/preprocessing_cache/`. The cache key is a hash of the method and its settings, plus the subjects, channels, window size, overlap and sampling rate. Change any of those and a new cache entry is created. Old entries are not removed automatically. Delete the folder to reclaim space or to force recomputation.

## 7. Two ways to run

- **Single run** (`semg train`): one filter, one model.
- **Grid search** (`semg grid`): for each filter, preprocess once, then train every model on it. A failing model is recorded as `FAILED` and the search continues. See [Grid search](grid-search.md).

## Limits to keep in mind

- Window labels come from the majority class, so windows at movement transitions contain mixed labels.
- Results depend on which subjects you train on. Use the full subject set for final numbers.
- The synthetic data generator (see [Augmentation](augmentation.md)) is not validated against real recordings.
