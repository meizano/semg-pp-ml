# Results and outputs

Everything the platform writes goes under the output folder: `outputs/` by default, or the folder set with `--output` or `output_dir`.

## Folder layout

```text
outputs/
├── result_svm_butterworth.json         <- one file per `semg train` run
├── model_svm_butterworth.pkl           <- trained model (classical: .pkl, deep: .pt)
├── visualizations/
│   └── svm_butterworth/                <- plots for that run
│       ├── confusion_matrix_svm_butterworth.png
│       └── ...
├── preprocessing_cache/                <- cached filtered signals, scaler and metrics
│   ├── 3f9a1c0d2b7e4a51.npz
│   ├── 3f9a1c0d2b7e4a51_scaler.pkl
│   └── 3f9a1c0d2b7e4a51_metrics.json
├── grid_search_summary.csv             <- `semg grid`: one row per run
└── grid_search_results.json            <- `semg grid`: best run per metric
```

File names combine the model and preprocessing method, so runs do not overwrite each other unless you repeat the same pair.

## `result_<model>_<prep>.json`

```json
{
  "model_type": "svm",
  "prep_method": "butterworth",
  "metrics": {
    "accuracy": 0.52,
    "f1_score": 0.49,
    "precision": 0.49,
    "recall": 0.52,
    "inference_latency_ms": 1.1,
    "prep_snr_mean_db": 12.3
  },
  "training_time": 4.8,
  "inference_time": 0.02,
  "epochs_run": 1,
  "n_params": 0,
  "status": "OK",
  "error": null
}
```

| Field | Meaning |
|---|---|
| `status` | `OK`, or `FAILED` with `error` describing the problem |
| `metrics` | test-set results (see below) |
| `training_time`, `inference_time` | seconds |
| `epochs_run` | number of epochs (1 for classical models) |
| `n_params` | number of model parameters |

## Metrics

| Metric | Meaning |
|---|---|
| `accuracy` | share of test windows labelled correctly |
| `precision`, `recall`, `f1_score` | weighted averages over classes |
| `auc_roc_weighted` | area under the ROC curve, weighted by class frequency |
| `specificity_per_class` | per-class specificity |
| `inference_latency_ms` | average time to classify one window |
| `avg_throughput_samples_per_sec` | training throughput |
| `prep_*` | signal-quality metrics of the preprocessing: SNR, PSD band ratio and RMSE, prefixed with `prep_` |

Accuracy alone can mislead when classes are unequal. Compare F1 and recall too.

## Plots

- **Confusion matrix** for every run (not for `--no-viz`).
- **Training curves** for deep models (loss and accuracy per epoch).

## Models

- Classical models are saved with `joblib` as `.pkl`.
- Deep models are saved as `.pt` (PyTorch). Load them with `semg_pp_ml.models.registry.load_model(path, model_type)`.
- `--no-save` skips saving the model file.

## Preprocessing cache

The cache holds the filtered signal, the fitted scaler and the metrics. It is reused whenever the preprocessing settings are unchanged. Delete the folder to force recomputation:

```bash
rm -rf outputs/preprocessing_cache       # macOS / Linux
Remove-Item -Recurse outputs\preprocessing_cache   # Windows PowerShell
```

## Grid search summary

`grid_search_summary.csv` has one row per (preprocessing, model) pair:

- `prep`, `model`, `status`
- the metrics above, for `OK` rows
- `error`, for `FAILED` rows

`grid_search_results.json` contains the total run time, the grid settings, and the best run for each of `accuracy`, `f1_score`, `precision`, `recall` and `auc_roc_weighted`.

## Reading results in Python

```python
import pandas as pd

df = pd.read_csv("outputs/grid_search_summary.csv")
ok = df[df["status"] == "OK"]
print(ok.sort_values("f1_score", ascending=False)[["prep", "model", "accuracy", "f1_score"]].head(10))
```
