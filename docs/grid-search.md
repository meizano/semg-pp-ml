# Grid search

A grid search trains every combination of preprocessing method and model, then ranks them. It is the easiest way to find a good setup for your data.

## Run it

```bash
# use the grid defined in config/grid_search.yaml
semg grid --grid-config config/grid_search.yaml

# quick comparison from the command line
semg grid --preps raw butterworth --models svm rf knn --no-viz
```

Without `--grid-config` and without `--preps` or `--models`, the built-in defaults are used: five preprocessing methods and nineteen models. That is a long run, so narrow it down first.

## How it runs

1. For each preprocessing method, the platform preprocesses and windows the data **once** (cached on disk, see [How it works](how-it-works.md#6-caching)).
2. Every model that is not excluded for that method is trained on the prepared data.
3. A model that fails (for example, a configuration error or an out-of-memory error) is recorded as `FAILED` with the error message, and the search continues.
4. At the end, results are sorted and the best run for each metric is recorded.

## Excluding combinations

Some pairs make no sense, for example classical models on raw, unfiltered data. List them as `exclude_pairs`:

```yaml
exclude_pairs:
  - ["raw", "svm"]
  - ["raw", "knn"]
```

A preprocessing method whose models are all excluded is skipped entirely.

## Running in parallel

```bash
semg grid --grid-config config/grid_search.yaml --parallel --workers 2
```

Models for the same preprocessing method train in separate processes. This helps most for classical models. Deep models already use all cores or the GPU, so parallel runs may not be faster. Each worker needs memory for the prepared data.

## Reading the results

The summary table is written to `outputs/grid_search_summary.csv`, with one row per run:

| Column | Meaning |
|---|---|
| `prep`, `model` | the combination |
| `status` | `OK` or `FAILED` |
| `accuracy`, `f1_score`, `precision`, `recall`, `auc_roc_weighted` | test-set metrics |
| `error` | the error message, for failed runs |

The best run for each metric is in `outputs/grid_search_results.json`.

Open the CSV in a spreadsheet and sort by `f1_score`, or use the **Grid Search** page of the [web app](web-app.md).

## Tips

- Compare models on the same preprocessing first, then vary the preprocessing for the best model.
- Keep `--no-viz` for large grids. Plots are saved for every run otherwise.
- For final numbers, use all subjects and an evaluation folder of separate sessions (see [Data](data.md)).
