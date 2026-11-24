# sEMG Preprocessing & Machine Learning Platform (NinaPro DB8)

Surface EMG preprocessing, feature/window generation and model training for the
**NinaPro DB8** dataset: classical models, deep learning, transformers, graph networks
and TinyML export, with a CLI, a Streamlit app and an optional REST API.

- **Preprocessing**: Butterworth, Kalman, hybrid Butterworth-Kalman, VMD, wavelet and notch, with SNR / PSD / RMSE metrics. Results are cached on disk.
- **Two paradigms**: classification (gestures from `stimulus`) and regression-ready glove data (`glove`).
- **Experiments**: single runs and grid searches (preprocessing x model) that compute each preprocessing once.
- **Augmentation**: label-preserving augmentation and a physiology-inspired synthetic sEMG generator (see below).

---

## Documentation

Full guides are in [`docs/`](docs/README.md):

- [Getting started](docs/getting-started.md): install and run your first experiment
- [How it works](docs/how-it-works.md): the pipeline explained
- [CLI reference](docs/cli.md) and [Configuration](docs/configuration.md)
- [Data](docs/data.md), [Models](docs/models.md), [Grid search](docs/grid-search.md), [Results](docs/outputs.md)
- [Augmentation](docs/augmentation.md), [Web app](docs/web-app.md), [REST API](docs/api.md)
- [Troubleshooting](docs/troubleshooting.md)

---

## 1. Install

Requires Python 3.10+.

```bash
git clone https://github.com/meizano/semg-pp-ml.git
cd semg-pp-ml
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -e ".[dl,ui]"
```

Pick only the extras you need:

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* | numpy, scipy, pandas, scikit-learn, PyWavelets, vmdpy, pydantic | preprocessing, classical models, CLI, augmentation |
| `dl` | torch | CNN, LSTM, transformer, TCN, graph and TinyML models |
| `ui` | streamlit, plotly | the web app |
| `tracking` | mlflow | experiment tracking |
| `api` | FastAPI, SQLAlchemy, Celery, JWT | the optional REST backend |
| `dev` | pytest, ruff, black | running tests |
| `all` | everything above | |

Check the install:

```bash
semg info
```

## 2. Get the data

Put the NinaPro DB8 `.mat` files (`S<subject>_*.mat`, with `emg`, `stimulus` and optionally `glove`) in one folder:

```text
data/ninapro_db8/          <- dataset.root (default)
└── train/                 <- dataset.train_dir (default)
    ├── S1_A1_E1.mat
    └── ...
```

If the `.mat` files sit directly in `data/ninapro_db8/`, that works too. Check the layout before training:

```bash
semg check-data --data-dir data/ninapro_db8
```

## 3. Run

```bash
# one experiment: preprocessing + model
semg train --prep butterworth_kalman --model svm --set dataset.subjects=[1,2,3]

# deep model, fewer epochs
semg train --prep vmd --model cnn1d --epochs 20

# grid search over preprocessing x model (exclusions come from config/grid_search.yaml)
semg grid --grid-config config/grid_search.yaml --preps raw vmd --models svm rf

# web app
streamlit run app.py
```

Results go to `outputs/` (override with `--output`): JSON per run, the trained model, plots and `grid_search_summary.csv`.

`python run_cli.py --mode single|grid|info` still works and maps to the commands above.

### Configuration

Settings come from a YAML file (default `config/default.yaml`), then from the flags, then from `--set`:

```bash
semg train --config my.yaml --set training.epochs=5 --set model.svm.C=3 --set preprocessing.method=vmd
```

- Values are parsed as YAML (`1e-3`, `[1, 2]`, `true`, `null`).
- Unknown keys are an error (no silent typos). The same applies to YAML files.
- Create starter files with `semg init-config`.

Main keys:

| Key | Meaning | Default |
|---|---|---|
| `dataset.root` | dataset folder | `data/ninapro_db8` |
| `dataset.train_dir` / `eval_dir` | subfolders of `root` (absolute paths also work) | `train` / `evaluation` |
| `dataset.subjects`, `channels`, `classes` | selection | subjects 1-10 |
| `signal.fs`, `window_size`, `overlap` | windowing (samples) | 2000, 400, 100 |
| `preprocessing.method` | `raw`, `butterworth`, `kalman`, `butterworth_kalman`, `vmd`, `wavelet`, `notch` | `butterworth_kalman` |
| `training.*` | epochs, batch size, learning rate, split | |
| `model.type` | model name (`svm`, `rf`, `knn`, `cnn1d`, `lstm`, `transformer`, ...) | `cnn1d` |

## 4. Python API

```python
from semg_pp_ml.config import load_config, apply_overrides
from semg_pp_ml.experiments import run_single_experiment, run_grid_search_optimized

cfg = apply_overrides(load_config(), ["dataset.subjects=[1,2]", "training.epochs=5"])
result = run_single_experiment(cfg, prep_method="raw", model_type="svm")
print(result.status, result.metrics["accuracy"])
```

## 5. Augmentation and artificial data

`semg_pp_ml.augmentation` (numpy/scipy only) provides:

- **Label-preserving augmentations** for `(n_windows, n_samples, n_channels)` arrays: Gaussian noise at a target SNR, amplitude scaling, time warping, electrode shift, cross-talk, channel dropout, baseline wander, mains hum and same-class mixup.
- **A physiology-inspired synthetic generator**: motor-unit pool with size-principle recruitment, biphasic MUAP templates and class-specific spatial patterns. Higher-force classes have higher RMS (covered by tests). It is not biophysically validated, so check it against real recordings before relying on it.

```bash
semg augment --synthetic --n-classes 6 --per-class 200 --out artificial.npz
semg augment --input windows.npz --copies 2 --out augmented.npz
```

A survey of related repositories and what to take from each is in [`docs/related_work.md`](docs/related_work.md).

## 6. Web app and API

- `streamlit run app.py`: dataset check, preprocessing lab, model training, grid search, saved artifacts, TinyML export. The Grid Search and Preprocessing Lab pages have a dataset-folder field in the sidebar.
- `uvicorn semg_pp_ml.api.main:app --reload` (needs the `api` extra): REST backend with JWT auth, background jobs and a SQLite database by default (`DATABASE_URL` for PostgreSQL). Interactive docs at `/docs`.

## 7. Google Colab

```python
!git clone https://github.com/meizano/semg-pp-ml.git
%cd semg-pp-ml
!pip install -e ".[dl]"
from google.colab import drive; drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/NinaPro_DB8 ./data/ninapro_db8
!semg train --prep wavelet --model cnn1d --epochs 30
```

## 8. Development

```bash
pip install -e ".[dev,dl]"
pytest                 # runs in about 20 s on a laptop
```

Layout:

```text
src/semg_pp_ml/
  cli.py              `semg` command (train, grid, check-data, info, init-config, augment)
  config/             pydantic config models, YAML loading, --set overrides
  data/               .mat loading, vectorised sliding windows, train/test split
  preprocessing/      filters, pipeline with caching, signal metrics
  models/             classical, deep learning, transformer, graph, embedded models
  experiments/        runner (one run), grid search, tracking
  augmentation/       augmentation transforms and synthetic sEMG generator
  metrics/ visualization/ utils/
  api/                optional FastAPI backend
app.py  pages/        Streamlit UI
tests/                pytest suite
```

## License

MIT (see `LICENSE` once added; the repository does not yet ship a license file).
