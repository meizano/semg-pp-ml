# Web app (Streamlit)

The web app is a dashboard for exploring the data, running experiments and reviewing results without the command line.

## Start it

```bash
pip install -e ".[ui]"        # once
streamlit run app.py
```

Open the address Streamlit prints, usually `http://localhost:8501`. To reach it from another computer on your network:

```bash
streamlit run app.py --server.address 0.0.0.0
```

## Pages

| Page | What you do there |
|---|---|
| **Home** (`app.py`) | See system information, check the dataset folder, and jump to the other pages. |
| **Preprocessing Lab** | Try filters (Butterworth, Kalman, VMD, wavelet, notch) on a signal, compare signal-quality metrics (SNR, PSD band power, RMSE), and export a preprocessed `.npz`. |
| **Model Training** | Train a model from the model list, follow the epoch log, and see inference benchmarks (throughput, latency, CPU and RAM). |
| **Grid Search** | Set up a batch of preprocessing x model runs, with exclusion rules, and see the ranked results. |
| **Model Performance** | Load a saved model and review its metrics and confusion matrix. |
| **Saved Artifacts** | Browse the models, plots and cached files in `outputs/`. |
| **TinyML Export** | Export a small model to ONNX and generate an Arduino sketch. |

Pages are listed in the sidebar. The home page also has buttons to the Preprocessing Lab and the Model Training Lab.

## Dataset folder

The Preprocessing Lab and Grid Search pages have a **Train Data Folder** box in the sidebar. It defaults to `data/ninapro_db8`. Change it to point at another dataset folder. Use the folder that contains the `.mat` files or its parent, as described in [Data](data.md).

## Tips

- Start with a small subject selection to get quick feedback.
- For long jobs (large grid searches), use the [CLI](cli.md) and `semg grid` instead. They are easier to leave running and to script.
- The web app uses the same library code as the command line, so results from either should match. The Model Training page has its own loading code (see below), so compare its numbers with the CLI once before relying on them.

## Known limits

- Some pages duplicate loading logic from the library (for example Model Training). Results match the CLI, but changes to the library may need matching changes in those pages.
- The pages are meant for a single user on a trusted network. They have no login.
