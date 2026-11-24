# Getting started

This guide takes you from a clean machine to a first trained model.

## 1. Requirements

- Python 3.10 or newer (`python --version`).
- Git, to clone the repository.
- About 2 GB of disk space for the NinaPro DB8 data, plus outputs.
- A GPU is optional. Classical models run on CPU. Deep models are faster on a GPU.

## 2. Install

```bash
git clone https://github.com/meizano/semg-pp-ml.git
cd semg-pp-ml

python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate

pip install -e ".[dl,ui]"
```

`-e` installs the project in editable mode, so changes under `src/` take effect immediately. If you only need classical models and the command line, `pip install -e .` is enough.

Check the install:

```bash
semg --version
semg info              # CPU, RAM and GPU information
```

On Windows, if `semg` is "not recognized", see [Troubleshooting](troubleshooting.md#semg-is-not-recognized). `python -m semg_pp_ml.cli --version` always works.

## 3. Put the data in place

Copy the NinaPro DB8 `.mat` files into `data/ninapro_db8/train/`. See [Data](data.md) for the exact layout. Then check that the platform can see them:

```bash
semg check-data
```

The output should show the folder path, the subjects found, the number of `.mat` files and `EMG key: yes`. If it says `No .mat files found`, fix the path first. Use `--data-dir` to point somewhere else.

## 4. Run your first experiment

```bash
semg train --prep butterworth --model svm --set dataset.subjects=[1,2]
```

What this does:

1. Loads the `.mat` files of subjects 1 and 2.
2. Applies a Butterworth band-pass filter and standardises the signal.
3. Cuts 400-sample windows (step 300) and labels each window by its majority class.
4. Splits the windows into train and test sets (stratified, 80/20) unless an `evaluation` folder exists.
5. Trains an SVM and prints accuracy, F1, precision, recall and latency.
6. Writes the results to `outputs/`.

Subjects 1 and 2 keep the first run quick. Remove `--set dataset.subjects=[1,2]` to use the default (subjects 1 to 10).

## 5. Try a deep model

```bash
semg train --prep vmd --model cnn1d --epochs 20
```

This needs the `dl` extra (PyTorch). Training time depends on your hardware.

## 6. Compare many models

```bash
semg grid --preps raw butterworth --models svm rf knn
```

See [Grid search](grid-search.md).

## 7. Use the web app

```bash
streamlit run app.py
```

Open the URL it prints, usually `http://localhost:8501`. See [Web app](web-app.md).

## Next steps

- [How it works](how-it-works.md): what each step does and why.
- [Configuration](configuration.md): change windows, filters and training settings.
- [CLI reference](cli.md): every command and flag.
