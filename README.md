# sEMG Preprocessing & Machine Learning Platform for NinaPro DB8

[![Python Version](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Surface Electromyography (sEMG) signal preprocessing and machine learning platform designed for the **NinaPro DB8** dataset. It supports both **Classification** (discrete gesture recognition) and **Regression** (continuous finger kinematics decoding using CyberGlove data), featuring a rich Model Zoo (25+ architectures), detailed epoch-by-epoch datetime CSV logging, real-time performance benchmarking (WPS/FPS & CPU/RAM usage), a standalone CLI runner, Google Colab support, bilingual UI (English & Bahasa Indonesia), and an interactive Streamlit web application with process animations and pipeline visualizers.

---

## 🌟 Key Features

1. **Signal Preprocessing Pipeline**:
   - Butterworth filters (Bandpass / Notch)
   - Wavelet Denoising
   - Variational Mode Decomposition (VMD)
   - Kalman Filtering & Hybrid Butterworth-Kalman pipelines
   - SNR, PSD periodograms, and RMSE evaluation metrics.

2. **Extensive Model Zoo (25+ Models)**:
   - **Classical ML**: SVM, Random Forest, KNN, LDA, Naive Bayes, Scikit-Learn MLP.
   - **Deep Learning (PyTorch)**: MLP, CNN 1D, CNN 2D, Simple RNN, LSTM, BiLSTM, GRU, GRU-LSTM, CNN-RNN, CNN-LSTM, TCN.
   - **Transformers**: Transformer Encoder, CNN-Transformer, Multi-Scale Attention Fusion Network (MAFN), mEMG Transformer.
   - **Graph Networks**: GCN, GAT treating EMG channels as graph nodes.
   - **Embedded / TinyML**: TinyMLP, TinyCNN with ONNX export and Arduino C++ sketch generation (`.ino`) for microcontrollers.

3. **Dual Paradigm Support (Classification vs Regression)**:
   - **Classification**: Recognizes discrete hand/finger gestures using NinaPro `stimulus` labels (CrossEntropyLoss).
   - **Regression**: Decodes continuous joint angles using NinaPro `glove` (CyberGlove) kinematic data (MSE/MAE loss).

4. **Detailed CSV Logging with Datetime & Epoch Steps**:
   - Automatically records every training session and evaluation run into timestamped CSV log files (`outputs/logs/`) capturing ISO datetime, epoch-by-epoch loss, accuracy, validation metrics, and learning rate.

5. **Bilingual Support (English & Bahasa Indonesia)**:
   - Easily switch languages via JSON configuration files (`config/locales/lang_eng.json` and `config/locales/lang_id.json`).

6. **Standalone CLI Runner & Google Colab Support**:
   - Run single operations or grid searches from the terminal via `run_cli.py` or run interactively in **Google Colab** using `colab_runner.py`.

7. **Real-Time Performance & Resource Benchmarking**:
   - Measures inference throughput (**Windows Per Second / WPS**), latency, and system resource utilization (**CPU % & RAM MB** via `psutil`).

---

## 📦 Installation Guide

### Prerequisites
- Python 3.10 or higher
- Git

### Step-1: Clone the Repository
```bash
git clone https://github.com/your-username/semg-pp-ml.git
cd semg-pp-ml
```

### Step-2: Create & Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step-3: Install Dependencies
```bash
pip install --upgrade pip
pip install -e .
```

---

## 🚀 How to Use

### 1. Run the Interactive Web App (Streamlit)
```bash
streamlit run app.py
```
Opens the web app at `http://localhost:8501`.

### 2. Run via Command Line Interface (CLI)
```bash
python run_cli.py --mode single --prep wavelet --model cnn1d --epochs 20
```
Or run a grid search:
```bash
python run_cli.py --mode grid
```

### 3. Run on Google Colab
Open a notebook in Google Colab and run:
```python
!git clone https://github.com/your-username/semg-pp-ml.git
%cd semg-pp-ml
!pip install -e .
!python run_cli.py --mode single --prep wavelet --model cnn1d --epochs 25
```

---

## 🪟 Windows User Tips & Troubleshooting

- **PowerShell Execution Policy**: If script activation fails, run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **LAN Access**: Run Streamlit across your network:
  ```bash
  streamlit run app.py --server.address 0.0.0.0
  ```

---

## 🧪 Data Augmentation & Artificial sEMG Data

`src/semg_pp_ml/augmentation/` (numpy/scipy only) provides:

- **Label-preserving augmentations** for `(n_windows, n_samples, n_channels)` arrays: Gaussian noise at a target SNR, amplitude scaling, time warping, electrode shift, cross-talk, channel dropout, baseline wander, mains hum, and same-class mixup.
- **A physiology-inspired synthetic generator**: motor-unit pool with size-principle recruitment, biphasic MUAP templates, and class-specific spatial patterns. Higher force classes produce larger RMS (covered by tests).

```bash
# Generate 200 artificial windows per class for 6 gesture classes
python -m semg_pp_ml.augmentation --synthetic --n-classes 6 --per-class 200 --out artificial.npz

# Augment an existing dataset (X, y) with two augmented copies per window
python -m semg_pp_ml.augmentation --input windows.npz --copies 2 --out augmented.npz

# Run tests
pytest tests/test_augmentation.py
```

The synthetic model is not biophysically validated. Check its output against real recordings before relying on it. A survey of related repositories and what to take from each is in [`docs/related_work.md`](docs/related_work.md).

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
