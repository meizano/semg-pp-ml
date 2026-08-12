"""
Data loading utilities for NinaPro DB8 dataset.
"""
from __future__ import annotations

import os
import glob
import numpy as np
import scipy.io as sio
from typing import Optional, Tuple, List
from pathlib import Path


def load_mat_files_from_folder(
    folder_path: str | Path,
    subjects: List[int],
    channels: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load all .mat files from a folder for given subjects.

    Returns:
        emg: (n_samples, n_channels) or (n_samples, n_selected_channels)
        labels: (n_samples,)
        glove: (n_samples, n_dof) or None
    """
    folder_path = Path(folder_path)
    all_emg = []
    all_labels = []
    all_glove = []

    if not folder_path.exists():
        print(f"[INFO] Folder not found: {folder_path}")
        return np.array([]), np.array([]), None

    for subj in subjects:
        pattern = folder_path / f"S{subj}_*.mat"
        files = sorted(glob.glob(str(pattern)))
        print(f"[LOAD] Subject {subj} from '{folder_path.name}': {len(files)} files")

        for fpath in files:
            fname = os.path.basename(fpath)
            try:
                mat = sio.loadmat(fpath)
            except NotImplementedError:
                print(f"       [SKIP] {fname}: v7.3 HDF5 format not supported")
                continue

            # EMG data
            if 'emg' in mat:
                emg = mat['emg']
            elif 'EMG' in mat:
                emg = mat['EMG']
            else:
                print(f"       [SKIP] {fname}: no emg key")
                continue

            # Stimulus/Labels
            if 'stimulus' in mat:
                stim = mat['stimulus']
            elif 'restimulus' in mat:
                stim = mat['restimulus']
            else:
                print(f"       [SKIP] {fname}: no stimulus key")
                continue

            if emg.ndim == 1:
                emg = emg.reshape(-1, 1)
            stim = np.array(stim).squeeze()

            # Select channels if specified
            if channels is not None:
                emg = emg[:, channels]

            all_emg.append(emg)
            all_labels.append(stim)

            # Glove data (optional)
            if 'glove' in mat:
                glove = mat['glove']
                if glove.ndim == 1:
                    glove = glove.reshape(-1, 1)
                all_glove.append(glove)
            else:
                all_glove.append(None)

    if not all_emg:
        return np.array([]), np.array([]), None

    emg_full = np.vstack(all_emg)
    labels_full = np.concatenate(all_labels)

    glove_full = None
    if any(g is not None for g in all_glove):
        if all(g is not None for g in all_glove):
            shapes = [g.shape[1] for g in all_glove]
            if len(set(shapes)) == 1:
                glove_full = np.vstack(all_glove)
            else:
                print(f"[WARN] CyberGlove shape inconsistent: {set(shapes)}. Skipping glove.")
        else:
            print(f"[INFO] Some files missing CyberGlove. Skipping glove.")

    print(f"[OK] Loaded: {emg_full.shape[0]} samples x {emg_full.shape[1]} ch | Glove: {'Yes (' + str(glove_full.shape[1]) + ' DoF)' if glove_full is not None else 'No'}")
    return emg_full, labels_full, glove_full


def sliding_window(
    emg: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    overlap: int,
    selected_classes: List[int],
    glove: Optional[np.ndarray] = None,
    purity_threshold: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Apply sliding window to EMG data with label purity filtering.

    Args:
        emg: (n_samples, n_channels)
        labels: (n_samples,)
        window_size: Number of samples per window
        overlap: Number of overlapping samples
        selected_classes: Classes to include (e.g., [0, 1, 2, 3, 4, 5])
        glove: Optional glove data (n_samples, n_dof)
        purity_threshold: Minimum fraction of majority class in window

    Returns:
        X: (n_windows, window_size, n_channels)
        y: (n_windows,) - majority class labels
        Y_glove: (n_windows, n_dof) or None
    """
    step = window_size - overlap
    if step <= 0:
        raise ValueError(f"OVERLAP ({overlap}) must be less than WINDOW_SIZE ({window_size})")

    n = emg.shape[0]
    X, y = [], []
    Y_glove = [] if glove is not None else None

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        wlab = labels[start:end]

        unique, counts = np.unique(wlab, return_counts=True)
        majority_label = int(unique[np.argmax(counts)])
        purity = counts.max() / window_size

        if majority_label not in selected_classes:
            continue
        if purity < purity_threshold:
            continue

        X.append(emg[start:end, :])
        y.append(majority_label)

        if glove is not None:
            Y_glove.append(np.mean(glove[start:end, :], axis=0))

    X_arr = np.array(X)
    y_arr = np.array(y)

    if Y_glove is not None:
        return X_arr, y_arr, np.array(Y_glove)
    return X_arr, y_arr, None


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    glove: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Stratified train/test split with optional glove data.
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X))
    idx_tr, idx_te = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    glove_tr, glove_te = None, None
    if glove is not None:
        glove_tr, glove_te = glove[idx_tr], glove[idx_te]

    return X_tr, X_te, y_tr, y_te, glove_tr, glove_te


def prepare_dataset(
    train_dir: str | Path,
    eval_dir: str | Path,
    subjects: List[int],
    channels: List[int],
    classes: List[int],
    window_size: int,
    overlap: int,
    test_size: float = 0.2,
    random_state: int = 42,
    has_eval: bool = True
) -> dict:
    """
    Complete dataset preparation pipeline.

    Returns dict with:
        X_tr, y_tr, X_te, y_te, glove_tr, glove_te, le, n_classes
    """
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical

    # Load training data
    emg_train, labels_train, glove_train = load_mat_files_from_folder(
        train_dir, subjects, channels
    )

    # Load evaluation data if available
    emg_eval = np.array([])
    labels_eval = np.array([])
    glove_eval = None

    if has_eval and Path(eval_dir).exists():
        eval_files = list(Path(eval_dir).glob("*.mat"))
        if eval_files:
            emg_eval, labels_eval, glove_eval = load_mat_files_from_folder(
                eval_dir, subjects, channels
            )

    # Windowing
    X_train, y_train, glove_train_win = sliding_window(
        emg_train, labels_train, window_size, overlap, classes, glove_train
    )

    if emg_eval.size > 0:
        X_eval, y_eval, glove_eval_win = sliding_window(
            emg_eval, labels_eval, window_size, overlap, classes, glove_eval
        )
    else:
        X_eval, y_eval, glove_eval_win = np.array([]), np.array([]), None

    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    n_classes = len(np.unique(y_train_enc))

    if X_eval.size > 0:
        y_eval_enc = le.transform(y_eval)
        X_tr, y_tr = X_train, to_categorical(y_train_enc, n_classes)
        X_te, y_te = X_eval, to_categorical(y_eval_enc, n_classes)
        glove_tr, glove_te = glove_train_win, glove_eval_win
    else:
        y_train_cat = to_categorical(y_train_enc, n_classes)
        X_tr, X_te, y_tr, y_te, glove_tr, glove_te = train_test_split_stratified(
            X_train, y_train_cat, test_size, random_state, glove_train_win
        )

    return {
        'X_tr': X_tr, 'y_tr': y_tr,
        'X_te': X_te, 'y_te': y_te,
        'glove_tr': glove_tr, 'glove_te': glove_te,
        'le': le, 'n_classes': n_classes,
        'emg_train_raw': emg_train,
        'labels_train_raw': labels_train,
    }