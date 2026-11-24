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


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Integer labels -> one-hot matrix (replaces the TensorFlow ``to_categorical``)."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


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

    if not folder_path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder_path}. Set dataset.root / dataset.train_dir "
            f"in the config, or pass --data-dir."
        )

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
        raise FileNotFoundError(
            f"No usable S<subject>_*.mat files for subjects {list(subjects)} in {folder_path}"
        )

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
            print("[INFO] Some files missing CyberGlove. Skipping glove.")

    print(f"[OK] Loaded: {emg_full.shape[0]} samples x {emg_full.shape[1]} ch | Glove: {'Yes (' + str(glove_full.shape[1]) + ' DoF)' if glove_full is not None else 'No'}")
    return emg_full, labels_full, glove_full


def sliding_window(
    emg: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    overlap: int,
    selected_classes: List[int],
    glove: Optional[np.ndarray] = None,
    purity_threshold: float = 0.0
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

    emg = np.asarray(emg)
    labels = np.asarray(labels)
    n = emg.shape[0]
    n_win = (n - window_size) // step + 1 if n >= window_size else 0
    if n_win == 0:
        empty_glove = None if glove is None else np.empty((0, glove.shape[1]))
        return (np.empty((0, window_size, emg.shape[1]), dtype=emg.dtype),
                np.empty((0,), dtype=np.int64), empty_glove)

    # Windows are processed in chunks so the index matrix stays small even when the
    # overlap is large (many windows). Each chunk is vectorised; there is no per-window loop.
    offsets = np.arange(window_size)[None, :]
    chunk = max(1, 2_000_000 // window_size)     # ~16 MB of indices per chunk
    X_parts, y_parts, g_parts = [], [], []
    for first in range(0, n_win, chunk):
        starts = np.arange(first, min(first + chunk, n_win)) * step
        idx = starts[:, None] + offsets                       # (k, window_size)
        win_labels = labels[idx]

        # Majority label per window. Counting per distinct value (few classes) and taking
        # argmax picks the smallest label on ties, matching the previous np.unique logic.
        values = np.unique(win_labels)
        counts = np.stack([(win_labels == v).sum(axis=1) for v in values], axis=1)
        majority = values[np.argmax(counts, axis=1)].astype(np.int64)
        purity = counts.max(axis=1) / window_size

        keep = np.isin(majority, list(selected_classes)) & (purity >= purity_threshold)
        kept = idx[keep]
        X_parts.append(emg[kept])                             # (m, window_size, n_channels)
        y_parts.append(majority[keep])
        if glove is not None:
            g_parts.append(glove[kept].mean(axis=1))

    X_arr = np.concatenate(X_parts) if X_parts else np.empty((0, window_size, emg.shape[1]), emg.dtype)
    y_arr = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=np.int64)
    Y_glove = (np.concatenate(g_parts) if g_parts else np.empty((0, glove.shape[1]))) \
        if glove is not None else None
    return X_arr, y_arr, Y_glove


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
        X_tr, y_tr = X_train, one_hot(y_train_enc, n_classes)
        X_te, y_te = X_eval, one_hot(y_eval_enc, n_classes)
        glove_tr, glove_te = glove_train_win, glove_eval_win
    else:
        y_train_cat = one_hot(y_train_enc, n_classes)
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