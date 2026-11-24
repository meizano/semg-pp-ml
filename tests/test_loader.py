"""Windowing and loading: the vectorised sliding_window must match the original loop."""
import numpy as np
import pytest

from semg_pp_ml.data.loader import load_mat_files_from_folder, one_hot, sliding_window


def _reference_sliding_window(emg, labels, window_size, overlap, selected_classes,
                              glove=None, purity_threshold=0.0):
    """The original (loop-based) implementation, kept here as the specification."""
    step = window_size - overlap
    X, y, G = [], [], []
    for start in range(0, emg.shape[0] - window_size + 1, step):
        end = start + window_size
        unique, counts = np.unique(labels[start:end], return_counts=True)
        majority = int(unique[np.argmax(counts)])
        purity = counts.max() / window_size
        if majority not in selected_classes or purity < purity_threshold:
            continue
        X.append(emg[start:end, :])
        y.append(majority)
        if glove is not None:
            G.append(np.mean(glove[start:end, :], axis=0))
    return np.array(X), np.array(y), (np.array(G) if glove is not None else None)


@pytest.mark.parametrize("window,overlap,purity", [(40, 10, 0.0), (25, 0, 0.6), (16, 15, 0.9)])
def test_sliding_window_matches_reference(window, overlap, purity):
    rng = np.random.default_rng(3)
    n = 1000
    emg = rng.standard_normal((n, 4))
    # blocky labels with some noise so that purity filtering matters
    labels = np.repeat(rng.integers(0, 7, size=n // 50 + 1), 50)[:n]
    labels[rng.random(n) < 0.05] = 9
    glove = rng.standard_normal((n, 3))
    classes = [0, 1, 2, 3, 4, 5, 6]

    X, y, G = sliding_window(emg, labels, window, overlap, classes, glove, purity)
    Xr, yr, Gr = _reference_sliding_window(emg, labels, window, overlap, classes, glove, purity)

    assert X.shape == Xr.shape and np.array_equal(X, Xr)
    assert np.array_equal(y, yr)
    assert np.allclose(G, Gr)


def test_sliding_window_respects_selected_classes_and_ties():
    emg = np.zeros((12, 2))
    labels = np.array([3, 3, 1, 1, 3, 3, 1, 1, 3, 3, 1, 1])   # exact ties in every window
    X, y, G = sliding_window(emg, labels, window_size=4, overlap=0, selected_classes=[1, 3])
    # every window is an exact tie between 1 and 3; the smallest label wins, as in the reference
    assert list(y) == [1, 1, 1]
    _, yr, _ = _reference_sliding_window(emg, labels, 4, 0, [1, 3])
    assert list(y) == list(yr)
    X2, y2, _ = sliding_window(emg, labels, 4, 0, selected_classes=[9])
    assert X2.shape[0] == 0 and y2.shape == (0,)


def test_sliding_window_empty_and_invalid():
    X, y, G = sliding_window(np.zeros((5, 2)), np.zeros(5), 10, 2, [0])
    assert X.shape == (0, 10, 2) and G is None
    with pytest.raises(ValueError):
        sliding_window(np.zeros((5, 2)), np.zeros(5), 10, 10, [0])


def test_one_hot():
    out = one_hot(np.array([0, 2, 1]), 3)
    assert out.dtype == np.float32
    assert np.array_equal(out, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32))


def test_missing_folder_raises_instead_of_inventing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset folder not found"):
        load_mat_files_from_folder(tmp_path / "nope", [1])


def test_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No usable"):
        load_mat_files_from_folder(tmp_path, [1])


def test_loads_fake_dataset(fake_dataset_root):
    emg, labels, glove = load_mat_files_from_folder(fake_dataset_root / "train", [1, 2], [3, 4, 5, 11, 12, 13])
    assert emg.shape[1] == 6 and emg.shape[0] == labels.shape[0] == glove.shape[0]
    assert set(np.unique(labels)) <= set(range(0, 7))
