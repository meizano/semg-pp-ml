"""
Data loading and preprocessing package.
"""
from .loader import (
    load_mat_files_from_folder,
    sliding_window,
    train_test_split_stratified,
    prepare_dataset,
)

__all__ = [
    "load_mat_files_from_folder",
    "sliding_window",
    "train_test_split_stratified",
    "prepare_dataset",
]