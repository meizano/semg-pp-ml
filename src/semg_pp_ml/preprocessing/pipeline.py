"""
Preprocessing pipeline for sEMG signals.
Handles filtering, scaling, caching, and metrics computation.
"""
from __future__ import annotations

import os
import hashlib
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
import joblib

from ..config import ExperimentConfig, PreprocessingConfig
from .filters import apply_preprocessing, FILTER_REGISTRY
from .metrics import compute_all_preprocessing_metrics
from ..data.loader import load_mat_files_from_folder, sliding_window, prepare_dataset


@dataclass
class PreprocessingResult:
    """Container for preprocessing results."""
    emg_processed: np.ndarray
    emg_scaled: np.ndarray
    scaler: Any  # StandardScaler
    metrics: Dict[str, Any]
    config_hash: str
    method: str
    channels: List[int]
    subjects: List[int]


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline:
    1. Load raw data
    2. Apply filtering
    3. Scale (StandardScaler)
    4. Compute metrics
    5. Cache results
    """

    def __init__(self, config: ExperimentConfig, cache_dir: Optional[Path] = None):
        self.config = config
        self.cache_dir = cache_dir or (config.output_dir / "preprocessing_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_config_hash(self, method: str) -> str:
        """Compute hash of preprocessing configuration for caching."""
        prep_config = self.config.preprocessing
        # Include relevant config parameters
        hash_dict = {
            'method': method,
            'channels': self.config.dataset.channels,
            'subjects': self.config.dataset.subjects,
            'window_size': self.config.signal.window_size,
            'overlap': self.config.signal.overlap,
            'fs': self.config.signal.fs,
        }

        # Add method-specific params
        if method == 'butterworth' or method == 'butterworth_kalman':
            hash_dict['butterworth'] = prep_config.butterworth.model_dump()
        if method == 'kalman' or method == 'butterworth_kalman':
            hash_dict['kalman'] = prep_config.kalman.model_dump()
        if method == 'vmd':
            hash_dict['vmd'] = prep_config.vmd.model_dump()
        if method == 'wavelet':
            hash_dict['wavelet'] = prep_config.wavelet.model_dump()
        if method == 'notch':
            hash_dict['notch'] = prep_config.notch.model_dump()

        hash_str = json.dumps(hash_dict, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def _get_cache_paths(self, config_hash: str) -> Tuple[Path, Path, Path]:
        """Get cache file paths for data, scaler, and metrics."""
        base = self.cache_dir / config_hash
        return (
            base.with_suffix('.npz'),      # processed data
            base.with_name(f"{base.name}_scaler.pkl"),  # scaler
            base.with_name(f"{base.name}_metrics.json")  # metrics
        )

    def _save_to_cache(self, config_hash: str, result: PreprocessingResult) -> None:
        """Save preprocessing results to cache."""
        data_path, scaler_path, metrics_path = self._get_cache_paths(config_hash)

        # Save processed data
        np.savez_compressed(
            data_path,
            emg_processed=result.emg_processed,
            emg_scaled=result.emg_scaled,
            channels=np.array(result.channels),
            subjects=np.array(result.subjects),
        )

        # Save scaler
        joblib.dump(result.scaler, scaler_path)

        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': result.metrics,
                'config_hash': result.config_hash,
                'method': result.method,
                'channels': result.channels,
                'subjects': result.subjects,
            }, f, indent=2)

    def _load_from_cache(self, config_hash: str) -> Optional[PreprocessingResult]:
        """Load preprocessing results from cache."""
        data_path, scaler_path, metrics_path = self._get_cache_paths(config_hash)

        if not (data_path.exists() and scaler_path.exists() and metrics_path.exists()):
            return None

        try:
            data = np.load(data_path)
            scaler = joblib.load(scaler_path)

            with open(metrics_path, 'r') as f:
                meta = json.load(f)

            return PreprocessingResult(
                emg_processed=data['emg_processed'],
                emg_scaled=data['emg_scaled'],
                scaler=scaler,
                metrics=meta['metrics'],
                config_hash=meta['config_hash'],
                method=meta['method'],
                channels=meta['channels'].tolist(),
                subjects=meta['subjects'].tolist(),
            )
        except Exception as e:
            print(f"[Cache] Failed to load cache: {e}")
            return None

    def run(
        self,
        method: Optional[str] = None,
        force_recompute: bool = False,
        save_visualizations: bool = False
    ) -> PreprocessingResult:
        """
        Run the complete preprocessing pipeline.

        Args:
            method: Preprocessing method (uses config default if None)
            force_recompute: Ignore cache and recompute
            save_visualizations: Save signal comparison plots

        Returns:
            PreprocessingResult with processed data, scaler, and metrics
        """
        method = method or self.config.preprocessing.method
        config_hash = self._compute_config_hash(method)

        # Try cache first
        if not force_recompute:
            cached = self._load_from_cache(config_hash)
            if cached is not None:
                print(f"[CACHE] Loaded preprocessing from cache: {config_hash}")
                return cached

        print(f"[PIPELINE] Running preprocessing: {method}")

        # Load raw data
        emg_raw, labels_raw, glove_raw = load_mat_files_from_folder(
            self.config.dataset.train_dir,
            self.config.dataset.subjects,
            self.config.dataset.channels
        )

        # Apply filtering
        emg_filtered = apply_preprocessing(
            emg_raw, self.config.signal.fs, method, self.config.preprocessing
        )

        # Compute metrics on training data
        metrics = compute_all_preprocessing_metrics(emg_raw, emg_filtered, self.config.signal.fs)

        # Scale
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        emg_scaled = scaler.fit_transform(emg_filtered)

        result = PreprocessingResult(
            emg_processed=emg_filtered,
            emg_scaled=emg_scaled,
            scaler=scaler,
            metrics=metrics,
            config_hash=config_hash,
            method=method,
            channels=self.config.dataset.channels,
            subjects=self.config.dataset.subjects,
        )

        # Save to cache
        self._save_to_cache(config_hash, result)

        # Save visualizations if requested
        if save_visualizations:
            self._save_visualizations(emg_raw, emg_filtered, method)

        return result

    def _save_visualizations(self, raw: np.ndarray, filtered: np.ndarray, method: str) -> None:
        """Save signal comparison plots."""
        from ..visualization.signals import plot_signal_comparison, plot_preprocessing_metrics

        viz_dir = self.config.output_dir / "visualizations" / "preprocessing" / method
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Signal comparison
        plot_signal_comparison(
            raw, filtered, self.config.signal.fs,
            self.config.dataset.channels, str(viz_dir), method
        )

        # Metrics
        metrics = compute_all_preprocessing_metrics(raw, filtered, self.config.signal.fs)
        snr_per_ch = metrics['snr_per_channel_db']
        psd_ratios = metrics['psd_band_ratio']
        rmse_per_ch = metrics['rmse_noise_per_channel']
        mae_per_ch = metrics['mae_noise_per_channel']

        plot_preprocessing_metrics(
            snr_per_ch, psd_ratios, rmse_per_ch, mae_per_ch,
            self.config.dataset.channels, str(viz_dir), method,
            metrics['snr_mean_db'], metrics['psd_mean_ratio'],
            metrics['rmse_noise_mean'], metrics['mae_noise_mean']
        )

    def prepare_for_training(
        self,
        method: Optional[str] = None,
        eval_dir: Optional[str] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Prepare complete dataset for training (windowing, encoding, splitting).

        Returns dict with X_tr, y_tr, X_te, y_te, le, n_classes, glove data, etc.
        """
        method = method or self.config.preprocessing.method
        result = self.run(method, force_recompute)

        # Load evaluation data if available
        has_eval = False
        emg_eval_raw = np.array([])
        labels_eval_raw = np.array([])
        glove_eval_raw = None

        if eval_dir or self.config.dataset.eval_dir:
            eval_path = Path(eval_dir or self.config.dataset.eval_dir)
            if eval_path.exists():
                eval_files = list(eval_path.glob("*.mat"))
                if eval_files:
                    has_eval = True
                    emg_eval_raw, labels_eval_raw, glove_eval_raw = load_mat_files_from_folder(
                        eval_path, self.config.dataset.subjects, self.config.dataset.channels
                    )

        # Windowing
        X_train, y_train, glove_train = sliding_window(
            result.emg_scaled, labels_raw,
            self.config.signal.window_size, self.config.signal.overlap,
            self.config.dataset.classes, glove_raw
        )

        if has_eval and emg_eval_raw.size > 0:
            # Apply same preprocessing to eval
            emg_eval_filtered = apply_preprocessing(
                emg_eval_raw, self.config.signal.fs, method, self.config.preprocessing
            )
            emg_eval_scaled = result.scaler.transform(emg_eval_filtered)

            X_eval, y_eval, glove_eval = sliding_window(
                emg_eval_scaled, labels_eval_raw,
                self.config.signal.window_size, self.config.signal.overlap,
                self.config.dataset.classes, glove_eval_raw
            )
        else:
            X_eval, y_eval, glove_eval = np.array([]), np.array([]), None

        # Label encoding
        from sklearn.preprocessing import LabelEncoder
        from tensorflow.keras.utils import to_categorical

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        n_classes = len(np.unique(y_train_enc))
        y_train_cat = to_categorical(y_train_enc, n_classes)

        if X_eval.size > 0:
            y_eval_enc = le.transform(y_eval)
            y_eval_cat = to_categorical(y_eval_enc, n_classes)
            X_tr, y_tr = X_train, y_train_cat
            X_te, y_te = X_eval, y_eval_cat
            glove_tr, glove_te = glove_train, glove_eval
        else:
            from ..data.loader import train_test_split_stratified
            X_tr, X_te, y_tr, y_te, glove_tr, glove_te = train_test_split_stratified(
                X_train, y_train_cat,
                self.config.training.test_size,
                self.config.training.random_state,
                glove_train
            )

        return {
            'X_tr': X_tr, 'y_tr': y_tr,
            'X_te': X_te, 'y_te': y_te,
            'glove_tr': glove_tr, 'glove_te': glove_te,
            'le': le, 'n_classes': n_classes,
            'preprocessing_result': result,
            'method': method,
        }

    def list_cached(self) -> List[Dict[str, Any]]:
        """List all cached preprocessing results."""
        cached = []
        for metrics_file in self.cache_dir.glob("*_metrics.json"):
            try:
                with open(metrics_file, 'r') as f:
                    meta = json.load(f)
                cached.append({
                    'config_hash': meta['config_hash'],
                    'method': meta['method'],
                    'channels': meta['channels'],
                    'subjects': meta['subjects'],
                    'snr_mean_db': meta['metrics'].get('snr_mean_db'),
                    'psd_mean_ratio': meta['metrics'].get('psd_mean_ratio'),
                })
            except Exception:
                pass
        return cached

    def clear_cache(self, config_hash: Optional[str] = None) -> None:
        """Clear cache for specific hash or all."""
        if config_hash:
            for path in self.cache_dir.glob(f"{config_hash}*"):
                path.unlink(missing_ok=True)
        else:
            for path in self.cache_dir.glob("*"):
                path.unlink(missing_ok=True)


def run_preprocessing_cli(
    config_path: str,
    method: str,
    output_path: str,
    force: bool = False,
    viz: bool = False
) -> None:
    """CLI entry point for preprocessing only."""
    config = load_config(config_path)
    config.preprocessing.method = method

    pipeline = PreprocessingPipeline(config)
    result = pipeline.run(method=method, force_recompute=force, save_visualizations=viz)

    # Save processed data for later use
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        emg_processed=result.emg_processed,
        emg_scaled=result.emg_scaled,
        config_hash=result.config_hash,
        method=result.method,
        channels=np.array(result.channels),
        subjects=np.array(result.subjects),
    )
    joblib.dump(result.scaler, output_path.with_name(f"{output_path.stem}_scaler.pkl"))

    with open(output_path.with_name(f"{output_path.stem}_metrics.json"), 'w') as f:
        json.dump(result.metrics, f, indent=2)

    print(f"[OK] Preprocessing complete. Saved to {output_path}")
    print(f"     SNR: {result.metrics['snr_mean_db']:.2f} dB")
    print(f"     PSD Ratio: {result.metrics['psd_mean_ratio']:.4f}")
    print(f"     RMSE: {result.metrics['rmse_noise_mean']:.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run sEMG preprocessing")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--method", required=True, help="Preprocessing method")
    parser.add_argument("--output", required=True, help="Output file path (.npz)")
    parser.add_argument("--force", action="store_true", help="Force recompute")
    parser.add_argument("--viz", action="store_true", help="Save visualizations")
    args = parser.parse_args()

    run_preprocessing_cli(args.config, args.method, args.output, args.force, args.viz)