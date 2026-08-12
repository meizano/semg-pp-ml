"""
Standalone CLI Runner for sEMG Preprocessing & Machine Learning Platform.
Allows running single operations or grid searches without requiring the web UI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from semg_pp_ml.config import load_config
from semg_pp_ml.experiments.runner import run_single_experiment
from semg_pp_ml.experiments.grid_search import run_grid_search_optimized
from semg_pp_ml.utils.system import get_system_info, print_system_info


def main():
    parser = argparse.ArgumentParser(description="sEMG ML Platform CLI Runner")
    parser.add_argument("--mode", type=str, choices=["single", "grid", "info"], default="single", help="Execution mode")
    parser.add_argument("--prep", type=str, default="wavelet", help="Preprocessing method (e.g. wavelet, vmd, notch, kalman, butterworth)")
    parser.add_argument("--model", type=str, default="cnn1d", help="Model type (e.g. svm, rf, cnn1d, lstm, transformer)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="data/ninapro_db8", help="Path to dataset directory")
    parser.add_argument("--config", type=str, default=None, help="Path to custom YAML config file")

    args = parser.parse_args()

    print("="*70)
    print(" sEMG Preprocessing & Machine Learning Platform (CLI Runner)")
    print("="*70)

    if args.mode == "info":
        sys_info = get_system_info()
        print_system_info(sys_info)
        return

    config = load_config(args.config)
    config.dataset.train_dir = args.data_dir
    config.preprocessing.method = args.prep
    config.model.type = args.model
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size

    if args.mode == "single":
        print(f"[CLI] Running Single Experiment: Preprocessing={args.prep} | Model={args.model}")
        result = run_single_experiment(config)
        print(f"[SUCCESS] Result Status: {result.status}")
        print(f"          Test Accuracy: {result.metrics.get('accuracy', 0)*100:.2f}%")
        print(f"          CSV Log Saved in outputs/logs/")
    elif args.mode == "grid":
        print(f"[CLI] Running Optimized Grid Search...")
        grid_res = run_grid_search_optimized(config)
        print(f"[SUCCESS] Grid Search Completed in {grid_res.total_time:.2f} seconds.")
        print(f"          Summary CSV saved in outputs/grid_results/")


if __name__ == "__main__":
    main()
