"""
Command line interface for the sEMG platform, installed as the ``semg`` command.

    semg info                              system and GPU information
    semg init-config                       write starter YAML files into ./config
    semg check-data --data-dir PATH        verify the NinaPro layout before training
    semg train --prep vmd --model cnn1d    one preprocessing + model experiment
    semg grid --preps raw vmd --models svm rf
    semg augment --synthetic --out x.npz   artificial sEMG windows (see augmentation)

Every config value can be overridden on any run with ``--set key.path=value``,
e.g. ``--set training.epochs=5 --set model.svm.C=3``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import apply_overrides, create_default_configs, load_config
from .config.models import DEFAULT_DATA_ROOT, ExperimentConfig


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Config file, then convenience flags, then --set overrides (last one wins)."""
    config = load_config(args.config)
    overrides: List[str] = []
    if getattr(args, "data_dir", None):
        overrides.append(f"dataset.root={Path(args.data_dir).as_posix()}")
    if getattr(args, "output", None):
        overrides.append(f"output_dir={Path(args.output).as_posix()}")
    for flag, key in (("prep", "preprocessing.method"), ("model", "model.type"),
                      ("epochs", "training.epochs"), ("batch_size", "training.batch_size")):
        value = getattr(args, flag, None)
        if value is not None:
            overrides.append(f"{key}={value}")
    overrides += getattr(args, "set", None) or []
    return apply_overrides(config, overrides) if overrides else config


def cmd_info(args: argparse.Namespace) -> int:
    from .utils.system import get_system_info, print_system_info
    print_system_info(get_system_info())
    return 0


def cmd_check_data(args: argparse.Namespace) -> int:
    from .utils.dataset_checker import inspect_ninapro_dataset
    config = _build_config(args)
    folder = config.dataset.train_path
    report = inspect_ninapro_dataset(folder, subjects=config.dataset.subjects)
    print(f"Dataset folder : {report['data_dir']}")
    print(f"  exists        : {report['exists']}")
    print(f"  subjects found: {report['subjects_found']}")
    print(f"  .mat files    : {report['total_files']}")
    print(f"  EMG key       : {'yes' if report['has_emg'] else 'no'}")
    if not report["exists"] or report["total_files"] == 0:
        print("\nNo .mat files found. Put NinaPro files (S<n>_*.mat) in that folder, or pass --data-dir.")
        return 1
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    create_default_configs(args.dir)
    print(f"Wrote starter configs to {Path(args.dir).resolve()}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .experiments.runner import run_single_experiment, save_result_json
    config = _build_config(args)
    result = run_single_experiment(
        config,
        save_model_flag=not args.no_save,
        skip_visualization=args.no_viz,
    )
    out = Path(config.output_dir) / f"result_{result.model_type}_{result.prep_method}.json"
    save_result_json(result, out)
    if result.status != "OK":
        print(f"\n[FAILED] {result.error}", file=sys.stderr)
        return 1
    print(f"\n[OK] Accuracy {result.metrics['accuracy'] * 100:.2f}%  |  result: {out}")
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    from .config import load_grid_config
    from .experiments.grid_search import run_grid_search_optimized
    config = _build_config(args)
    grid = load_grid_config(args.grid_config)
    if args.preps:
        grid.preps = args.preps
    if args.models:
        grid.models = args.models
    result = run_grid_search_optimized(
        config, grid, skip_visualization=args.no_viz,
        parallel=args.parallel, max_workers=args.workers,
    )
    failed = sum(r.status != "OK" for r in result.results)
    print(f"\n{len(result.results) - failed} OK, {failed} failed, {result.total_time:.1f}s")
    return 1 if result.results and failed == len(result.results) else 0


def cmd_augment(args: argparse.Namespace) -> int:
    from .augmentation.__main__ import main as augment_main
    augment_main(args.rest)
    return 0


def _add_config_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config (default: config/default.yaml if it exists)")
    p.add_argument("--data-dir", help=f"dataset root folder (default: {DEFAULT_DATA_ROOT})")
    p.add_argument("--output", help="output folder (default: from config, 'outputs')")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override any config value, e.g. --set training.epochs=5 (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semg", description="sEMG preprocessing and machine learning platform (NinaPro DB8)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sub.add_parser("info", help="show system and GPU information").set_defaults(func=cmd_info)

    p = sub.add_parser("check-data", help="verify the dataset folder before training")
    _add_config_flags(p)
    p.set_defaults(func=cmd_check_data)

    p = sub.add_parser("init-config", help="write starter YAML config files")
    p.add_argument("--dir", default="config", help="target folder (default: config)")
    p.set_defaults(func=cmd_init_config)

    p = sub.add_parser("train", help="run one preprocessing + model experiment")
    _add_config_flags(p)
    p.add_argument("--prep", help="preprocessing method (e.g. raw, butterworth, vmd, wavelet)")
    p.add_argument("--model", help="model type (e.g. svm, rf, cnn1d, lstm, transformer)")
    p.add_argument("--epochs", type=int, help="training epochs (deep models)")
    p.add_argument("--batch-size", type=int, dest="batch_size", help="batch size (deep models)")
    p.add_argument("--no-save", action="store_true", help="do not save the trained model")
    p.add_argument("--no-viz", action="store_true", help="skip plots")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("grid", help="run a grid search over preprocessing x models")
    _add_config_flags(p)
    p.add_argument("--grid-config", help="YAML with preps / models / exclude_pairs")
    p.add_argument("--preps", nargs="+", help="override preprocessing methods")
    p.add_argument("--models", nargs="+", help="override models")
    p.add_argument("--no-viz", action="store_true", help="skip plots")
    p.add_argument("--parallel", action="store_true", help="train models in parallel processes")
    p.add_argument("--workers", type=int, default=2, help="max parallel workers (default 2)")
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("augment", help="generate or augment sEMG windows (see semg augment -h)",
                       add_help=False)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_augment)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
