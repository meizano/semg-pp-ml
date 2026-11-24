"""Command line: augment an existing window dataset or generate an artificial one.

Examples:
    python -m semg_pp_ml.augmentation --synthetic --n-classes 6 --per-class 200 --out artificial.npz
    python -m semg_pp_ml.augmentation --input windows.npz --copies 2 --out augmented.npz

Input/output files are .npz with arrays ``X`` (n_windows, n_samples, n_channels) and ``y``.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import SyntheticConfig, augment_dataset, generate_synthetic_semg
from .transforms import TRANSFORMS


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="semg augment", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true", help="generate artificial data instead of augmenting")
    p.add_argument("--n-classes", type=int, default=6)
    p.add_argument("--per-class", type=int, default=200)
    p.add_argument("--fs", type=float, default=2000.0)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--input", help="input .npz with X and y (augmentation mode)")
    p.add_argument("--copies", type=int, default=1, help="augmented copies per window")
    p.add_argument("--ops", nargs="+", default=None, choices=sorted(TRANSFORMS),
                   help="augmentations to apply (default: noise amplitude time_warp electrode_shift crosstalk)")
    p.add_argument("--no-mixup", action="store_true")
    args = p.parse_args(argv)

    if args.synthetic:
        cfg = SyntheticConfig(fs=args.fs, n_channels=args.channels, seed=args.seed)
        X, y, _ = generate_synthetic_semg(args.n_classes, args.per_class, cfg)
    else:
        if not args.input:
            p.error("--input is required unless --synthetic is given")
        with np.load(args.input) as data:
            X, y = data["X"], data["y"]
        kwargs = {"n_copies": args.copies, "fs": args.fs, "mixup": not args.no_mixup,
                  "rng": np.random.default_rng(args.seed)}
        if args.ops:
            kwargs["ops"] = tuple(args.ops)
        X, y = augment_dataset(X, y, **kwargs)

    np.savez_compressed(args.out, X=X, y=y)
    print(f"wrote {args.out}: X{X.shape} {X.dtype}, classes={np.unique(y).size}")


if __name__ == "__main__":
    main()
