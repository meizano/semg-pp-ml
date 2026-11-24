"""
Backward-compatible wrapper. Prefer the ``semg`` command (``pip install -e .``).

    python run_cli.py --mode info
    python run_cli.py --mode single --prep wavelet --model cnn1d --epochs 20
    python run_cli.py --mode grid
"""
from __future__ import annotations

import sys

from semg_pp_ml.cli import main

_MODE_TO_COMMAND = {"info": "info", "single": "train", "grid": "grid"}


def translate(argv: list[str]) -> list[str]:
    """Turn the old ``--mode X`` form into the new subcommand form."""
    if "--mode" not in argv:
        return argv
    i = argv.index("--mode")
    mode = argv[i + 1]
    if mode not in _MODE_TO_COMMAND:
        raise SystemExit(f"unknown --mode {mode!r}; use one of {sorted(_MODE_TO_COMMAND)}")
    rest = argv[:i] + argv[i + 2:]
    return [_MODE_TO_COMMAND[mode], *rest]


if __name__ == "__main__":
    raise SystemExit(main(translate(sys.argv[1:])))
