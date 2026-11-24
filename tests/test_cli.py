"""Command line: subcommands, exit codes, legacy --mode translation, lazy imports."""
import os
import subprocess
import sys

import pytest

from semg_pp_ml.cli import main
from run_cli import translate


def test_check_data_reports_missing_and_present(fake_dataset_root, tmp_path, capsys):
    assert main(["check-data", "--data-dir", str(fake_dataset_root)]) == 0
    assert main(["check-data", "--data-dir", str(tmp_path / "empty")]) == 1
    assert "No .mat files" in capsys.readouterr().out


def test_train_command_with_overrides(fake_dataset_root, tmp_path, capsys):
    code = main([
        "train", "--data-dir", str(fake_dataset_root), "--prep", "raw", "--model", "svm",
        "--output", str(tmp_path / "out"), "--no-viz", "--no-save",
        "--set", "dataset.subjects=[1,2]", "--set", "dataset.eval_dir=none",
    ])
    assert code == 0
    assert "Accuracy" in capsys.readouterr().out


def test_bad_override_is_usage_error(capsys):
    assert main(["train", "--set", "training.epohcs=3"]) == 2
    assert "unknown config key" in capsys.readouterr().err


def test_init_config_writes_files(tmp_path):
    assert main(["init-config", "--dir", str(tmp_path / "cfg")]) == 0
    assert (tmp_path / "cfg" / "default.yaml").exists()


def test_legacy_mode_translation():
    assert translate(["--mode", "info"]) == ["info"]
    assert translate(["--mode", "single", "--prep", "vmd", "--epochs", "3"]) == \
        ["train", "--prep", "vmd", "--epochs", "3"]
    assert translate(["--mode", "grid"]) == ["grid"]
    with pytest.raises(SystemExit):
        translate(["--mode", "bogus"])


def test_cli_import_does_not_load_torch():
    """`semg info` / `--help` must start fast: torch is only needed by deep models."""
    code = "import sys, semg_pp_ml.cli; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": "src"}, check=True)
    assert out.stdout.strip() == "False"
