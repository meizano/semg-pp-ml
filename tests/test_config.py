"""Config behaviour: path resolution, overrides, strictness and validation."""
from pathlib import Path

import pytest
import yaml

from semg_pp_ml.config import (
    ExperimentConfig, apply_overrides, load_config, load_grid_config, save_config,
)
from semg_pp_ml.config.models import DEFAULT_DATA_ROOT


def test_default_config_file_loads():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    assert cfg.dataset.root.name == "ninapro_db8"


def test_load_config_missing_explicit_path_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_unknown_yaml_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("training:\n  epohcs: 3\n", encoding="utf-8")
    with pytest.raises(Exception):   # pydantic ValidationError (extra="forbid")
        load_config(p)


def test_assignment_is_validated_and_coerced():
    cfg = ExperimentConfig()
    cfg.output_dir = "results/x"                 # string -> Path (used to break later)
    assert isinstance(cfg.output_dir, Path) and cfg.output_dir.is_absolute()
    with pytest.raises(Exception):
        cfg.training.epochs = "many"


def test_train_path_is_relative_to_root(tmp_path):
    cfg = ExperimentConfig()
    cfg.dataset.root = tmp_path
    assert cfg.dataset.train_path == (tmp_path / "train").resolve()
    cfg.dataset.train_dir = str(tmp_path / "elsewhere")
    assert cfg.dataset.train_path == tmp_path / "elsewhere"


def test_train_path_falls_back_to_root_with_mat_files(fake_dataset_root):
    cfg = ExperimentConfig()
    cfg.dataset.root = fake_dataset_root / "train"   # user points straight at the files
    assert cfg.dataset.train_path == fake_dataset_root / "train"


def test_default_root_constant():
    assert ExperimentConfig().dataset.root == DEFAULT_DATA_ROOT.resolve()


def test_apply_overrides_types_and_nesting():
    cfg = apply_overrides(ExperimentConfig(), [
        "training.epochs=5", "training.lr=1e-3", "preprocessing.method=vmd",
        "model.type=svm", "model.svm.C=3", "model.tcn_dilations=[1, 2]", "tags.owner=lab",
    ])
    assert cfg.training.epochs == 5 and cfg.training.lr == pytest.approx(1e-3)
    assert cfg.preprocessing.method == "vmd" and cfg.model.type == "svm"
    assert cfg.model.svm["C"] == 3 and cfg.model.tcn_dilations == [1, 2]
    assert cfg.tags == {"owner": "lab"}


@pytest.mark.parametrize("bad", ["training.epohcs=5", "nosuch.x=1", "noequals", "=3"])
def test_bad_overrides_fail_loudly(bad):
    with pytest.raises((KeyError, ValueError)):
        apply_overrides(ExperimentConfig(), [bad])


def test_override_with_invalid_value_fails_validation():
    with pytest.raises(Exception):
        apply_overrides(ExperimentConfig(), ["preprocessing.method=not_a_method"])


def test_save_and_reload_round_trip(tmp_path):
    cfg = apply_overrides(ExperimentConfig(), ["training.epochs=7"])
    save_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml").training.epochs == 7
    assert yaml.safe_load((tmp_path / "c.yaml").read_text())["training"]["epochs"] == 7


def test_grid_config_loader(tmp_path):
    assert load_grid_config(None).preps        # defaults
    p = tmp_path / "g.yaml"
    p.write_text("preps: [raw]\nmodels: [svm]\nexclude_pairs: []\n", encoding="utf-8")
    g = load_grid_config(p)
    assert g.preps == ["raw"] and g.models == ["svm"]
