"""End-to-end: single run, grid search, failure isolation and the one-load-per-run guarantee."""
import pytest

import semg_pp_ml.preprocessing.pipeline as pipeline_mod
from semg_pp_ml.config import ExperimentConfig, GridSearchConfig, apply_overrides
from semg_pp_ml.experiments import run_grid_search_optimized, run_single_experiment


@pytest.fixture
def cfg(fake_dataset_root, tmp_path):
    base = ExperimentConfig()
    base.dataset.root = fake_dataset_root
    base.dataset.subjects = [1, 2]
    base.dataset.eval_dir = "no_eval_here"        # forces the stratified split path
    base.output_dir = tmp_path / "out"
    return base


def test_single_run_svm_end_to_end(cfg):
    result = run_single_experiment(cfg, prep_method="raw", model_type="svm", skip_visualization=False)
    assert result.status == "OK", result.error
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert result.metrics["prep_method"] == "raw"
    assert (cfg.output_dir / "visualizations" / "svm_raw").exists()
    assert result.model_path and (cfg.output_dir / "model_svm_raw.pkl").exists()


def test_each_run_reads_raw_mat_files_once(cfg, monkeypatch):
    calls = []
    original = pipeline_mod.load_mat_files_from_folder

    def counting(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "load_mat_files_from_folder", counting)
    cfg.output_dir = cfg.output_dir / "fresh"           # empty cache -> full pipeline
    result = run_single_experiment(cfg, prep_method="butterworth", model_type="svm",
                                   skip_visualization=True, save_model_flag=False)
    assert result.status == "OK", result.error
    assert len(calls) == 1, f"training folder read {len(calls)} times, expected 1"


def test_missing_data_is_a_failed_result_not_an_exception(cfg):
    bad = apply_overrides(cfg, [f"dataset.root={cfg.output_dir}"])   # no .mat files there
    result = run_single_experiment(bad, prep_method="raw", model_type="svm", skip_visualization=True)
    assert result.status == "FAILED" and "not found" in result.error.lower()


def test_grid_search_exclusions_and_failure_isolation(cfg):
    grid = GridSearchConfig(
        preps=["raw", "butterworth"],
        models=["svm", "knn", "no_such_model"],
        exclude_pairs=[["raw", "svm"], ["raw", "knn"]],
    )
    result = run_grid_search_optimized(cfg, grid, skip_visualization=True)
    pairs = sorted((r.prep_method, r.model_type, r.status) for r in result.results)
    assert ("raw", "no_such_model", "FAILED") in pairs
    assert ("butterworth", "svm", "OK") in pairs and ("butterworth", "knn", "OK") in pairs
    assert not any(p == "raw" and m in ("svm", "knn") for p, m, _ in pairs)
    assert (cfg.output_dir / "grid_search_summary.csv").exists()
    assert "accuracy" in result.best_by_metric
