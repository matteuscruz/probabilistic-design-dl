from pathlib import Path

import pytest
import yaml

from main import run


def _write_naive_config(path: Path, artifacts_dir: Path):
    cfg = {
        "model": {"name": "naive_bayes"},
        "data": {
            "dataset": "iris",
            "features": [0, 1],
            "root": "data",
            "mnist_name": "MNIST",
            "mnist_corrupted_name": "MNIST_corrupted",
        },
        "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
        "train": {
            "epochs": 10,
            "learning_rate": 0.01,
            "seed": 42,
            "verbose": 0,
        },
        "artifacts": {
            "enabled": True,
            "base_dir": str(artifacts_dir),
            "save_model": True,
            "save_history": True,
            "save_figures": True,
            "naive_binary_epochs": 2,
        },
        "eval": {"metric": "accuracy"},
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_naive_bayes_creates_experiment_artifacts(tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_probability")

    cfg_path = tmp_path / "naive.yaml"
    artifacts_dir = tmp_path / "artifacts"
    _write_naive_config(cfg_path, artifacts_dir)

    score = run(str(cfg_path))

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    experiment_dirs = sorted(
        path for path in artifacts_dir.glob("exp*") if path.is_dir()
    )
    assert len(experiment_dirs) == 1

    experiment_dir = experiment_dirs[0]
    assert (experiment_dir / "model" / "naive_bayes.npz").exists()
    assert (experiment_dir / "history" / "metadata.json").exists()
    assert (experiment_dir / "history" / "metrics.csv").exists()
    assert (experiment_dir / "history" / "predictions_multiclass.csv").exists()
    assert (experiment_dir / "history" / "binary_nlls.csv").exists()
    assert (experiment_dir / "history" / "binary_scales.csv").exists()
    assert (experiment_dir / "history" / "predictions_binary.csv").exists()

    expected_figures = [
        "01_scatter_training.png",
        "02_class_conditionals_contours.png",
        "03_decision_regions.png",
        "04_binary_loss_scales.png",
        "05_logistic_regression_contours.png",
    ]
    for figure_name in expected_figures:
        assert (experiment_dir / "figures" / figure_name).exists()


def test_experiment_index_increments(tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_probability")

    cfg_path = tmp_path / "naive.yaml"
    artifacts_dir = tmp_path / "artifacts"
    _write_naive_config(cfg_path, artifacts_dir)

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    config["artifacts"]["save_figures"] = False
    config["artifacts"]["naive_binary_epochs"] = 1
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run(str(cfg_path))
    run(str(cfg_path))

    experiment_dirs = sorted(
        path.name for path in artifacts_dir.glob("exp*") if path.is_dir()
    )
    assert experiment_dirs == ["exp0", "exp1"]
