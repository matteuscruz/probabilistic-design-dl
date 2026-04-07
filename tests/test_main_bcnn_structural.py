from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def mnist_data_root(tmp_path):
    def write_dataset(name: str, seed: int):
        dataset_dir = tmp_path / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)
        x_train = rng.integers(0, 256, size=(24, 28, 28), dtype=np.uint8)
        y_train = rng.integers(0, 10, size=(24,), dtype=np.int64)
        x_test = rng.integers(0, 256, size=(10, 28, 28), dtype=np.uint8)
        y_test = rng.integers(0, 10, size=(10,), dtype=np.int64)

        np.save(dataset_dir / "x_train.npy", x_train)
        np.save(dataset_dir / "y_train.npy", y_train)
        np.save(dataset_dir / "x_test.npy", x_test)
        np.save(dataset_dir / "y_test.npy", y_test)

    write_dataset("MNIST", seed=123)
    write_dataset("MNIST_corrupted", seed=456)
    return tmp_path


def _write_config(path: Path, model_name: str, data_root: Path):
    cfg = {
        "model": {"name": model_name},
        "data": {
            "dataset": "mnist",
            "root": str(data_root),
            "mnist_name": "MNIST",
            "mnist_corrupted_name": "MNIST_corrupted",
        },
        "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
        "train": {
            "epochs": 1,
            "learning_rate": 0.001,
            "seed": 42,
            "verbose": 0,
        },
        "artifacts": {
            "enabled": True,
            "base_dir": str(path.parent / "artifacts"),
            "save_model": True,
            "save_history": True,
            "save_figures": True,
            "naive_binary_epochs": 2,
            "cnn_bayesian_ensemble_size": 5,
        },
        "eval": {"metric": "accuracy"},
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


@pytest.mark.parametrize(
    "model_name",
    ["cnn_deterministic", "cnn_probabilistic", "bayesian_cnn"],
)
def test_main_runs_all_cnn_model_types(model_name, mnist_data_root, tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_probability")
    from main import run

    cfg_path = tmp_path / f"{model_name}.yaml"
    _write_config(cfg_path, model_name, mnist_data_root)

    result = run(str(cfg_path))

    assert isinstance(result, dict)
    assert result["model"] == model_name
    assert "accuracy_test" in result
    assert "accuracy_corrupted" in result
    assert 0.0 <= result["accuracy_test"] <= 1.0
    assert 0.0 <= result["accuracy_corrupted"] <= 1.0

    artifacts_root = tmp_path / "artifacts"
    experiment_dirs = sorted(
        path for path in artifacts_root.glob("exp*") if path.is_dir()
    )
    assert len(experiment_dirs) == 1

    experiment_dir = experiment_dirs[0]
    assert (experiment_dir / "model").is_dir()
    assert (experiment_dir / "history").is_dir()
    assert (experiment_dir / "figures").is_dir()

    assert any((experiment_dir / "model").glob("*.h5"))
    assert (experiment_dir / "history" / "training_history.csv").exists()
    assert (experiment_dir / "history" / "metrics.csv").exists()
    assert (experiment_dir / "history" / "entropy_summary.json").exists()
    assert (experiment_dir / "figures" / "training_history.png").exists()
    assert (experiment_dir / "figures" / "02_mnist_samples.png").exists()
    assert (experiment_dir / "figures" / "03_mnist_corrupted_samples.png").exists()
    assert (experiment_dir / "figures" / "07_entropy_mnist.png").exists()
    assert (experiment_dir / "figures" / "08_entropy_mnist_c.png").exists()
