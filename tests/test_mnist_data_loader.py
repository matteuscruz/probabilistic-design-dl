from pathlib import Path

import numpy as np
import pytest


def _write_dataset(root: Path, name: str, n_train: int = 12, n_test: int = 6):
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    x_train = rng.integers(0, 256, size=(n_train, 28, 28), dtype=np.uint8)
    y_train = rng.integers(0, 10, size=(n_train,), dtype=np.int64)
    x_test = rng.integers(0, 256, size=(n_test, 28, 28), dtype=np.uint8)
    y_test = rng.integers(0, 10, size=(n_test,), dtype=np.int64)

    np.save(dataset_dir / "x_train.npy", x_train)
    np.save(dataset_dir / "y_train.npy", y_train)
    np.save(dataset_dir / "x_test.npy", x_test)
    np.save(dataset_dir / "y_test.npy", y_test)


def test_load_data_shapes_and_types(tmp_path):
    pytest.importorskip("tensorflow")
    from src.data.mnist import load_data

    _write_dataset(tmp_path, "MNIST")

    (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh) = load_data(
        "MNIST",
        data_root=tmp_path,
    )

    assert x_train.shape == (12, 28, 28, 1)
    assert x_test.shape == (6, 28, 28, 1)
    assert y_train.shape == (12,)
    assert y_test.shape == (6,)
    assert y_train_oh.shape == (12, 10)
    assert y_test_oh.shape == (6, 10)
    assert x_train.dtype == np.float32
    assert np.all((x_train >= 0.0) & (x_train <= 1.0))
