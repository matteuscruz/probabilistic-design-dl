from pathlib import Path

import numpy as np
import tensorflow as tf


def load_data(name, data_root="data"):
    data_dir = Path(data_root) / name
    _ensure_dataset_available(name=name, data_root=data_root)

    x_train = _load_images(data_dir / "x_train.npy")
    y_train = _load_labels(data_dir / "y_train.npy")
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=10)

    x_test = _load_images(data_dir / "x_test.npy")
    y_test = _load_labels(data_dir / "y_test.npy")
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh)


def _load_images(path):
    images = np.load(path)
    images = 1.0 - images / 255.0
    images = images.astype(np.float32)

    if images.ndim == 3:
        images = images[..., np.newaxis]

    return images


def _load_labels(path):
    labels = np.load(path)
    return labels.astype(np.int64)


def _ensure_dataset_available(name, data_root="data"):
    data_dir = Path(data_root) / name
    required_files = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]

    if all((data_dir / file_name).exists() for file_name in required_files):
        return

    if name == "MNIST":
        _download_mnist_npy(data_root=data_root)
        return

    if name == "MNIST_corrupted":
        _create_corrupted_mnist_npy(data_root=data_root)
        return

    raise FileNotFoundError(
        f"Dataset '{name}' not found at '{data_dir}'. "
        "Provide x_train/y_train/x_test/y_test .npy files in this folder."
    )


def _download_mnist_npy(data_root="data"):
    data_root_path = Path(data_root)
    mnist_dir = data_root_path / "MNIST"
    mnist_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    np.save(mnist_dir / "x_train.npy", x_train.astype(np.uint8))
    np.save(mnist_dir / "y_train.npy", y_train.astype(np.int64))
    np.save(mnist_dir / "x_test.npy", x_test.astype(np.uint8))
    np.save(mnist_dir / "y_test.npy", y_test.astype(np.int64))


def _create_corrupted_mnist_npy(data_root="data"):
    data_root_path = Path(data_root)
    corrupted_dir = data_root_path / "MNIST_corrupted"
    corrupted_dir.mkdir(parents=True, exist_ok=True)

    _download_mnist_npy(data_root=data_root)

    mnist_dir = data_root_path / "MNIST"
    x_train = np.load(mnist_dir / "x_train.npy")
    y_train = np.load(mnist_dir / "y_train.npy")
    x_test = np.load(mnist_dir / "x_test.npy")
    y_test = np.load(mnist_dir / "y_test.npy")

    rng = np.random.default_rng(42)
    train_noise = rng.normal(loc=0.0, scale=25.0, size=x_train.shape)
    test_noise = rng.normal(loc=0.0, scale=25.0, size=x_test.shape)

    x_train_corrupted = np.clip(
        x_train.astype(np.float32) + train_noise, 0, 255
    ).astype(np.uint8)
    x_test_corrupted = np.clip(x_test.astype(np.float32) + test_noise, 0, 255).astype(
        np.uint8
    )

    np.save(corrupted_dir / "x_train.npy", x_train_corrupted)
    np.save(corrupted_dir / "y_train.npy", y_train.astype(np.int64))
    np.save(corrupted_dir / "x_test.npy", x_test_corrupted)
    np.save(corrupted_dir / "y_test.npy", y_test.astype(np.int64))
