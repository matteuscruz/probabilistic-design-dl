from pathlib import Path

import numpy as np
import tensorflow as tf


def load_data(name, data_root="data"):
    data_dir = Path(data_root) / name
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
