from pathlib import Path

import numpy as np

from src.config.runtime import load_config
from src.data.loaders import load_iris_sepal_dataset
from src.data.split import train_test_split_dataset


def test_load_default_config():
    cfg = load_config()
    assert cfg.data["dataset"] == "iris"
    assert cfg.train["epochs"] > 0


def test_load_yaml_config_from_file():
    cfg_path = Path("config/default.yaml")
    cfg = load_config(str(cfg_path))
    assert cfg.split["test_size"] == 0.2


def test_data_loader_and_split_shapes():
    x, y = load_iris_sepal_dataset()
    assert x.shape[1] == 2
    assert x.shape[0] == y.shape[0]

    x_train, x_test, y_train, y_test = train_test_split_dataset(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=True,
    )
    assert x_train.shape[0] + x_test.shape[0] == x.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    assert set(np.unique(y_train)).issubset({0, 1, 2})
