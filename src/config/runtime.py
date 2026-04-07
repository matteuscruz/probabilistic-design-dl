from dataclasses import dataclass
from dataclasses import field
import copy


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class RuntimeConfig:
    data: dict = field(default_factory=lambda: {"dataset": "iris", "features": [0, 1]})
    split: dict = field(default_factory=lambda: {"test_size": 0.2, "random_state": None, "stratify": True})
    train: dict = field(default_factory=lambda: {"epochs": 500, "learning_rate": 0.01})
    eval: dict = field(default_factory=lambda: {"metric": "accuracy"})


DEFAULT_CONFIG = {
    "data": {"dataset": "iris", "features": [0, 1]},
    "split": {"test_size": 0.2, "random_state": None, "stratify": True},
    "train": {"epochs": 500, "learning_rate": 0.01},
    "eval": {"metric": "accuracy"},
}


def load_config(config_path=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return RuntimeConfig(**cfg)

    override = _load_yaml_or_json(config_path)
    merged = _deep_merge(cfg, override)
    return RuntimeConfig(**merged)


def _load_yaml_or_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        content = stream.read()

    try:
        import yaml

        data = yaml.safe_load(content)
        return data or {}
    except ImportError:
        import json

        return json.loads(content)
