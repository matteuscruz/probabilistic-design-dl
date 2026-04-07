from dataclasses import asdict
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
import csv
import json


class ExperimentManager:
    def __init__(self, enabled=False, base_dir="artifacts", experiment_dir=None):
        self.enabled = bool(enabled)
        self.base_dir = Path(base_dir)
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None

        if self.enabled and self.experiment_dir is not None:
            self.model_dir = self.experiment_dir / "model"
            self.history_dir = self.experiment_dir / "history"
            self.figures_dir = self.experiment_dir / "figures"
        else:
            self.model_dir = None
            self.history_dir = None
            self.figures_dir = None

    @classmethod
    def create(cls, config, model_name, config_path=None):
        artifacts_cfg = getattr(config, "artifacts", {}) or {}
        enabled = bool(artifacts_cfg.get("enabled", True))
        base_dir = Path(artifacts_cfg.get("base_dir", "artifacts"))

        if not enabled:
            return cls(enabled=False, base_dir=base_dir)

        base_dir.mkdir(parents=True, exist_ok=True)
        experiment_index = _next_experiment_index(base_dir)
        experiment_dir = base_dir / f"exp{experiment_index}"

        manager = cls(enabled=True, base_dir=base_dir, experiment_dir=experiment_dir)
        manager.model_dir.mkdir(parents=True, exist_ok=False)
        manager.history_dir.mkdir(parents=True, exist_ok=False)
        manager.figures_dir.mkdir(parents=True, exist_ok=False)

        manager.save_json(
            "history",
            "metadata.json",
            {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "model": model_name,
                "config_path": config_path,
                "config": _config_to_dict(config),
            },
        )
        return manager

    def save_json(self, section, filename, data):
        if not self.enabled:
            return None
        output_path = self._section_dir(section) / filename
        with open(output_path, "w", encoding="utf-8") as file_stream:
            json.dump(data, file_stream, indent=2)
        return output_path

    def save_csv(self, section, filename, headers, rows):
        if not self.enabled:
            return None
        output_path = self._section_dir(section) / filename
        with open(output_path, "w", encoding="utf-8", newline="") as file_stream:
            writer = csv.writer(file_stream)
            writer.writerow(headers)
            writer.writerows(rows)
        return output_path

    def save_npz_model(self, filename, **arrays):
        if not self.enabled:
            return None
        output_path = self.model_dir / filename
        import numpy as np

        np.savez(output_path, **arrays)
        return output_path

    def model_path(self, filename):
        if not self.enabled:
            return None
        return self.model_dir / filename

    def figure_path(self, filename):
        if not self.enabled:
            return None
        return self.figures_dir / filename

    def _section_dir(self, section):
        if section == "model":
            return self.model_dir
        if section == "history":
            return self.history_dir
        if section == "figures":
            return self.figures_dir
        raise ValueError(f"Unknown section: {section}")


def _next_experiment_index(base_dir):
    max_index = -1
    for child in base_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("exp"):
            continue
        suffix = child.name[3:]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _config_to_dict(config):
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return dict(config)
