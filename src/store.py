"""Model + config persistence helpers."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import yaml

from .model import StylometryModel


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def save_model(model: StylometryModel, model_path: str, profile_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    Path(profile_path).write_text(json.dumps(model.summary(), indent=2))


def load_model(model_path: str) -> StylometryModel:
    return joblib.load(model_path)
