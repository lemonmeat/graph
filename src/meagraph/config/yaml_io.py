"""YAML <-> typed config models."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def load_yaml(path: str | Path, model: type[M]) -> M:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return model.model_validate(data)


def save_yaml(config: BaseModel, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    return path
