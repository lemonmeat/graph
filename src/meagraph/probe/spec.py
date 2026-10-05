"""Probe specifications as data: a YAML file of physical parameters plus a CSV grid map.

Package specs live in ``meagraph/probe/data``. A spec can also be loaded from any YAML path,
so new arrays need a file, not code.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, model_validator

_DATA = files("meagraph.probe") / "data"

# The probe used when none is named: the custom 4x4x4 array. Other arrays are chosen by name.
DEFAULT_PROBE = "cube4x4x4_E-00303"


class GridContact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    row: PositiveInt
    col: PositiveInt
    layer: PositiveInt = 1


class ProbeSpec(BaseModel):
    """Electrode geometry. Positions follow from grid indices and pitches (see the YAML headers)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    ndim: Literal[2, 3]
    pitch_row_um: PositiveFloat
    pitch_col_um: PositiveFloat
    layer_spacing_um: PositiveFloat | None = None
    contact_radius_um: PositiveFloat
    reference_labels: tuple[str, ...] = ()
    geometry_verified: bool = False
    notes: str = ""
    contacts: tuple[GridContact, ...]

    @model_validator(mode="after")
    def _check(self) -> ProbeSpec:
        labels = [c.label for c in self.contacts]
        if len(set(labels)) != len(labels):
            raise ValueError(f"{self.name}: duplicate contact labels")
        cells = [(c.row, c.col, c.layer) for c in self.contacts]
        if len(set(cells)) != len(cells):
            raise ValueError(f"{self.name}: two contacts share a grid cell")
        if set(self.reference_labels) & set(labels):
            raise ValueError(f"{self.name}: reference labels must not also be contacts")
        if self.ndim == 3 and self.layer_spacing_um is None:
            raise ValueError(f"{self.name}: a 3D probe needs layer_spacing_um")
        if self.ndim == 2 and any(c.layer != 1 for c in self.contacts):
            raise ValueError(f"{self.name}: a 2D probe has a single layer")
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(c.label for c in self.contacts)

    def grid_indices(self) -> np.ndarray:
        """(n_contacts, 3) array of (row, col, layer)."""
        return np.array([(c.row, c.col, c.layer) for c in self.contacts], dtype=np.int64)

    def positions_um(self) -> np.ndarray:
        """(n_contacts, ndim) positions: x from col, y from row, z from layer."""
        g = self.grid_indices().astype(np.float64) - 1.0
        xy = np.column_stack([g[:, 1] * self.pitch_col_um, g[:, 0] * self.pitch_row_um])
        if self.ndim == 2:
            return xy
        return np.column_stack([xy, g[:, 2] * self.layer_spacing_um])

    def positions_xyz_um(self, labels: Sequence[str] | None = None) -> np.ndarray:
        """(n, 3) positions of ``labels`` (default: every contact, in spec order); planar probes get z = 0."""
        pos = self.positions_um()
        if pos.shape[1] == 2:
            pos = np.column_stack([pos, np.zeros(len(pos))])
        if labels is None:
            return pos
        index = {c: k for k, c in enumerate(self.labels)}
        missing = [c for c in labels if c not in index]
        if missing:
            raise KeyError(f"channels {missing} are not on probe {self.name!r}")
        return pos[[index[c] for c in labels]]


def available_probe_specs() -> list[str]:
    return sorted(p.name[: -len(".yaml")] for p in _DATA.iterdir() if p.name.endswith(".yaml"))


def load_probe_spec(name_or_path: str | Path) -> ProbeSpec:
    """Load a packaged spec by name (see :func:`available_probe_specs`) or any spec YAML by path."""
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.exists():
        text, map_dir = path.read_text(), path.parent
        read_map = lambda name: (map_dir / name).read_text()  # noqa: E731
    else:
        resource = _DATA / f"{name_or_path}.yaml"
        if not resource.is_file():
            raise KeyError(f"unknown probe spec {name_or_path!r}; available: {available_probe_specs()}")
        text = resource.read_text()
        read_map = lambda name: (_DATA / name).read_text()  # noqa: E731
    raw = yaml.safe_load(text)
    map_file = raw.pop("map_file")
    rows = csv.DictReader(read_map(map_file).splitlines())
    raw["contacts"] = [GridContact(**r) for r in rows]
    return ProbeSpec(**raw)
