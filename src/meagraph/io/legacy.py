"""Reader for ``<recording>.spikes.h5`` sidecars written by the legacy ``spikes.py``.

Legacy timestamps are seconds from the first analog sample. They are shifted by the stored
``t_start`` onto the recording clock (DECISIONS.md D5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from meagraph.spiketrains import SpikeTrains

SIDECAR_SUFFIX = ".spikes.h5"


def sidecar_path(recording_path: str | Path) -> Path:
    p = Path(recording_path)
    return p.with_name(p.name[: -len(p.suffix)] + SIDECAR_SUFFIX)


@dataclass(frozen=True, eq=False)
class LegacySpikes:
    trains: SpikeTrains
    amplitudes_uv: dict[str, np.ndarray]
    waveforms_uv: dict[str, np.ndarray]  # (n_samples, n_spikes), as written by spikes.py
    sigma_uv: dict[str, float]
    cutout_ms: tuple[float, float]  # (ms before, ms after) the detection sample
    params: dict
    source: str


def read_spikes_sidecar(path: str | Path) -> LegacySpikes:
    """Read a sidecar by its own path (see :func:`sidecar_path`)."""
    with h5py.File(path, "r") as f:
        params = json.loads(f.attrs["params"])
        source = str(f.attrs.get("source", ""))
        times, amps, wfs, sigmas = {}, {}, {}, {}
        for name, grp in f["SpikeStream"].items():
            electrode = name.split("_", 1)[1]
            times[electrode] = grp["ts"][:] + params["t_start"]
            amps[electrode] = grp["amp"][:]
            wfs[electrode] = grp["waveforms"][:]
            sigmas[electrode] = float(grp.attrs["sigma_uv"])
    t_start = float(params["t_start"])
    trains = SpikeTrains.from_dict(times, t_start, t_start + float(params["duration"]))
    pre, post = params["cutout_ms"]
    return LegacySpikes(trains, amps, wfs, sigmas, (float(pre), float(post)), params, source)
