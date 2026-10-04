"""From detection results to burst-controlled connectivity graphs (DECISIONS.md D14).

Every estimate is made twice: on all spikes, and with network-burst periods removed. Edges
significant in both are *robust*: they are not explained by the shared firing in bursts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from meagraph.connectivity.base import ConnectivityResult, estimate
from meagraph.connectivity.graph import save_result
from meagraph.detect.bursts import BurstConfig, network_bursts, remove_periods
from meagraph.probe.spec import ProbeSpec, load_probe_spec
from meagraph.spiketrains import SpikeTrains


@dataclass(frozen=True, eq=False)
class BurstControlled:
    all: ConnectivityResult
    no_bursts: ConnectivityResult
    robust: ConnectivityResult  # ``all`` with significant = significant in both
    bursts: np.ndarray  # (k, 2) periods removed for ``no_bursts``


def burst_controlled(
    trains: SpikeTrains,
    method: str,
    config: BaseModel | dict | None = None,
    bursts: np.ndarray | None = None,
    burst_config: BurstConfig | None = None,
) -> BurstControlled:
    """``bursts`` defaults to network bursts detected on ``trains`` itself."""
    periods = network_bursts(trains, burst_config) if bursts is None else bursts
    full = estimate(trains, method, config)
    part = estimate(remove_periods(trains, periods), method, config)
    robust = replace(full, significant=full.significant & part.significant)
    return BurstControlled(full, part, robust, periods)


def positions_for(channel_ids, spec: ProbeSpec | str) -> np.ndarray:
    spec = load_probe_spec(spec) if isinstance(spec, str) else spec
    index = {c: k for k, c in enumerate(spec.labels)}
    missing = [c for c in channel_ids if c not in index]
    if missing:
        raise KeyError(f"channels {missing} are not on probe {spec.name!r}")
    pos = spec.positions_um()[[index[c] for c in channel_ids]]
    return pos if pos.shape[1] == 3 else np.column_stack([pos, np.zeros(len(pos))])


def stimulation_periods(stim, pre_ms: float = 1.0, post_ms: float = 200.0) -> np.ndarray:
    """``[onset - pre, offset + post]`` around every pulse of every StimEvents, merged.

    Spontaneous-activity connectivity must not include stimulus-locked co-firing: shared drive
    makes unconnected pairs correlate (benchmark scenario "stim null")."""
    rows = []
    for s in stim:
        off = s.offsets_s if s.offsets_s is not None else s.onsets_s
        rows.append(np.column_stack([s.onsets_s - pre_ms / 1e3, off + post_ms / 1e3]))
    if not rows:
        return np.zeros((0, 2))
    p = np.concatenate(rows)
    p = p[np.argsort(p[:, 0])]
    merged = [list(p[0])]
    for lo, hi in p[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return np.array(merged)


def graphs_from_detection(detection, methods=("cch_jitter", "cch_hollow", "sttc"), probe="cube4x4x4_E-00303",
                          channels: str = "active", burst_config: BurstConfig | None = None,
                          exclude_periods: np.ndarray | None = None) -> dict[str, BurstControlled]:  # fmt: skip
    """Connectivity among a detection result's channels (``active``: QC-active only, D10).

    Spikes inside ``exclude_periods`` (e.g. :func:`stimulation_periods`) are dropped first. Network
    bursts are detected on all non-excluded channels, since bursts are array-wide events.
    """
    usable = [c for c in detection.trains.unit_ids if c not in detection.excluded]
    chosen = list(detection.active_channels) if channels == "active" else usable
    base = detection.trains if exclude_periods is None else remove_periods(detection.trains, exclude_periods)
    trains = base.select(chosen).with_positions(positions_for(chosen, probe))
    periods = network_bursts(base.select(usable), burst_config)
    return {m: burst_controlled(trains, m, bursts=periods) for m in methods}


def save_burst_controlled(bc: BurstControlled, folder: str | Path, inputs=(), overwrite: bool = False) -> Path:
    folder = Path(folder)
    for name in ("all", "no_bursts", "robust"):
        save_result(getattr(bc, name), folder / name, inputs=inputs, overwrite=overwrite)
    np.savetxt(folder / "network_bursts.csv", bc.bursts, delimiter=",", header="start_s,stop_s", comments="", fmt="%.4f")
    return folder
