"""From detection results to burst-controlled connectivity graphs (DECISIONS.md D14).

Every estimate is made twice: on all spikes, and with network-burst periods removed. Edges
significant in both are *robust*: they are not explained by the shared firing in bursts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from meagraph.connectivity.base import ConnectivityResult, estimate
from meagraph.connectivity.graph import save_result
from meagraph.detect.bursts import BurstConfig, network_bursts
from meagraph.detect.threshold import DetectionResult
from meagraph.spiketrains import SpikeTrains
from meagraph.stimulation.artifacts import stimulation_periods


class GraphConfig(BaseModel):
    """How a detection result becomes graphs. Saved in every graph folder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channels: Literal["active", "all"] = "active"  # QC-active channels only (D10, D14)
    exclude_stim_pre_ms: NonNegativeFloat = 1.0  # stimulation periods left out (D18); post 0 keeps them
    exclude_stim_post_ms: NonNegativeFloat = 200.0
    bursts: BurstConfig = BurstConfig()


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
    part = estimate(trains.without_periods(periods), method, config)
    robust = replace(full, significant=full.significant & part.significant)
    return BurstControlled(full, part, robust, periods)


def graphs_from_detection(
    detection: DetectionResult,
    methods: Sequence[str] = ("cch_jitter", "cch_hollow", "sttc"),
    config: GraphConfig | None = None,
) -> dict[str, BurstControlled]:
    """Connectivity among a detection result's channels, one :class:`BurstControlled` per method.

    Spikes in stimulation periods (from ``detection.stim``) are dropped first. Network bursts are
    detected on every non-excluded channel, since bursts are array-wide events. Node positions
    are the electrode positions stored with the detection.
    """
    cfg = config or GraphConfig()
    usable = [c for c in detection.trains.unit_ids if c not in detection.excluded]
    chosen = list(detection.active_channels) if cfg.channels == "active" else usable
    base = detection.trains
    if detection.stim and cfg.exclude_stim_post_ms > 0:
        base = base.without_periods(stimulation_periods(detection.stim, cfg.exclude_stim_pre_ms, cfg.exclude_stim_post_ms))
    periods = network_bursts(base.select(usable), cfg.bursts)
    return {m: burst_controlled(base.select(chosen), m, bursts=periods) for m in methods}


def save_burst_controlled(bc: BurstControlled, folder: str | Path, config: GraphConfig | None = None,
                          inputs=(), overwrite: bool = False) -> Path:  # fmt: skip
    """``folder/{all,no_bursts,robust}/`` (see :mod:`meagraph.connectivity.graph`) and ``network_bursts.csv``."""
    folder = Path(folder)
    for name in ("all", "no_bursts", "robust"):
        save_result(getattr(bc, name), folder / name, inputs=inputs, overwrite=overwrite, pipeline=config)
    np.savetxt(folder / "network_bursts.csv", bc.bursts, delimiter=",", header="start_s,stop_s", comments="", fmt="%.4f")
    return folder
