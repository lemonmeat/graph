"""From detection results to burst-controlled connectivity graphs (DECISIONS.md D14).

Every estimate is made twice: on all spikes, and with network-burst periods removed. Edges
significant in both are *robust*: they are not explained by the shared firing in bursts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from meagraph.connectivity.base import ConnectivityResult, estimate
from meagraph.connectivity.graph import load_result, save_result
from meagraph.detect.bursts import BurstConfig, network_bursts
from meagraph.detect.store import load_detection
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
    robust: ConnectivityResult  # ``all`` with significant = significant in both, with the same sign
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
    return BurstControlled(full, part, replace(full, significant=robust_mask(full, part)), periods)


def robust_mask(full: ConnectivityResult, part: ConnectivityResult, full_sig=None, part_sig=None) -> np.ndarray:
    """Edges significant in both results with the same sign (for signed methods: same kind of edge)."""
    a = full.significant if full_sig is None else full_sig
    b = part.significant if part_sig is None else part_sig
    if not full.signed:
        return a & b
    with np.errstate(invalid="ignore"):
        return a & b & (np.sign(full.weights) == np.sign(part.weights))


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
    base = analysed_trains(detection, cfg)
    periods = network_bursts(base.select(usable), cfg.bursts)
    return {m: burst_controlled(base.select(chosen), m, bursts=periods) for m in methods}


def analysed_trains(detection: DetectionResult, config: GraphConfig) -> SpikeTrains:
    """The detection's spikes with stimulation periods removed, as the graphs see them."""
    if detection.stim and config.exclude_stim_post_ms > 0:
        periods = stimulation_periods(detection.stim, config.exclude_stim_pre_ms, config.exclude_stim_post_ms)
        return detection.trains.without_periods(periods)
    return detection.trains


@dataclass(frozen=True, eq=False)
class SavedGraph:
    result: ConnectivityResult
    trains: SpikeTrains  # exactly the spikes the result was tested on: its nodes, same order
    detection: DetectionResult


def load_graph(folder: str | Path) -> SavedGraph:
    """A saved graph with the spikes it was tested on, e.g. to redraw its correlograms.

    ``folder`` is ``results/<recording>/graph_<method>/<all|no_bursts|robust>``. The detection
    folder is found by name next to ``graph_<method>``, so results can be moved as a whole.
    """
    folder = Path(folder)
    meta = yaml.safe_load((folder / "config.yaml").read_text())
    if not meta.get("pipeline"):
        raise ValueError(f"{folder} has no pipeline settings (written by an older meagraph); re-run `meagraph graph`")
    prov = json.loads((folder / "provenance.json").read_text())
    detection = load_detection(folder.parent.parent / Path(prov["inputs"][0]["path"]).parent.name)
    trains = analysed_trains(detection, GraphConfig(**meta["pipeline"]))
    if folder.name == "no_bursts":
        trains = trains.without_periods(np.loadtxt(folder.parent / "network_bursts.csv", delimiter=",", skiprows=1, ndmin=2))
    result = load_result(folder)
    return SavedGraph(result, trains.select(list(result.node_ids)), detection)


def save_burst_controlled(bc: BurstControlled, folder: str | Path, config: GraphConfig | None = None,
                          inputs=(), overwrite: bool = False) -> Path:  # fmt: skip
    """``folder/{all,no_bursts,robust}/`` (see :mod:`meagraph.connectivity.graph`) and ``network_bursts.csv``."""
    folder = Path(folder)
    for name in ("all", "no_bursts", "robust"):
        save_result(getattr(bc, name), folder / name, inputs=inputs, overwrite=overwrite, pipeline=config)
    np.savetxt(folder / "network_bursts.csv", bc.bursts, delimiter=",", header="start_s,stop_s", comments="", fmt="%.4f")
    return folder
