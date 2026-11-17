"""From detection results to burst-controlled connectivity graphs (DECISIONS.md D14).

Every estimate is made twice: on all spikes, and with network-burst periods removed. Edges
significant in both are *robust*: they are not explained by the shared firing in bursts.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from meagraph.connectivity.base import ConnectivityResult, Progress, estimate
from meagraph.connectivity.graph import load_result, save_result
from meagraph.connectivity.stim import StimConnectivityConfig
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


# ``progress(method, stage, done, total)`` for graph pipelines. ``stage`` is "all spikes" or
# "without bursts". Each estimate reports (0, 1) before and (1, 1) after; estimators with a long
# per-target loop (``glm``) also report each step in between, with total = number of targets.
GraphProgress = Callable[[str, str, int, int], None]


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
    progress: GraphProgress | None = None,
) -> BurstControlled:
    """``bursts`` defaults to network bursts detected on ``trains`` itself. See :data:`GraphProgress`."""
    periods = network_bursts(trains, burst_config) if bursts is None else bursts
    full = _estimate_reporting(trains, method, config, "all spikes", progress)
    part = _estimate_reporting(trains.without_periods(periods), method, config, "without bursts", progress)
    return BurstControlled(full, part, replace(full, significant=robust_mask(full, part)), periods)


def _estimate_reporting(trains, method, config, stage: str, progress: GraphProgress | None) -> ConnectivityResult:
    if progress is None:
        return estimate(trains, method, config)
    progress(method, stage, 0, 1)
    step: Progress = lambda done, total: progress(method, stage, done, total)  # noqa: E731
    result = estimate(trains, method, config, progress=step)
    progress(method, stage, 1, 1)
    return result


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
    progress: GraphProgress | None = None,
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
    return {m: burst_controlled(base.select(chosen), m, bursts=periods, progress=progress) for m in methods}


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
    # The input path may come from another OS (graphs computed on the lab PC store D:\...);
    # PureWindowsPath splits on both "\\" and "/", so it reads either.
    detection = load_detection(folder.parent.parent / PureWindowsPath(prov["inputs"][0]["path"]).parent.name)
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


# -- stimulus-evoked connectivity (D30) ------------------------------------------------------ #
def stim_conditions(detection: DetectionResult) -> list[str]:
    """The stimulus conditions of a recording's pulses with known sites, if there is more than one
    (each gets its own stim_evoked graph); [] otherwise (one graph of all pulses, as before)."""
    from meagraph.evoked import pulse_events
    from meagraph.io.mcs_events import mixed_conditions

    pulses = [s for s in pulse_events(detection.stim) if s.sites is not None and s.n]
    if not mixed_conditions(pulses):
        return []
    return [c for c in dict.fromkeys(np.concatenate([s.conditions for s in pulses]).tolist()) if c]


def stim_graph_from_detection(
    detection: DetectionResult,
    config: StimConnectivityConfig | None = None,
    channels: Literal["active", "all"] = "active",
    condition: str | None = None,
) -> ConnectivityResult:
    """Site -> electrode evoked connectivity (:mod:`meagraph.connectivity.stim`) from a detection
    result with per-pulse sites (a stimulation log, or ``--stim-site``).

    Readouts are the active (or all usable) channels; stimulated sites are added as nodes so their
    edges have a position. All spikes are used (stimulation periods are the point here), except
    those ``drop_pre_ms`` before a pulse (D26/D28). Latency floors come from detection's own
    blanking after each site's pulses. With ``condition`` (see :func:`stim_conditions`), only
    pulses of that stimulus condition are analysed; every pulse still keeps baselines clear. The
    condition is recorded in ``result.params["condition"]``.
    """
    from meagraph.connectivity.stim import StimConnectivityConfig as _Cfg
    from meagraph.connectivity.stim import stim_connectivity
    from meagraph.evoked import blanked_until_by_site_ms, pulse_events

    cfg = config or _Cfg()
    usable = [c for c in detection.trains.unit_ids if c not in detection.excluded]
    readouts = list(detection.active_channels) if channels == "active" else usable
    pulses = [s for s in pulse_events(detection.stim) if s.sites is not None and s.n]
    if not pulses:
        raise ValueError("no stimulation pulses with known sites; give a stimulation log or --stim-site")
    every = np.concatenate([s.onsets_s for s in pulses])  # every analysable pulse, of any condition
    if condition is not None:
        pulses = [s.select(s.conditions == condition) for s in pulses]
        pulses = [s for s in pulses if s.n]
        if not pulses:
            raise ValueError(f"no pulses with known sites at condition {condition!r}")
    onsets = np.concatenate([s.onsets_s for s in pulses])
    sites = np.concatenate([s.sites for s in pulses]).astype(object)
    known = [u for u in detection.trains.unit_ids if u in set(sites.tolist())]
    nodes = readouts + [u for u in known if u not in readouts]
    trains = detection.trains.select(nodes)
    if cfg.drop_pre_ms > 0:
        trains = trains.without_periods(np.column_stack([every - cfg.drop_pre_ms / 1e3, every - 1e-9]))
    floor_sites, floor = blanked_until_by_site_ms(detection, nodes, condition=condition)
    order = list(dict.fromkeys(sites.tolist()))
    rows = [floor[floor_sites.index(s)] if s in floor_sites else np.full(len(nodes), np.nan) for s in order]
    res = stim_connectivity(trains, onsets, sites, np.array(rows), readouts=readouts, config=cfg, clear_onsets_s=every)
    return replace(res, params={**res.params, "condition": condition or ""})


def condition_folder_name(condition: str) -> str:
    """Folder name for a condition's stim_evoked graph: "1000mV/500us/bi500" -> "1000mV_500us_bi500"."""
    return "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in condition) or "all"


def save_stim_graph(result: ConnectivityResult, folder: str | Path, channels: str = "active", inputs=(), overwrite: bool = False) -> Path:
    """``folder/all/`` (a standard result folder, so ``meagraph plot`` can draw it) plus
    ``folder/all/stim_pairs.csv``: every tested site-electrode pair with its latency floor,
    jitter, censoring and each edge rule's verdict. A graph of one stimulus condition goes to
    ``folder/<condition>/`` instead (:func:`condition_folder_name`)."""
    name = condition_folder_name(result.params.get("condition", ""))
    out = save_result(result, Path(folder) / name, inputs=inputs, overwrite=overwrite,
                      pipeline=GraphConfig(channels=channels, exclude_stim_post_ms=0.0))  # fmt: skip
    x = result.extra
    with open(out / "stim_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "electrode", "n_pulses", "effect_spikes_per_pulse", "p_value", "latency_ms", "latency_censored",
                    "floor_ms", "jitter_ms", "p_response", "p_baseline", "deconvolved", "evoked", "direct", "deconvolved_edge"])  # fmt: skip
        for i, j in zip(*np.nonzero(result.tested)):
            w.writerow([result.node_ids[i], result.node_ids[j], int(x["n_pulses_used"][i, j]), f"{result.weights[i, j]:.4g}",
                        f"{result.p_values[i, j]:.4g}", f"{result.delays_ms[i, j]:.3g}", int(x["latency_censored"][i, j]),
                        f"{x['floor_ms'][i, j]:.3g}", f"{x['jitter_ms'][i, j]:.3g}", f"{x['p_response'][i, j]:.3g}", f"{x['p_baseline'][i, j]:.3g}", f"{x['deconvolved'][i, j]:.4g}",
                        int(x["evoked"][i, j]), int(x["direct"][i, j]), int(x["deconvolved_mask"][i, j])])  # fmt: skip
    return out


def pulse_triggered_psth(detection: DetectionResult, trains: SpikeTrains, bin_ms: float = 1.0,
                         range_ms: tuple[float, float] = (-30.0, 60.0), condition: str = "") -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """``(psth, edges_ms)``: ``psth[i, j]`` counts spikes of unit ``j`` at each time after the pulses
    of site ``i`` (summed over pulses; zero rows for units that were never stimulated). With
    ``condition``, only pulses of that stimulus condition."""
    from meagraph.evoked import pulse_events

    edges = np.arange(range_ms[0], range_ms[1] + bin_ms / 2, bin_ms)
    out = np.zeros((trains.n_units, trains.n_units, edges.size - 1))
    pulses = [s for s in pulse_events(detection.stim) if s.sites is not None and s.n]
    if not pulses:
        return out, edges
    if condition:
        pulses = [s.select(s.conditions == condition) for s in pulses]
    onsets = np.concatenate([s.onsets_s for s in pulses])
    sites = np.concatenate([s.sites for s in pulses])
    for i, u in enumerate(trains.unit_ids):
        on = onsets[sites == u]
        if not on.size:
            continue
        for j, t in enumerate(trains.times_s):
            lo = np.searchsorted(t, on + edges[0] / 1e3)
            hi = np.searchsorted(t, on + edges[-1] / 1e3)
            rel = np.concatenate([t[a:b] - o for a, b, o in zip(lo, hi, on)]) * 1e3 if on.size else np.zeros(0)
            out[i, j] = np.histogram(rel, edges)[0]
    return out, edges
