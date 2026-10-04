"""Network bursts: brief periods when much of the array fires together (DECISIONS.md D15).

ISI_N method (Bakkum et al. 2013): pool every channel's spikes; wherever ``n_spikes``
consecutive pooled spikes fall within ``max_span_ms``, those spikes belong to a burst.
Overlapping runs merge into one burst, which is kept if at least ``min_channels``
distinct channels take part.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from meagraph.spiketrains import SpikeTrains


class BurstConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n_spikes: PositiveInt = 10
    max_span_ms: PositiveFloat = 100.0
    min_channels: PositiveInt = 3


def network_bursts(trains: SpikeTrains, config: BurstConfig | None = None) -> np.ndarray:
    """``(n_bursts, 2)`` array of [start, stop] times (s) on the recording clock."""
    cfg = config or BurstConfig()
    times = np.concatenate(trains.times_s) if trains.n_units else np.zeros(0)
    owner = np.concatenate([np.full(t.size, k) for k, t in enumerate(trains.times_s)]) if trains.n_units else np.zeros(0, int)
    order = np.argsort(times, kind="stable")
    times, owner = times[order], owner[order]
    n = cfg.n_spikes
    if times.size < n:
        return np.zeros((0, 2))
    span = times[n - 1 :] - times[: times.size - n + 1]
    starts = np.flatnonzero(span <= cfg.max_span_ms / 1e3)  # run k covers spikes k .. k + n - 1
    if starts.size == 0:
        return np.zeros((0, 2))
    # Runs that overlap or touch belong to the same burst.
    breaks = np.flatnonzero(np.diff(starts) > n - 1) + 1
    bursts = []
    for group in np.split(starts, breaks):
        first, last = group[0], group[-1] + n - 1
        if np.unique(owner[first : last + 1]).size >= cfg.min_channels:
            bursts.append((times[first], times[last]))
    return np.array(bursts, dtype=np.float64).reshape(-1, 2)


def remove_periods(trains: SpikeTrains, periods: np.ndarray, pad_s: float = 0.0) -> SpikeTrains:
    """Drop spikes inside ``[start - pad, stop + pad]`` of any period. Duration is unchanged."""
    p = np.asarray(periods, dtype=np.float64).reshape(-1, 2)
    if p.size == 0:
        return trains
    lo, hi = p[:, 0] - pad_s, p[:, 1] + pad_s
    kept = []
    for t in trains.times_s:
        k = np.searchsorted(lo, t, side="right") - 1
        inside = (k >= 0) & (t <= hi[np.maximum(k, 0)])
        kept.append(t[~inside])
    return SpikeTrains(trains.unit_ids, tuple(kept), trains.t_start_s, trains.t_stop_s, trains.channel_ids, trains.positions_um)
