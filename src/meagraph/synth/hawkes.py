"""Simple spiking networks with known directed connections (DECISIONS.md D16).

A linear Hawkes process simulated as a branching process: every spike of unit i causes, on
average, ``W[i, j]`` extra spikes in unit j (the spike transmission probability), each after the
connection's delay plus a little jitter. The true connectivity is exactly ``W``, which is what
the correlogram estimators measure, so recovery can be scored directly.

Optional confounds (each off by default): network bursts (shared rate surges), periodic
stimulation (shared, time-locked drive), and imperfect detection (missed and false spikes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt

from meagraph.probe.spec import load_probe_spec
from meagraph.spiketrains import SpikeTrains


class NetworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_s: PositiveFloat = 600.0
    n_units: PositiveInt = 16
    probe: str = "cube4x4x4_E-00303"  # units sit on randomly chosen contacts of this probe
    rate_hz: tuple[PositiveFloat, PositiveFloat] = (0.2, 2.0)  # baseline rates, log-uniform per unit
    connection_prob: NonNegativeFloat = 0.1
    weight: tuple[NonNegativeFloat, NonNegativeFloat] = (0.05, 0.2)  # transmission probability, uniform per connection
    delay_ms: tuple[PositiveFloat, PositiveFloat] = (1.5, 3.5)  # mean latency, uniform per connection
    delay_jitter_ms: NonNegativeFloat = 0.25  # sd of each transmitted spike's latency
    burst_rate_hz: NonNegativeFloat = 0.0  # network bursts per second
    burst_duration_ms: PositiveFloat = 100.0
    burst_gain: NonNegativeFloat = 20.0  # rate multiplier inside a burst
    burst_participation: float = 0.8  # probability that a unit joins a given burst
    stim_interval_s: NonNegativeFloat = 0.0  # 0: no stimulation
    stim_response_prob: float = 0.3
    stim_latency_ms: tuple[PositiveFloat, PositiveFloat] = (5.0, 15.0)
    miss_fraction: float = 0.0  # detection: fraction of true spikes lost
    false_rate_hz: NonNegativeFloat = 0.0  # detection: noise spikes added per unit
    dead_time_ms: NonNegativeFloat = 1.0  # detection cannot resolve spikes closer than this
    seed: int = 0


@dataclass(frozen=True, eq=False)
class SyntheticNetwork:
    trains: SpikeTrains
    weights: np.ndarray  # (source, target) true transmission probability; 0 = not connected
    delays_ms: np.ndarray  # mean latency of each connection; NaN = not connected
    bursts: np.ndarray  # (k, 2) [start, stop] s
    stim_times_s: np.ndarray
    config: NetworkConfig

    @property
    def connected(self) -> np.ndarray:
        return self.weights > 0


def _immigrants(cfg: NetworkConfig, rates: np.ndarray, rng: np.random.Generator):
    """Spikes not caused by other units: baseline, bursts and stimulation."""
    T, n = cfg.duration_s, cfg.n_units
    times, units = [], []
    for i, r in enumerate(rates):
        k = rng.poisson(r * T)
        times.append(rng.uniform(0, T, k))
        units.append(np.full(k, i))
    n_bursts = rng.poisson(cfg.burst_rate_hz * T)
    onsets = np.sort(rng.uniform(0, T, n_bursts))
    dur = cfg.burst_duration_ms / 1e3
    for t0 in onsets:
        for i in np.flatnonzero(rng.random(n) < cfg.burst_participation):
            k = rng.poisson(rates[i] * cfg.burst_gain * dur)
            times.append(rng.uniform(t0, t0 + dur, k))
            units.append(np.full(k, i))
    stim = np.arange(cfg.stim_interval_s, T, cfg.stim_interval_s) if cfg.stim_interval_s > 0 else np.zeros(0)
    for t0 in stim:
        hit = np.flatnonzero(rng.random(n) < cfg.stim_response_prob)
        times.append(t0 + rng.uniform(*cfg.stim_latency_ms, hit.size) / 1e3)
        units.append(hit)
    bursts = np.column_stack([onsets, onsets + dur]) if n_bursts else np.zeros((0, 2))
    return np.concatenate(times), np.concatenate(units).astype(np.int64), bursts, stim


def _dead_time(t: np.ndarray, dead_s: float) -> np.ndarray:
    if t.size < 2 or dead_s <= 0:
        return t
    keep = [t[0]]
    for x in t[1:]:
        if x - keep[-1] >= dead_s:
            keep.append(x)
    return np.array(keep)


def simulate_network(config: NetworkConfig | None = None) -> SyntheticNetwork:
    cfg = config or NetworkConfig()
    rng = np.random.default_rng(cfg.seed)
    n, T = cfg.n_units, cfg.duration_s

    spec = load_probe_spec(cfg.probe)
    if n > len(spec.contacts):
        raise ValueError(f"{n} units but probe {cfg.probe!r} has {len(spec.contacts)} contacts")
    pick = np.sort(rng.choice(len(spec.contacts), n, replace=False))
    labels = tuple(spec.labels[k] for k in pick)
    positions = spec.positions_um()[pick]
    if positions.shape[1] == 2:
        positions = np.column_stack([positions, np.zeros(n)])

    rates = np.exp(rng.uniform(*np.log(cfg.rate_hz), n))
    connected = rng.random((n, n)) < cfg.connection_prob
    np.fill_diagonal(connected, False)
    W = np.where(connected, rng.uniform(*cfg.weight, (n, n)), 0.0)
    D = np.where(connected, rng.uniform(*cfg.delay_ms, (n, n)), np.nan)
    if n and np.max(np.abs(np.linalg.eigvals(W))) >= 0.95:
        raise ValueError("network too excitable (spectral radius of W >= 0.95); lower weights or connection_prob")

    gen_t, gen_u, bursts, stim = _immigrants(cfg, rates, rng)
    all_t, all_u = [gen_t], [gen_u]
    sources, targets = np.nonzero(W)
    while gen_t.size:  # each generation of caused spikes; terminates because W is subcritical
        new_t, new_u = [], []
        for i, j in zip(sources, targets):
            t_i = gen_t[gen_u == i]
            k = rng.poisson(W[i, j], t_i.size)
            if k.sum():
                lat = (D[i, j] + rng.normal(0.0, cfg.delay_jitter_ms, k.sum())) / 1e3
                new_t.append(np.repeat(t_i, k) + np.maximum(lat, 0.0))
                new_u.append(np.full(k.sum(), j))
        gen_t = np.concatenate(new_t) if new_t else np.zeros(0)
        gen_u = np.concatenate(new_u) if new_u else np.zeros(0, np.int64)
        all_t.append(gen_t)
        all_u.append(gen_u)
    t_all, u_all = np.concatenate(all_t), np.concatenate(all_u)

    times = {}
    for k, label in enumerate(labels):
        t = np.sort(t_all[(u_all == k) & (t_all >= 0) & (t_all < T)])
        t = t[rng.random(t.size) >= cfg.miss_fraction]
        t = np.sort(np.concatenate([t, rng.uniform(0, T, rng.poisson(cfg.false_rate_hz * T))]))
        times[label] = _dead_time(t, cfg.dead_time_ms / 1e3)
    trains = SpikeTrains.from_dict(times, 0.0, T).with_positions(positions)
    return SyntheticNetwork(trains, W, D, bursts, stim, cfg)
