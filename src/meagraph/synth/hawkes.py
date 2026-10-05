"""Simple spiking networks with known directed connections (DECISIONS.md D16).

A linear Hawkes process simulated as a branching process: every spike of unit i causes, on
average, ``W[i, j]`` extra spikes in unit j (the spike transmission probability), each after the
connection's delay plus a little jitter. The true connectivity is exactly ``W``, which is what
the correlogram estimators measure, so recovery can be scored directly.

Optional confounds (each off by default): network bursts (shared rate surges), periodic
stimulation (shared, time-locked drive), and imperfect detection (missed and false spikes).

Optional inhibition (off by default; DECISIONS.md D22): a fraction of units are inhibitory, and
every connection they make is inhibitory (Dale's principle). After an inhibitory spike, each
target spike in the window [delay, delay + ``inhibition_ms``] is deleted with the connection's
probability ``S[i, j]``; deleted spikes cause nothing. With inhibition the cascade is simulated in
time order, since a later inhibitory spike can only delete spikes that come after it. Without
inhibition both orders give the same process, and the faster generation-by-generation
simulation is used (so excitatory-only networks are unchanged by this option).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt

from meagraph.probe.spec import DEFAULT_PROBE, load_probe_spec
from meagraph.spiketrains import SpikeTrains


class NetworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_s: PositiveFloat = 600.0
    n_units: PositiveInt = 16
    probe: str = DEFAULT_PROBE  # units sit on randomly chosen contacts of this probe
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
    inhibitory_fraction: float = 0.0  # fraction of units whose connections are all inhibitory
    inhibition: tuple[float, float] = (0.6, 0.6)  # suppression probability, uniform per inhibitory connection
    inhibition_ms: PositiveFloat = 10.0  # suppression lasts this long after the connection's delay
    seed: int = 0


@dataclass(frozen=True, eq=False)
class SyntheticNetwork:
    trains: SpikeTrains
    weights: np.ndarray  # (source, target) true transmission probability of excitatory connections; 0 = none
    delays_ms: np.ndarray  # mean latency of each connection, excitatory or inhibitory; NaN = not connected
    bursts: np.ndarray  # (k, 2) [start, stop] s
    stim_times_s: np.ndarray
    config: NetworkConfig
    suppression: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # inhibitory connections; 0 = none

    @property
    def excitatory(self) -> np.ndarray:
        return self.weights > 0

    @property
    def inhibitory(self) -> np.ndarray:
        return self.suppression > 0 if self.suppression.size else np.zeros_like(self.excitatory)

    @property
    def connected(self) -> np.ndarray:
        """Any connection, excitatory or inhibitory."""
        return self.excitatory | self.inhibitory


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
    positions = spec.positions_xyz_um()[pick]

    rates = np.exp(rng.uniform(*np.log(cfg.rate_hz), n))
    connected = rng.random((n, n)) < cfg.connection_prob
    np.fill_diagonal(connected, False)
    W = np.where(connected, rng.uniform(*cfg.weight, (n, n)), 0.0)
    D = np.where(connected, rng.uniform(*cfg.delay_ms, (n, n)), np.nan)
    S = np.zeros((n, n))
    if cfg.inhibitory_fraction > 0:
        inhibitory_units = rng.random(n) < cfg.inhibitory_fraction
        S[inhibitory_units] = np.where(connected[inhibitory_units], rng.uniform(*cfg.inhibition, (int(inhibitory_units.sum()), n)), 0.0)
        W[inhibitory_units] = 0.0
    if n and np.max(np.abs(np.linalg.eigvals(W))) >= 0.95:
        raise ValueError("network too excitable (spectral radius of W >= 0.95); lower weights or connection_prob")

    gen_t, gen_u, bursts, stim = _immigrants(cfg, rates, rng)
    if S.any():
        t_all, u_all = _cascade_in_time_order(gen_t, gen_u, W, D, S, cfg, rng)
    else:
        t_all, u_all = _cascade_by_generation(gen_t, gen_u, W, D, cfg, rng)

    times = {}
    for k, label in enumerate(labels):
        t = np.sort(t_all[(u_all == k) & (t_all >= 0) & (t_all < T)])
        t = t[rng.random(t.size) >= cfg.miss_fraction]
        t = np.sort(np.concatenate([t, rng.uniform(0, T, rng.poisson(cfg.false_rate_hz * T))]))
        times[label] = _dead_time(t, cfg.dead_time_ms / 1e3)
    trains = SpikeTrains.from_dict(times, 0.0, T).with_positions(positions)
    return SyntheticNetwork(trains, W, D, bursts, stim, cfg, S)


def _cascade_by_generation(gen_t, gen_u, W, D, cfg: NetworkConfig, rng):
    """Excitatory-only cascade, one generation of caused spikes at a time (vectorized)."""
    all_t, all_u = [gen_t], [gen_u]
    sources, targets = np.nonzero(W)
    while gen_t.size:  # terminates because W is subcritical
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
    return np.concatenate(all_t), np.concatenate(all_u)


def _cascade_in_time_order(gen_t, gen_u, W, D, S, cfg: NetworkConfig, rng):
    """Cascade with inhibition: candidate spikes are processed in time order. Each survives the
    suppression windows active on its unit (independently per window), and only survivors cause
    excitatory children or open suppression windows on their inhibitory targets."""
    n, T = W.shape[0], cfg.duration_s
    heap = list(zip(gen_t.tolist(), gen_u.tolist()))
    heapq.heapify(heap)
    exc_targets = [np.flatnonzero(W[i]) for i in range(n)]
    inh_targets = [np.flatnonzero(S[i]) for i in range(n)]
    windows: list[list[tuple[float, float, float]]] = [[] for _ in range(n)]  # per unit: heap of (end, start, keep)
    dur, jitter = cfg.inhibition_ms / 1e3, cfg.delay_jitter_ms / 1e3
    out_t, out_u = [], []
    while heap:
        t, j = heapq.heappop(heap)
        if t >= T:
            continue  # beyond the recording, and so are its children
        active = windows[j]
        while active and active[0][0] < t:
            heapq.heappop(active)
        keep = np.prod([k for _, start, k in active if start <= t]) if active else 1.0
        if keep < 1.0 and rng.random() >= keep:
            continue
        out_t.append(t)
        out_u.append(j)
        for k in exc_targets[j]:
            for _ in range(rng.poisson(W[j, k])):
                heapq.heappush(heap, (t + max(D[j, k] / 1e3 + rng.normal(0.0, jitter), 0.0), int(k)))
        for k in inh_targets[j]:
            start = t + D[j, k] / 1e3
            heapq.heappush(windows[k], (start + dur, start, 1.0 - S[j, k]))
    return np.array(out_t, dtype=np.float64), np.array(out_u, dtype=np.int64)
