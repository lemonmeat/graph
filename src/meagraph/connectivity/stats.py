"""Building blocks shared by the estimators: correlograms, window counts, surrogates, STTC, FDR.

Lag convention everywhere: a positive lag means the *target* fires after the *source*.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import false_discovery_control, nbinom, norm, poisson
from spikeinterface.postprocessing import compute_correlograms

from meagraph.spiketrains import SpikeTrains


def cross_correlograms(trains: SpikeTrains, bin_ms: float, max_lag_ms: float, sampling_frequency_hz: float = 10_000.0):
    """All-pairs correlograms via SpikeInterface (DECISIONS.md D13).

    Returns ``(ccg, lag_edges_ms)``: ``ccg[i, j, k]`` counts spikes of j whose time after a
    spike of i falls in ``[lag_edges_ms[k], lag_edges_ms[k + 1])``.
    """
    sorting = trains.to_sorting(sampling_frequency_hz)
    ccg, edges = compute_correlograms(sorting, window_ms=2 * max_lag_ms, bin_ms=bin_ms, method="numpy")
    # SpikeInterface's ccg[a, b] counts t_a - t_b; transposing gives t_target - t_source.
    return np.ascontiguousarray(ccg.transpose(1, 0, 2)).astype(np.float64), np.asarray(edges, dtype=np.float64)


def window_counts(source_s: np.ndarray, target_s: np.ndarray, lo_s: float, hi_s: float) -> int:
    """Number of (source, target) spike pairs with ``target - source`` in ``[lo_s, hi_s)``."""
    return int(np.sum(np.searchsorted(target_s, source_s + hi_s, "left") - np.searchsorted(target_s, source_s + lo_s, "left")))


def interval_jitter(times_s: np.ndarray, window_s: float, t_start: float, t_stop: float, rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` interval-jitter surrogates of one train, shape (n, n_spikes), each row sorted.

    Every spike is redrawn uniformly within the fixed window ``[t_start + k*window, ...)`` that
    contains it (Amarasingham et al. 2012); the last window ends at ``t_stop``. This is the
    definition of Elephant's ``jitter_spikes`` without its ``t_start`` offset bug (D13).
    """
    t = np.asarray(times_s, dtype=np.float64)
    k = np.floor((t - t_start) / window_s)
    left = t_start + k * window_s
    width = np.minimum(left + window_s, t_stop) - left
    return np.sort(left + rng.random((n, t.size)) * width, axis=1)


def sttc(a: np.ndarray, b: np.ndarray, dt_s: float, t_start: float, t_stop: float) -> float:
    """Spike time tiling coefficient (Cutts & Eglen 2014), vectorized; matches Elephant (tested)."""
    if a.size == 0 or b.size == 0:
        return np.nan
    p_a = _fraction_within(a, b, dt_s)
    p_b = _fraction_within(b, a, dt_s)
    t_a = _tiled_fraction(a, dt_s, t_start, t_stop)
    t_b = _tiled_fraction(b, dt_s, t_start, t_stop)
    return 0.5 * (_term(p_a, t_b) + _term(p_b, t_a))


def _term(p: float, t: float) -> float:
    return (p - t) / (1 - p * t) if p * t != 1 else 1.0


def _fraction_within(a: np.ndarray, b: np.ndarray, dt_s: float) -> float:
    """Fraction of spikes in ``a`` with a spike of ``b`` within ±dt (inclusive)."""
    hit = np.searchsorted(b, a - dt_s, "left") < np.searchsorted(b, a + dt_s, "right")
    return float(np.mean(hit))


def _tiled_fraction(a: np.ndarray, dt_s: float, t_start: float, t_stop: float) -> float:
    """Fraction of [t_start, t_stop] within ±dt of any spike in ``a``."""
    covered = 2 * dt_s * a.size - np.sum(np.clip(2 * dt_s - np.diff(a), 0, None))
    covered -= max(0.0, dt_s - (a[0] - t_start)) + max(0.0, dt_s - (t_stop - a[-1]))
    return float(covered / (t_stop - t_start))


def surrogate_p(observed: float, surrogates: np.ndarray, tail: str = "greater") -> float:
    """Monte-Carlo p-value with the +1 correction, so it is never 0 (but never below 1/(N+1))."""
    s = np.asarray(surrogates)
    extreme = np.sum(s >= observed) if tail == "greater" else np.sum(s <= observed)
    return float((1 + extreme) / (1 + s.size))


def count_tail_p(observed: float, surrogates: np.ndarray, tail: str = "greater") -> float:
    """p-value of a count against surrogate counts, below Monte-Carlo resolution (DECISIONS.md D17).

    A distribution is moment-matched to the surrogates: negative binomial when they are
    overdispersed (as network bursts make them), Poisson otherwise. The mean gets +0.5/N so that
    surrogates that are all zero still give a finite, honest p-value.
    """
    s = np.asarray(surrogates, dtype=np.float64)
    mean = (s.sum() + 0.5) / s.size
    var = s.var(ddof=1) if s.size > 1 else mean
    if var > mean * (1 + 1e-6):
        dist = nbinom(mean**2 / (var - mean), mean / var)
    else:
        dist = poisson(mean)
    return float(dist.sf(observed - 1) if tail == "greater" else dist.cdf(observed))


def normal_tail_p(observed: float, surrogates: np.ndarray, tail: str = "greater") -> float:
    """p-value of a continuous statistic from a normal fitted to the surrogates (D17)."""
    s = np.asarray(surrogates, dtype=np.float64)
    sd = s.std(ddof=1) if s.size > 1 else 0.0
    if not np.isfinite(sd) or sd == 0:
        return surrogate_p(observed, s, tail)
    z = (observed - s.mean()) / sd
    return float(norm.sf(z) if tail == "greater" else norm.cdf(z))


def fdr_mask(p_values: np.ndarray, q: float) -> np.ndarray:
    """Benjamini–Hochberg over the finite entries of ``p_values``; NaN entries are never significant."""
    out = np.zeros(p_values.shape, dtype=bool)
    finite = np.isfinite(p_values)
    if finite.any():
        adjusted = false_discovery_control(np.clip(p_values[finite], 0, 1), method="bh")
        out[finite] = adjusted <= q
    return out
