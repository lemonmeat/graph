"""Cross-correlogram estimators of putative monosynaptic connections.

``cch_hollow``: excess in the synaptic window over a slow baseline from a partially hollow
Gaussian convolution, tested against Poisson (Stark & Abeles 2009; English et al. 2017).
A port of the legacy ``spontaneous_ccg.py`` that adds the promised correction across pairs.

``cch_jitter``: excess in the synaptic window over interval-jitter surrogates (Amarasingham
et al. 2012). The surrogates keep each train's spike count in every jitter window, so any
co-modulation slower than the window (bursts, rate changes) is in the null.

Both report excitatory connections. The weight is the spike transmission probability:
excess target spikes in the window per source spike.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from scipy.stats import poisson

from meagraph.connectivity.base import ConnectivityResult, empty_matrices, register, testable
from meagraph.connectivity.stats import count_tail_p, cross_correlograms, fdr_mask, interval_jitter, surrogate_p, window_counts
from meagraph.spiketrains import SpikeTrains


class _Common(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bin_ms: PositiveFloat = 0.5
    max_lag_ms: PositiveFloat = 30.0
    window_ms: tuple[float, float] = (1.0, 4.0)  # synaptic window [lo, hi) after the source spike
    q: PositiveFloat = 0.05  # Benjamini-Hochberg level across tested ordered pairs
    min_spikes: PositiveInt = 100


def _window_bins(edges: np.ndarray, window_ms: tuple[float, float]) -> np.ndarray:
    lo, hi = window_ms
    sel = (edges[:-1] >= lo - 1e-9) & (edges[1:] <= hi + 1e-9)
    if not sel.any():
        raise ValueError(f"no correlogram bin lies inside the window {window_ms} ms")
    return sel


@register
class CCHHollow:
    name = "cch_hollow"

    class Config(_Common):
        kernel_sd_ms: PositiveFloat = 10.0
        hollow_fraction: float = 0.6  # centre weight of the kernel is scaled by (1 - hollow_fraction)
        drop_symmetric: bool = True  # legacy common-input heuristic: drop pairs significant both ways at ~equal lag

    def estimate(self, trains: SpikeTrains, config: Config | None = None) -> ConnectivityResult:
        cfg = config or self.Config()
        n = trains.n_units
        ccg, edges = cross_correlograms(trains, cfg.bin_ms, cfg.max_lag_ms)
        centres = (edges[:-1] + edges[1:]) / 2
        win = _window_bins(edges, cfg.window_ms)

        sd_bins = cfg.kernel_sd_ms / cfg.bin_ms
        half = int(4 * sd_bins)
        x = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (x / sd_bins) ** 2)
        kernel[half] *= 1.0 - cfg.hollow_fraction
        kernel /= kernel.sum()
        padded = np.pad(ccg, ((0, 0), (0, 0), (half, half)), mode="edge")
        baseline = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), -1, padded)
        baseline = np.maximum(baseline, 1e-6)

        obs, lam = ccg[:, :, win], baseline[:, :, win]
        p_bins = poisson.sf(obs - 1, lam)  # P(X >= observed) per bin
        p = np.minimum(p_bins.min(axis=-1) * win.sum(), 1.0)  # Bonferroni within the window

        weights, delays, p_values, _ = empty_matrices(n)
        mask = testable(trains, cfg.min_spikes)
        excess = (obs - lam).sum(axis=-1) / np.maximum(trains.n_spikes()[:, None], 1)
        best = centres[win][np.argmax(obs - lam, axis=-1)]
        p_values[mask], weights[mask], delays[mask] = p[mask], excess[mask], best[mask]
        significant = fdr_mask(p_values, cfg.q)
        dropped = np.zeros_like(significant)
        if cfg.drop_symmetric:
            dropped = significant & significant.T & (np.abs(delays - delays.T) < 1.0)
            significant &= ~dropped
        return ConnectivityResult(
            method=self.name, node_ids=trains.unit_ids, weights=weights, delays_ms=delays, p_values=p_values,
            significant=significant, directed=True, params=cfg.model_dump(), n_spikes=trains.n_spikes(),
            duration_s=trains.duration_s, positions_um=trains.positions_um,
            extra=dict(ccg=ccg, lag_edges_ms=edges, baseline=baseline, dropped_symmetric=dropped),
        )  # fmt: skip


@register
class CCHJitter:
    name = "cch_jitter"

    class Config(_Common):
        jitter_window_ms: PositiveFloat = 10.0
        n_surrogates: PositiveInt = 1000
        seed: int = 0

    def estimate(self, trains: SpikeTrains, config: Config | None = None) -> ConnectivityResult:
        cfg = config or self.Config()
        n = trains.n_units
        mask = testable(trains, cfg.min_spikes)
        ccg, edges = cross_correlograms(trains, cfg.bin_ms, cfg.max_lag_ms)
        centres = (edges[:-1] + edges[1:]) / 2
        win = _window_bins(edges, cfg.window_ms)
        lo, hi = (w / 1e3 for w in cfg.window_ms)

        used = np.flatnonzero(mask.any(axis=0) | mask.any(axis=1))
        rng = np.random.default_rng(cfg.seed)
        surrogates = {
            i: interval_jitter(trains.times_s[i], cfg.jitter_window_ms / 1e3, trains.t_start_s, trains.t_stop_s, rng, cfg.n_surrogates)
            for i in used
        }
        observed = np.zeros((n, n))
        null = np.zeros((n, n, cfg.n_surrogates))
        for i, j in zip(*np.nonzero(mask)):
            observed[i, j] = window_counts(trains.times_s[i], trains.times_s[j], lo, hi)
            si, sj = surrogates[i], surrogates[j]
            null[i, j] = [window_counts(si[k], sj[k], lo, hi) for k in range(cfg.n_surrogates)]

        weights, delays, p_values, _ = empty_matrices(n)
        p_inh, p_empirical = np.full((n, n), np.nan), np.full((n, n), np.nan)
        expected = null.mean(axis=-1)
        excess_bins = ccg[:, :, win] - (expected / win.sum())[:, :, None]
        for i, j in zip(*np.nonzero(mask)):
            p_values[i, j] = count_tail_p(observed[i, j], null[i, j], "greater")
            p_inh[i, j] = count_tail_p(observed[i, j], null[i, j], "less")
            p_empirical[i, j] = surrogate_p(observed[i, j], null[i, j], "greater")
            weights[i, j] = (observed[i, j] - expected[i, j]) / len(trains.times_s[i])
            delays[i, j] = centres[win][np.argmax(excess_bins[i, j])]
        return ConnectivityResult(
            method=self.name, node_ids=trains.unit_ids, weights=weights, delays_ms=delays, p_values=p_values,
            significant=fdr_mask(p_values, cfg.q), directed=True, params=cfg.model_dump(), n_spikes=trains.n_spikes(),
            duration_s=trains.duration_s, positions_um=trains.positions_um,
            extra=dict(ccg=ccg, lag_edges_ms=edges, expected_window_count=expected, p_inhibitory=p_inh, p_empirical=p_empirical),
        )  # fmt: skip
