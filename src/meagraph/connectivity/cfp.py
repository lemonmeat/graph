"""Conditional firing probability (CFP; le Feber et al. 2007): functional relations at burst scale.

``CFP[i, j](tau)`` is the probability that electrode j fires in the 1 ms bin at ``tau`` after a
spike of electrode i, for tau in [0, 500) ms. A related pair has a CFP curve with a clear peak;
its height above the curve's offset is the relation's *strength* and its latency the *delay*.

In this implementation (DECISIONS.md D22, Proposed):

- **Test.** "Clear peak" is made a test: the statistic is the peak of the curve smoothed over
  ``min_width_ms`` minus the curve's mean. The null is the same statistic after shifting every
  train circularly by an independent random offset, which keeps each train's own structure
  (its bursts and rate) but breaks the timing between trains. A normal fitted to the surrogates
  gives the p-value (D17); Benjamini-Hochberg selects pairs.
- **Strength and delay** come from fitting ``M / (1 + ((tau - T) / w)**2) + offset`` (the
  le Feber form; the exact original fit should be checked against the paper) to the curve.
- **Validity** (as in Martiniuc et al. 2015): the peak must be at least ``min_width_ms`` wide at
  80 % of its height (for this curve that width is ``w``), and the delay at most ``max_delay_ms``.

Shared network bursts are part of what CFP measures, so co-bursting pairs are related even
without a synapse. CFP is a measure of functional coupling, not of monosynaptic connections.
"""

from __future__ import annotations

import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from scipy.optimize import OptimizeWarning, curve_fit

from meagraph.connectivity.base import ConnectivityResult, empty_matrices, register, testable
from meagraph.connectivity.stats import circular_shift, cross_correlograms, fdr_mask, normal_tail_p
from meagraph.spiketrains import SpikeTrains


def lorentzian(tau, m, t, w, offset):
    return m / (1.0 + ((tau - t) / w) ** 2) + offset


def _curves(trains: SpikeTrains, bin_ms: float, max_lag_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """CFP curves (source, target, lag) for lags [0, max_lag), and the bin centres in ms."""
    ccg, edges = cross_correlograms(trains, bin_ms, max_lag_ms)
    pos = edges[:-1] >= -1e-9
    n = np.maximum(trains.n_spikes(), 1).astype(np.float64)
    return ccg[:, :, pos] / n[:, None, None], (edges[:-1][pos] + edges[1:][pos]) / 2


def _statistic(curves: np.ndarray, smooth_bins: int) -> np.ndarray:
    kernel = np.ones(smooth_bins) / smooth_bins
    smoothed = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), -1, curves)
    return smoothed.max(axis=-1) - curves.mean(axis=-1)


@register
class CFP:
    name = "cfp"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        bin_ms: PositiveFloat = 1.0
        max_lag_ms: PositiveFloat = 500.0
        min_width_ms: PositiveFloat = 5.0  # smoothing of the test statistic, and the minimum peak width
        max_delay_ms: PositiveFloat = 250.0
        n_surrogates: PositiveInt = 1000
        q: PositiveFloat = 0.05
        min_spikes: PositiveInt = 100
        seed: int = 0

    def estimate(self, trains: SpikeTrains, config: Config | None = None) -> ConnectivityResult:
        cfg = config or self.Config()
        n, t0, t1 = trains.n_units, trains.t_start_s, trains.t_stop_s
        mask = testable(trains, cfg.min_spikes)
        weights, delays, p_values, _ = empty_matrices(n)
        smooth = max(int(round(cfg.min_width_ms / cfg.bin_ms)), 1)
        curves, lags = _curves(trains, cfg.bin_ms, cfg.max_lag_ms)
        observed = _statistic(curves, smooth)

        rng = np.random.default_rng(cfg.seed)
        lo, hi = cfg.max_lag_ms / 1e3, trains.duration_s - cfg.max_lag_ms / 1e3
        null = np.empty((n, n, cfg.n_surrogates))
        for k in range(cfg.n_surrogates):
            shifts = rng.uniform(lo, max(hi, lo), n)
            shifted = SpikeTrains(trains.unit_ids, tuple(circular_shift(t, t0, t1, s) for t, s in zip(trains.times_s, shifts)), t0, t1)
            null[:, :, k] = _statistic(_curves(shifted, cfg.bin_ms, cfg.max_lag_ms)[0], smooth)

        fit = np.full((n, n, 4), np.nan)  # M, T, w, offset
        valid = np.zeros((n, n), dtype=bool)
        for i, j in zip(*np.nonzero(mask)):
            p_values[i, j] = normal_tail_p(observed[i, j], null[i, j], "greater")
            fit[i, j] = _fit(curves[i, j], lags, smooth)
            m, t, w, _ = fit[i, j]
            if np.isfinite(m):
                weights[i, j], delays[i, j] = m, t
                valid[i, j] = w >= cfg.min_width_ms and t <= cfg.max_delay_ms
        significant = fdr_mask(p_values, cfg.q) & valid
        return ConnectivityResult(
            method=self.name, node_ids=trains.unit_ids, weights=weights, delays_ms=delays, p_values=p_values,
            significant=significant, directed=True, params=cfg.model_dump(), n_spikes=trains.n_spikes(),
            duration_s=trains.duration_s, positions_um=trains.positions_um,
            extra=dict(cfp=curves, lags_ms=lags, fit_m_t_w_offset=fit, valid_peak=valid, test_significant=fdr_mask(p_values, cfg.q)),
        )  # fmt: skip


def _fit(curve: np.ndarray, lags: np.ndarray, smooth: int) -> np.ndarray:
    """Least-squares fit of the le Feber curve; NaNs if it does not converge."""
    smoothed = np.convolve(curve, np.ones(smooth) / smooth, mode="same")
    offset0 = float(np.median(curve))
    k = int(np.argmax(smoothed))
    p0 = [max(float(smoothed[k]) - offset0, 1e-9), float(lags[k]), 10.0, offset0]
    bounds = ([0.0, float(lags[0]), float(lags[1] - lags[0]), 0.0], [np.inf, float(lags[-1]), float(lags[-1]), np.inf])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(lorentzian, lags, curve, p0=p0, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError):
        return np.full(4, np.nan)
    return params
