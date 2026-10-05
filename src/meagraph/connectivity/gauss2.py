"""Two-Gaussian correlogram weight of Kumar et al. 2026 (Nat. Electron. 9:532), for comparison
with the lab's earlier analyses (DECISIONS.md D23).

As in the paper: a cross-correlogram (2 ms bins, Fig. 3e) is fitted with

    a1 * exp(-(x - b1)**2 / (2 c1**2)) + a2 * exp(-(x - b2)**2 / (2 c2**2))

and the connection weight is ``(a1 + a2) / (c1 + c2)``. The side of zero on which the fitted
curve peaks gives the net direction: a peak at positive lag means the target fires after the
source. The paper describes no significance test, so ``p_values`` are NaN and every tested pair
whose peak is off zero is an edge in its net direction (``min_weight`` can raise the bar).

Unknowns that change the weight's scale (to be matched to the original NeuroExplorer settings):
the correlogram normalization (here counts per bin, or firing rate in Hz) and the lag range
(here +-100 ms). Lags are in ms, so ``c`` is in ms.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt
from scipy.optimize import OptimizeWarning, curve_fit

from meagraph.connectivity.base import ConnectivityResult, empty_matrices, register, testable
from meagraph.connectivity.stats import cross_correlograms
from meagraph.spiketrains import SpikeTrains


def gauss2(x, a1, b1, c1, a2, b2, c2):
    return a1 * np.exp(-((x - b1) ** 2) / (2 * c1**2)) + a2 * np.exp(-((x - b2) ** 2) / (2 * c2**2))


def fit_gauss2(counts: np.ndarray, lags_ms: np.ndarray, bin_ms: float) -> np.ndarray:
    """Least-squares fit from several starts (peak at the largest bin, at the smoothed maximum,
    and at zero lag); returns the best (a1, b1, c1, a2, b2, c2), narrower Gaussian first, or NaNs.

    Widths are bounded by [bin/2, 10 x the lag span]; the paper's (MATLAB) fit is unbounded, and
    on flat correlograms the second Gaussian can widen to absorb the baseline (D23)."""
    span = float(lags_ms[-1] - lags_ms[0])
    base = float(np.median(counts))
    k_raw = int(np.argmax(counts))
    k_smooth = int(np.argmax(np.convolve(counts, np.ones(5) / 5, mode="same")))
    lo = [0.0, float(lags_ms[0]), bin_ms / 2, 0.0, float(lags_ms[0]), bin_ms / 2]
    hi = [np.inf, float(lags_ms[-1]), 10 * span, np.inf, float(lags_ms[-1]), 10 * span]
    starts = [
        [max(float(counts[k_raw]) - base, 1e-9), float(lags_ms[k_raw]), bin_ms, max(base, 1e-9), 0.0, span / 2],
        [max(float(counts[k_smooth]) - base, 1e-9), float(lags_ms[k_smooth]), 3 * bin_ms, max(base, 1e-9), 0.0, span / 2],
        [max(float(counts.max()) - base, 1e-9), 0.0, 5 * bin_ms, max(base, 1e-9), 0.0, span],
    ]
    best, best_sse = np.full(6, np.nan), np.inf
    scale = max(float(counts.max()), 1e-9)
    for p0 in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                p, _ = curve_fit(lambda x, a1, b1, c1, a2, b2, c2: gauss2(x, a1 * scale, b1, c1, a2 * scale, b2, c2) / scale,
                                 lags_ms, counts / scale, p0=[p0[0] / scale, *p0[1:3], p0[3] / scale, *p0[4:]],
                                 bounds=([lo[0], *lo[1:3], lo[3], *lo[4:]], hi), maxfev=5000)  # fmt: skip
        except (RuntimeError, ValueError):
            continue
        p = np.array([p[0] * scale, p[1], p[2], p[3] * scale, p[4], p[5]])
        sse = float(np.sum((gauss2(lags_ms, *p) - counts) ** 2))
        if sse < best_sse:
            best, best_sse = p, sse
    if np.isfinite(best).all() and best[2] > best[5]:
        best = np.r_[best[3:], best[:3]]
    return best


@register
class CCHGauss2:
    name = "cch_gauss2"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        bin_ms: PositiveFloat = 2.0  # Kumar et al. 2026, Fig. 3e
        max_lag_ms: PositiveFloat = 100.0  # not stated in the paper
        normalization: Literal["counts", "rate_hz"] = "counts"  # NeuroExplorer setting not stated
        min_weight: NonNegativeFloat = 0.0  # no threshold is described in the paper
        min_spikes: PositiveInt = 100

    def estimate(self, trains: SpikeTrains, config: Config | None = None) -> ConnectivityResult:
        cfg = config or self.Config()
        n = trains.n_units
        mask = testable(trains, cfg.min_spikes)
        ccg, edges = cross_correlograms(trains, cfg.bin_ms, cfg.max_lag_ms)
        lags = (edges[:-1] + edges[1:]) / 2
        if cfg.normalization == "rate_hz":  # target spikes per second, conditional on a source spike
            ccg = ccg / (np.maximum(trains.n_spikes(), 1)[:, None, None] * cfg.bin_ms / 1e3)
        weights, delays, p_values, _ = empty_matrices(n)
        fits = np.full((n, n, 6), np.nan)
        fine = np.linspace(lags[0], lags[-1], 20 * lags.size)
        for i, j in zip(*np.nonzero(np.triu(mask, 1))):
            p = fit_gauss2(ccg[i, j], lags, cfg.bin_ms)
            fits[i, j] = p
            if not np.isfinite(p).all():
                continue
            w = (p[0] + p[3]) / (p[2] + p[5])
            peak = float(fine[np.argmax(gauss2(fine, *p))])
            weights[i, j] = weights[j, i] = 0.0  # tested; the weight goes to the net direction
            delays[i, j] = delays[j, i] = np.nan
            if abs(peak) < cfg.bin_ms / 2:
                continue  # peak at zero lag: no net direction
            s, t = (i, j) if peak > 0 else (j, i)
            weights[s, t], delays[s, t] = w, abs(peak)
        with np.errstate(invalid="ignore"):
            significant = mask & (weights > 0) & (weights >= cfg.min_weight)
        return ConnectivityResult(
            method=self.name, node_ids=trains.unit_ids, weights=weights, delays_ms=delays, p_values=p_values,
            significant=significant, directed=True, params=cfg.model_dump(), n_spikes=trains.n_spikes(),
            duration_s=trains.duration_s, positions_um=trains.positions_um, tested_mask=mask,
            extra=dict(ccg=ccg, lag_edges_ms=edges, fit_a1_b1_c1_a2_b2_c2=fits, ranking=np.where(mask, np.nan_to_num(weights), np.nan)),
        )  # fmt: skip
