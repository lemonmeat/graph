"""Spike time tiling coefficient (Cutts & Eglen 2014): undirected, rate-robust co-firing.

STTC at ``dt_ms`` is tested against the same interval-jitter surrogates as ``cch_jitter``, so a
significant value means fine-timescale co-firing beyond what slower co-modulation explains.
STTC at ``dt_descriptive_ms`` (burst scale) is reported as a weight only, not tested.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from meagraph.connectivity.base import ConnectivityResult, empty_matrices, register, testable
from meagraph.connectivity.stats import fdr_mask, interval_jitter, normal_tail_p, sttc, surrogate_p
from meagraph.spiketrains import SpikeTrains


@register
class STTC:
    name = "sttc"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        dt_ms: PositiveFloat = 5.0
        dt_descriptive_ms: PositiveFloat = 50.0
        jitter_window_ms: PositiveFloat = 10.0
        n_surrogates: PositiveInt = 1000
        q: PositiveFloat = 0.05
        min_spikes: PositiveInt = 100
        seed: int = 0

    def estimate(self, trains: SpikeTrains, config: Config | None = None) -> ConnectivityResult:
        cfg = config or self.Config()
        n, t0, t1 = trains.n_units, trains.t_start_s, trains.t_stop_s
        mask = testable(trains, cfg.min_spikes) & np.triu(np.ones((n, n), dtype=bool), 1)
        rng = np.random.default_rng(cfg.seed)
        used = np.flatnonzero(mask.any(axis=0) | mask.any(axis=1))
        surrogates = {i: interval_jitter(trains.times_s[i], cfg.jitter_window_ms / 1e3, t0, t1, rng, cfg.n_surrogates) for i in used}

        weights, _, p_values, _ = empty_matrices(n)
        coarse, p_empirical = np.full((n, n), np.nan), np.full((n, n), np.nan)
        dt, dt_coarse = cfg.dt_ms / 1e3, cfg.dt_descriptive_ms / 1e3
        for i, j in zip(*np.nonzero(mask)):
            a, b = trains.times_s[i], trains.times_s[j]
            observed = sttc(a, b, dt, t0, t1)
            null = [sttc(surrogates[i][k], surrogates[j][k], dt, t0, t1) for k in range(cfg.n_surrogates)]
            weights[i, j] = weights[j, i] = observed
            p_values[i, j] = p_values[j, i] = normal_tail_p(observed, null, "greater")
            p_empirical[i, j] = p_empirical[j, i] = surrogate_p(observed, null, "greater")
            coarse[i, j] = coarse[j, i] = sttc(a, b, dt_coarse, t0, t1)

        upper = np.where(np.triu(np.ones((n, n), dtype=bool), 1), p_values, np.nan)
        significant = fdr_mask(upper, cfg.q)  # one test per unordered pair
        significant |= significant.T
        return ConnectivityResult(
            method=self.name, node_ids=trains.unit_ids, weights=weights, delays_ms=np.full((n, n), np.nan),
            p_values=p_values, significant=significant, directed=False, params=cfg.model_dump(),
            n_spikes=trains.n_spikes(), duration_s=trains.duration_s, positions_um=trains.positions_um,
            extra=dict(sttc_descriptive=coarse, p_empirical=p_empirical),
        )  # fmt: skip
