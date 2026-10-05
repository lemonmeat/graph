"""Total spiking probability edges (TSPE; De Blasi et al. 2019): excitatory and inhibitory edges.

TSPE filters each pair's normalized cross-correlation with edge filters: a local peak after the
source spike scores positive (excitatory), a local dip scores negative (inhibitory). The score
and its delay come from Elephant (``elephant.functional_connectivity``, D13), which returns
matrices indexed (target, source); they are transposed here to (source, target).

Elephant gives scores, not significance. Here each score is tested against the same 10 ms
interval-jitter surrogates as ``cch_jitter``: a normal fitted to the surrogate scores gives a
two-sided p-value (D17) and Benjamini-Hochberg selects edges. The weight is the score's excess
over the surrogate mean, so its sign says excitatory (+) or inhibitory (-); the raw score is in
``extra["score"]``. Elephant's experimental ``normalize`` option is not used:
it indexes the delay axis by delay value instead of position.

Known limitation (measured in the benchmark, DECISIONS.md D22): a strong excitatory A -> B makes
B -> A score negative, because the edge filter's leading window sees the A -> B peak at
negative lag. Such reverse "inhibitory" edges are counted separately (``inh_fp_reverse``).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import quantities as pq
from neo import SpikeTrain
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from scipy.stats import norm

from meagraph.connectivity.base import ConnectivityResult, empty_matrices, register, testable
from meagraph.connectivity.stats import fdr_mask, interval_jitter
from meagraph.spiketrains import SpikeTrains


def tspe_scores(times_s, t_start: float, t_stop: float, bin_ms: float, max_delay_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Elephant's TSPE on a list of spike-time arrays; returns (score, delay in bins), (source, target)."""
    from elephant.conversion import BinnedSpikeTrain
    from elephant.functional_connectivity import total_spiking_probability_edges

    trains = [SpikeTrain(np.asarray(t) * pq.s, t_start=t_start * pq.s, t_stop=t_stop * pq.s) for t in times_s]
    # Elephant logs every float rounding it corrects at bin edges; its loggers are named by file path.
    logs = [logging.getLogger(n) for n in list(logging.root.manager.loggerDict) if "elephant" in n]
    levels = [lg.level for lg in logs]
    for lg in logs:
        lg.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # quantities' deprecated copy= inside Elephant
            binned = BinnedSpikeTrain(trains, bin_size=bin_ms * pq.ms)
            score, delay = total_spiking_probability_edges(binned, max_delay=max_delay_bins)
    finally:
        for lg, level in zip(logs, levels):
            lg.setLevel(level)
    return np.asarray(score).T, np.asarray(delay).reshape(score.shape).T


@register
class TSPE:
    name = "tspe"
    signed = True

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        bin_ms: PositiveFloat = 1.0  # the bin size Elephant's default filters were tuned for
        max_delay_ms: PositiveFloat = 25.0
        jitter_window_ms: PositiveFloat = 10.0
        n_surrogates: PositiveInt = 1000
        q: PositiveFloat = 0.05
        min_spikes: PositiveInt = 100
        seed: int = 0

    def estimate(self, trains: SpikeTrains, config: Config | None = None) -> ConnectivityResult:
        cfg = config or self.Config()
        n, t0, t1 = trains.n_units, trains.t_start_s, trains.t_stop_s
        mask = testable(trains, cfg.min_spikes)
        max_delay = int(round(cfg.max_delay_ms / cfg.bin_ms))
        weights, delays, p_values, _ = empty_matrices(n)
        extra: dict = {}
        if mask.any():
            score, delay_bins = tspe_scores(trains.times_s, t0, t1, cfg.bin_ms, max_delay)
            rng = np.random.default_rng(cfg.seed)
            sur = [interval_jitter(t, cfg.jitter_window_ms / 1e3, t0, t1, rng, cfg.n_surrogates) for t in trains.times_s]
            null = np.stack([tspe_scores([s[k] for s in sur], t0, t1, cfg.bin_ms, max_delay)[0] for k in range(cfg.n_surrogates)], axis=-1)
            mean, sd = null.mean(axis=-1), null.std(axis=-1, ddof=1)
            z = np.where(sd > 0, (score - mean) / np.where(sd > 0, sd, 1.0), 0.0)
            p_exc, p_inh = norm.sf(z), norm.cdf(z)
            p_values[mask] = np.minimum(1.0, 2 * np.minimum(p_exc, p_inh))[mask]
            weights[mask] = (score - mean)[mask]
            delays[mask] = (delay_bins * cfg.bin_ms)[mask]
            extra = dict(score=np.where(mask, score, np.nan), z=np.where(mask, z, np.nan),
                         p_excitatory=np.where(mask, p_exc, np.nan), p_inhibitory=np.where(mask, p_inh, np.nan))  # fmt: skip
        significant = fdr_mask(p_values, cfg.q)
        return ConnectivityResult(
            method=self.name, node_ids=trains.unit_ids, weights=weights, delays_ms=delays, p_values=p_values,
            significant=significant, directed=True, params=cfg.model_dump(), n_spikes=trains.n_spikes(),
            duration_s=trains.duration_s, positions_um=trains.positions_um, extra=extra, signed=True,
        )  # fmt: skip
