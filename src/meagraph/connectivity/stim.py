"""Stimulus-evoked connectivity: site -> electrode edges from responses to pulses (DECISIONS.md D30).

Ported from the retired ``stim_connectivity.py``. For every stimulated site ``s`` and readout
electrode ``c``:

1. **Response.** Spikes of ``c`` in ``[floor, response_ms]`` after each of ``s``'s pulses. The
   floor is where detection stopped blanking ``c`` after ``s``'s pulses (pulse + artifact
   recovery + guard), so it is the earliest latency this analysis can ever see; it is reported
   per pair (``extra["floor_ms"]``).
2. **Baseline.** Each pulse's own pre-pulse period, cut into windows of the same length. Windows
   that overlap an earlier pulse or the ``clear_ms`` after it are not used. ``effect`` = mean
   response count - mean baseline count per pulse (spikes per pulse above baseline).
3. **Test, at the level of pulses.** ``k`` = pulses followed by at least one spike in the
   window. Under the null, pulse ``m`` responds with its own baseline probability ``p_m`` (the
   fraction of its baseline windows with a spike, ``(hits + 0.5) / (windows + 1)``), so ``k`` is
   Poisson-binomial and the p-value is its exact upper tail. Every pulse is compared with its own
   recent past, so a pulse delivered during a burst is compared with burst activity. Spike
   counts are not tested directly: one evoked burst gives many spikes that are not independent,
   and a count test with 5 pulses gave p ~ 1e-90 (2026-11-16). Benjamini-Hochberg over all
   site-electrode pairs; edges are positive effects only. A channel that already fires in most
   baseline windows cannot show a response this way (its p_m is near 1): reported, not tested
   for rate increases.
4. **Latency and jitter.** The latency is the first PSTH bin after the floor whose count exceeds
   the baseline bins' mean + 3 SD; a latency equal to the floor is censored (the response may have
   started inside blanking). The jitter is the SD of first-spike latencies over responding pulses.
5. **Indirect paths.** The legacy network-deconvolution step (``A = R (I + R)^-1``, R scaled to
   spectral radius 1/1.05) on the significant effects. It can explain an edge only through other
   *stimulated* sites, because rows exist only for them.

``edge_rule`` decides ``significant``: ``evoked`` (step 3), ``direct`` (also latency <=
``direct_max_ms`` and jitter <= ``max_jitter_ms``, as in the legacy script) or ``deconvolved``
(direct, plus deconvolved weight > ``min_deconvolved``). The other masks are in ``extra``.

Not ported: the legacy script re-detected spikes on artifact-subtracted voltage (median epoch
subtraction, 0.6 ms blanking). Here spikes come from detection with its measured recovery
blanking (D10, D25), so latencies below the floor are not observable. ``meagraph audit`` shows
the recovery per electrode.

Inputs are plain arrays, so this module needs no file reading (tests/test_architecture.py).
:func:`meagraph.connectivity.pipeline.stim_graph_from_detection` wires it to detection results.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt

from meagraph.connectivity.base import ConnectivityResult, empty_matrices
from meagraph.connectivity.stats import fdr_mask
from meagraph.spiketrains import SpikeTrains

METHOD = "stim_evoked"


class StimConnectivityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    response_ms: PositiveFloat = 30.0  # response window ends here after pulse onset
    pre_ms: PositiveFloat = 300.0  # baseline: this long before each pulse...
    drop_pre_ms: NonNegativeFloat = 2.0  # ...ending this long before it (pulse-locked artifact, D26/D28)
    clear_ms: NonNegativeFloat = 200.0  # baseline windows must start this long after any earlier pulse
    psth_bin_ms: PositiveFloat = 1.0
    latency_sd: PositiveFloat = 3.0
    q: PositiveFloat = 0.05
    min_pulses: PositiveInt = 5
    edge_rule: Literal["evoked", "direct", "deconvolved"] = "evoked"
    direct_max_ms: PositiveFloat = 5.0  # legacy direct-edge rule
    max_jitter_ms: PositiveFloat = 1.5
    min_deconvolved: NonNegativeFloat = 0.03


def _counts(t: np.ndarray, starts: np.ndarray, length_s: float) -> np.ndarray:
    return np.searchsorted(t, starts + length_s, "left") - np.searchsorted(t, starts, "left")


def _baseline_starts(onsets: np.ndarray, all_onsets: np.ndarray, length_s: float, cfg: StimConnectivityConfig) -> list[np.ndarray]:
    """Per pulse, start times of the baseline windows of ``length_s`` in its pre-pulse period that
    start at least ``clear_ms`` after every earlier pulse."""
    lo_rel, hi_rel = -cfg.pre_ms / 1e3, -cfg.drop_pre_ms / 1e3
    n = int(np.floor((hi_rel - lo_rel) / length_s + 1e-9))
    out = []
    srt = np.sort(all_onsets)
    for t in onsets:
        starts = t + hi_rel - length_s * np.arange(1, n + 1)  # windows end at -drop_pre, going back
        k = np.searchsorted(srt, t, "left")  # earlier pulses: srt[:k]
        if k:
            starts = starts[starts >= srt[k - 1] + cfg.clear_ms / 1e3]
        out.append(starts)
    return out


def poisson_binomial_sf(k: int, p: np.ndarray) -> float:
    """P(sum of independent Bernoulli(p_m) >= k), exactly (dynamic programming)."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for pm in p:
        dist[1:] = dist[1:] * (1 - pm) + dist[:-1] * pm
        dist[0] *= 1 - pm
    return float(min(1.0, dist[k:].sum()))


def deconvolve(effect: np.ndarray) -> np.ndarray:
    """Legacy network deconvolution of a non-negative (n, n) effect matrix (zero diagonal)."""
    r = np.clip(np.nan_to_num(effect), 0, None)
    np.fill_diagonal(r, 0.0)
    if r.max() <= 0:
        return r
    ev = np.max(np.abs(np.linalg.eigvals(r)))
    rs = r / (ev * 1.05) if ev > 0 else r
    a = np.eye(len(r)) - np.linalg.inv(np.eye(len(r)) + rs)
    np.fill_diagonal(a, 0.0)
    return np.clip(a, 0, None)


def stim_connectivity(
    trains: SpikeTrains,
    onsets_s: np.ndarray,
    sites: Sequence[str],
    floor_ms: np.ndarray,
    readouts: Sequence[str] | None = None,
    config: StimConnectivityConfig | None = None,
    clear_onsets_s: np.ndarray | None = None,
) -> ConnectivityResult:
    """Site -> electrode evoked connectivity.

    ``trains`` holds every node: the readout electrodes and the stimulated sites (sites need not be
    readouts, but they must be units of ``trains`` to have a position). ``onsets_s`` and ``sites``
    give every analysed pulse and its site. ``floor_ms[k, c]`` is the latency floor for the k-th
    site in order of first use and unit ``c`` of ``trains``. ``readouts`` (default: every unit) are
    the tested targets. ``clear_onsets_s`` (default ``onsets_s``) are every pulse of the recording:
    baseline windows must start ``clear_ms`` after each of them, analysed or not (e.g. pulses of
    another stimulus condition).
    """
    cfg = config or StimConnectivityConfig()
    onsets_s = np.asarray(onsets_s, dtype=np.float64)
    all_onsets = onsets_s if clear_onsets_s is None else np.union1d(np.asarray(clear_onsets_s, dtype=np.float64), onsets_s)
    sites = np.asarray(sites, dtype=object)
    ids = list(trains.unit_ids)
    order = list(dict.fromkeys(sites.tolist()))
    readout_idx = [ids.index(r) for r in (readouts if readouts is not None else ids)]
    n = trains.n_units
    weights, latency, p_values, _ = empty_matrices(n)
    jitter, floor, n_used = np.full((n, n), np.nan), np.full((n, n), np.nan), np.zeros((n, n), dtype=int)
    censored = np.zeros((n, n), dtype=bool)
    p_resp, p_base = np.full((n, n), np.nan), np.full((n, n), np.nan)
    end_s = cfg.response_ms / 1e3
    skipped = []
    for k, site in enumerate(order):
        if site not in ids:
            skipped.append(site)
            continue
        i = ids.index(site)
        on = onsets_s[(sites == site) & (onsets_s - cfg.pre_ms / 1e3 >= trains.t_start_s) & (onsets_s + end_s <= trains.t_stop_s)]
        if on.size < cfg.min_pulses:
            skipped.append(site)
            continue
        for c in readout_idx:
            if c == i:
                continue
            f_s = float(floor_ms[k, c]) / 1e3
            floor[i, c] = f_s * 1e3
            length = end_s - f_s
            if length <= 0:
                continue
            base = _baseline_starts(on, all_onsets, length, cfg)
            use = np.array([b.size > 0 for b in base])
            if use.sum() < cfg.min_pulses:
                continue
            t = trains.times_s[c]
            on_u = on[use]
            post = _counts(t, on_u + f_s, length)
            base_counts = [_counts(t, b, length) for b, u in zip(base, use) if u]
            p0 = np.array([((bc > 0).sum() + 0.5) / (bc.size + 1) for bc in base_counts])
            p_values[i, c] = poisson_binomial_sf(int((post > 0).sum()), p0)
            weights[i, c] = post.mean() - float(np.mean([bc.mean() for bc in base_counts]))
            p_resp[i, c], p_base[i, c] = float((post > 0).mean()), float(p0.mean())
            n_used[i, c] = on_u.size
            # latency: first PSTH bin above baseline mean + latency_sd * SD
            b = cfg.psth_bin_ms / 1e3
            nb = max(1, int(np.floor(length / b + 1e-9)))
            edges = f_s + b * np.arange(nb + 1)
            psth = np.array([_counts(t, on_u + e, b).sum() for e in edges[:-1]])
            # Baseline PSTH bins from pulses whose whole pre-pulse period is clear of earlier pulses,
            # scaled to the number of pulses; if too few, a Poisson baseline at the windows' mean rate.
            full = max(bc.size for bc in base_counts)
            clear = np.array([bc.size == full for bc in base_counts])
            if clear.sum() >= cfg.min_pulses:
                pre_edges = -cfg.drop_pre_ms / 1e3 - b * np.arange(1, int(full * length / b + 1e-9) + 1)
                pre = np.array([_counts(t, on_u[clear] + e, b).sum() for e in pre_edges]) * (on_u.size / clear.sum())
                mu, sd = pre.mean(), pre.std()
            else:
                mu = float(sum(bc.mean() for bc in base_counts)) * b / length
                sd = np.sqrt(mu)
            hit = np.flatnonzero(psth > mu + cfg.latency_sd * sd)
            if hit.size:
                latency[i, c] = (edges[hit[0]] + b / 2) * 1e3
                censored[i, c] = hit[0] == 0
            j0 = np.searchsorted(t, on_u + f_s, "left")
            j1 = np.searchsorted(t, on_u + end_s, "left")
            first = np.array([(t[a] - o) * 1e3 for a, z, o in zip(j0, j1, on_u) if z > a])
            if first.size > 1:
                jitter[i, c] = float(np.std(first))
    with np.errstate(invalid="ignore"):
        evoked = fdr_mask(p_values, cfg.q) & (weights > 0)
        direct = evoked & (latency <= cfg.direct_max_ms) & (jitter <= cfg.max_jitter_ms)
    deconv = deconvolve(np.where(evoked, weights, 0.0))
    final = direct & (deconv > cfg.min_deconvolved)
    significant = {"evoked": evoked, "direct": direct, "deconvolved": final}[cfg.edge_rule]
    n_pulses = np.array([int((sites == u).sum()) for u in ids])
    return ConnectivityResult(
        method=METHOD, node_ids=trains.unit_ids, weights=weights, delays_ms=latency, p_values=p_values,
        significant=significant, directed=True, params=cfg.model_dump(), n_spikes=trains.n_spikes(),
        duration_s=trains.duration_s, positions_um=trains.positions_um, signed=False,
        extra=dict(evoked=evoked, direct=direct, deconvolved_mask=final, deconvolved=deconv, jitter_ms=jitter,
                   floor_ms=floor, latency_censored=censored, n_pulses_used=n_used, n_pulses=n_pulses,
                   p_response=p_resp, p_baseline=p_base,
                   skipped_sites=tuple(skipped)),
    )  # fmt: skip
