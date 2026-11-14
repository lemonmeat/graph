"""Stimulus-evoked responses per stimulation site (DECISIONS.md D28).

Three questions, asked of every site in a recording with per-pulse sites (a stimulation log):

1. **Does the site drive the network?** The fraction of its pulses followed by a network-burst
   onset in the response window, against the same fraction at sham times (moments with no pulse
   nearby) in the same network state. States: ``in_burst`` (the pulse lands inside a burst),
   ``active`` (the last burst ended less than ``active_gap_s`` ago) and ``silent``.
2. **Which electrodes respond?** Spike counts per electrode in the response window after the
   site's pulses against sham times, both outside bursts, plus first-spike latency and PSTHs.
3. **Can the sites be told apart?** A cross-validated multinomial logistic regression predicts
   the stimulated site (and, in a second task, "sham") from per-electrode spike counts in a few
   bins after each pulse. Chance and significance come from label permutation. For closed-loop
   training, decodable sites are candidate inputs and the electrodes that separate them are
   candidate readouts.

Blanking: detection removes each electrode's spikes until its artifact has recovered, and a
stimulated electrode's spikes for ``recovery_max_post_ms`` after its own pulses (D25). If that
gap differed between conditions, the decoder could identify a site from the blanking alone. So
each electrode's response window starts at the latest blanking end over every pulse and every
site, and the same window is used for pulses and shams.

Spike times come from a :class:`~meagraph.detect.threshold.DetectionResult`; nothing reads the
recording itself.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, model_validator
from scipy import optimize, stats

from meagraph.config import write_run_folder
from meagraph.detect.bursts import BurstConfig, network_bursts
from meagraph.detect.threshold import DetectionResult
from meagraph.intervals import inside, merge_intervals
from meagraph.io.mcs_events import StimEvents
from meagraph.spiketrains import SpikeTrains

STATES = ("silent", "active", "in_burst")


class EvokedConfig(BaseModel):
    """Every parameter of the evoked-response analysis. Saved with each result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channels: Literal["active", "all"] = "active"  # readout electrodes: QC-active only (D10)
    # Response window = [first edge, last edge] after pulse onset; the edges are also the decoder's
    # time bins. 12 ms clears blanking on most electrodes; at 28 bursts/min a burst starts by chance
    # within 12-150 ms of a random time only ~6 % of the time.
    bins_ms: tuple[PositiveFloat, ...] = (12.0, 30.0, 60.0, 100.0, 150.0)
    psth_ms: tuple[float, float] = (-100.0, 300.0)
    psth_bin_ms: PositiveFloat = 5.0
    # A burst counts as evoked if it starts from pulse onset to the window end: blanking hides the
    # first ~10 ms of spikes, not the burst, and evoked bursts often start at 9-12 ms.
    burst_from_ms: float = 0.0
    # Spikes this long before each pulse are dropped first. In the 2026-11-12 recordings every
    # electrode has crossings 1.0-1.5 ms before onset and none in the 4.5 ms before that: pulse-locked
    # artifact at the edge of the blanked window (D26). Kept, they start a "burst" just before the
    # pulse, which makes evoked bursts look like pulses delivered during a burst.
    drop_pre_ms: NonNegativeFloat = 2.0
    active_gap_s: PositiveFloat = 5.0  # "active" if the last burst ended less than this long before the pulse
    sham_step_s: PositiveFloat = 0.25  # sham times: a grid over the recording...
    sham_clear_s: PositiveFloat = 1.0  # ...at least this long after any pulse, and with none in the window
    n_folds: PositiveInt = 5
    n_repeats: PositiveInt = 5  # cross-validation repeated over this many random fold splits, then averaged
    n_permutations: NonNegativeInt = 200
    n_pair_permutations: NonNegativeInt = 100  # for the pairwise site-vs-site decoding matrix
    l2: PositiveFloat = 1.0  # ridge penalty of the decoder, on standardized features
    min_trials: PositiveInt = 5  # sites with fewer pulses are reported but not decoded or ranked
    alpha: PositiveFloat = 0.05
    seed: int = 0
    bursts: BurstConfig = BurstConfig()

    @model_validator(mode="after")
    def _check(self) -> EvokedConfig:
        if len(self.bins_ms) < 2 or any(b <= a for a, b in zip(self.bins_ms, self.bins_ms[1:])):
            raise ValueError("bins_ms needs at least two increasing edges")
        if not self.psth_ms[1] > self.psth_ms[0]:
            raise ValueError("psth_ms must be increasing")
        return self

    @property
    def window_ms(self) -> tuple[float, float]:
        return self.bins_ms[0], self.bins_ms[-1]


@dataclass(frozen=True, eq=False)
class Trials:
    """Pulses and sham times, with what followed each."""

    onset_s: np.ndarray
    label: np.ndarray  # stimulated site, or "sham"
    source: np.ndarray  # STG output, "" for shams
    state: np.ndarray  # one of STATES
    since_burst_s: np.ndarray  # time since the last burst ended (inf if none yet; 0 inside a burst)
    burst_latency_ms: np.ndarray  # onset of a burst starting in the window, NaN if none
    counts: np.ndarray  # (n_trials, n_channels, n_bins) spikes per electrode and decoder bin
    first_ms: np.ndarray  # (n_trials, n_channels) first spike in the window, NaN if none

    @property
    def is_sham(self) -> np.ndarray:
        return self.label == "sham"

    @property
    def evoked(self) -> np.ndarray:
        return np.isfinite(self.burst_latency_ms)

    def subset(self, mask: np.ndarray) -> Trials:
        return Trials(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))


@dataclass(frozen=True, eq=False)
class EvokedResult:
    config: EvokedConfig
    channels: tuple[str, ...]
    start_ms: np.ndarray  # per channel: where its response window starts (after blanking)
    sites: tuple[str, ...]  # stimulated sites, in order of first use
    pulses: Trials
    shams: Trials  # the full sham grid
    bursts: np.ndarray  # (k, 2) network bursts, s
    n_dropped_pre: int  # spikes removed just before pulses (``drop_pre_ms``), readout and burst channels
    site_table: list[dict]  # per site and state: evoked-burst probability vs sham
    response_table: list[dict]  # per site and channel: spike response vs sham
    channel_table: list[dict]  # per channel: how well it separates sites
    decoding: dict  # per task: balanced accuracy, chance, permutation p, confusion matrix
    psth: dict[str, np.ndarray]  # "edges_ms", and per label (site or "sham") rates (n_channels, n_bins) in Hz


# -- trials ---------------------------------------------------------------------------------- #
def pulse_events(stim: Sequence[StimEvents]) -> list[StimEvents]:
    """The pulse events. Marker events repeat an output's pulse times, so they are used only for
    outputs without Single Pulse events."""
    with_pulses = {s.source for s in stim if s.kind == "Single Pulse"}
    return [s for s in stim if s.kind == "Single Pulse" or s.source not in with_pulses]


def _states(times: np.ndarray, bursts: np.ndarray, active_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    merged = merge_intervals(bursts)
    in_burst = inside(times, merged)
    k = np.searchsorted(merged[:, 1], times, side="right") - 1 if merged.size else np.full(times.shape, -1)
    since = np.where(k >= 0, times - merged[np.maximum(k, 0), 1] if merged.size else np.inf, np.inf)
    since = np.where(in_burst, 0.0, since)
    state = np.where(in_burst, "in_burst", np.where(since < active_gap_s, "active", "silent"))
    return state.astype(object), since


def _trials(times, labels, sources, trains: SpikeTrains, start_ms, bursts, cfg: EvokedConfig) -> Trials:
    times = np.asarray(times, dtype=np.float64)
    edges = np.asarray(cfg.bins_ms) / 1e3
    lo, hi = edges[0], edges[-1]
    counts = np.zeros((times.size, trains.n_units, edges.size - 1), dtype=np.int64)
    first = np.full((times.size, trains.n_units), np.nan)
    for c, t in enumerate(trains.times_s):
        begin = max(lo, start_ms[c] / 1e3)
        cuts = np.maximum(edges, begin)  # bins before the window start are empty for every trial
        idx = np.searchsorted(t, times[:, None] + cuts[None, :], side="left")
        counts[:, c] = np.diff(idx, axis=1)
        i0, i1 = idx[:, 0], idx[:, -1]
        has = i1 > i0
        first[has, c] = (t[i0[has]] - times[has]) * 1e3
    state, since = _states(times, bursts, cfg.active_gap_s)
    latency = np.full(times.size, np.nan)
    if bursts.size:
        b0 = np.sort(bursts[:, 0])
        j = np.searchsorted(b0, times + cfg.burst_from_ms / 1e3, side="left")
        ok = j < b0.size
        ok[ok] &= b0[j[ok]] <= times[ok] + hi
        latency[ok] = (b0[j[ok]] - times[ok]) * 1e3
    return Trials(times, np.asarray(labels, dtype=object), np.asarray(sources, dtype=object), state, since, latency, counts, first)


def sham_times(onsets: np.ndarray, t_start_s: float, t_stop_s: float, cfg: EvokedConfig) -> np.ndarray:
    """Grid times at least ``sham_clear_s`` after any pulse and with no pulse in their window."""
    hi = cfg.window_ms[1] / 1e3
    pre = max(-cfg.psth_ms[0] / 1e3, 0.0)
    grid = np.arange(t_start_s + pre, t_stop_s - max(hi, cfg.psth_ms[1] / 1e3), cfg.sham_step_s)
    if onsets.size == 0:
        return grid
    on = np.sort(onsets)
    k = np.searchsorted(on, grid, side="right")  # pulses at or before each grid time: on[:k]
    before = np.where(k > 0, grid - on[np.maximum(k - 1, 0)], np.inf)
    after = np.where(k < on.size, on[np.minimum(k, on.size - 1)] - grid, np.inf)
    return grid[(before >= cfg.sham_clear_s) & (after > hi)]


# -- statistics ------------------------------------------------------------------------------ #
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaN stays NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        v = p[ok]
        order = np.argsort(v)
        ranked = v[order] * v.size / np.arange(1, v.size + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty_like(v)
        out[order] = np.minimum(ranked, 1.0)
        q[ok] = out
    return q


def _site_table(sites, pulses: Trials, shams: Trials) -> list[dict]:
    """Evoked-burst probability per site and state against shams in the same state. For "all",
    the sham rate is weighted by the site's own mix of states, so a site that happened to be
    stimulated mostly in active periods is compared with active-period shams."""
    sham_rate = {s: (float(shams.evoked[shams.state == s].mean()) if (shams.state == s).any() else float("nan")) for s in STATES}
    rows = []
    for site in sites:
        mine = pulses.subset(pulses.label == site)
        for state in ("all", *STATES):
            sel = mine if state == "all" else mine.subset(mine.state == state)
            n, k = len(sel.onset_s), int(sel.evoked.sum())
            if state == "all":
                w = np.array([(sel.state == s).sum() for s in STATES], dtype=float)
                r = np.array([sham_rate[s] for s in STATES])
                p0 = float(np.nansum(w * r) / w[np.isfinite(r)].sum()) if n and np.isfinite(r[w > 0]).all() else float("nan")
            else:
                p0 = sham_rate[state]
            p = float(stats.binomtest(k, n, min(max(p0, 1e-9), 1 - 1e-9), alternative="greater").pvalue) if n and np.isfinite(p0) else float("nan")
            lo, hi = wilson(k, n)
            lat = sel.burst_latency_ms[sel.evoked]
            rows.append({
                "site": site, "state": state, "n_pulses": n, "n_evoked": k,
                "p_evoked": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi,
                "p_sham": p0, "n_sham": int(len(shams.onset_s) if state == "all" else (shams.state == state).sum()),
                "p_value": p, "median_latency_ms": float(np.median(lat)) if lat.size else float("nan"),
            })  # fmt: skip
    return rows


def _response_table(sites, channels, pulses: Trials, shams: Trials) -> list[dict]:
    """Spikes per electrode in the window, pulses vs shams, both outside bursts (Mann-Whitney)."""
    quiet = shams.subset(shams.state != "in_burst")
    sham_total = quiet.counts.sum(axis=2)
    rows = []
    for site in sites:
        mine = pulses.subset((pulses.label == site) & (pulses.state != "in_burst"))
        total = mine.counts.sum(axis=2)
        for c, ch in enumerate(channels):
            x, y = total[:, c], sham_total[:, c]
            if x.size and y.size and (x.any() or y.any()):
                p = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
            else:
                p = float("nan")
            lat = mine.first_ms[:, c]
            rows.append({
                "site": site, "channel": ch, "n_pulses": int(x.size),
                "mean_spikes": float(x.mean()) if x.size else float("nan"),
                "mean_spikes_sham": float(y.mean()) if y.size else float("nan"),
                "p_any": float((x > 0).mean()) if x.size else float("nan"),
                "p_any_sham": float((y > 0).mean()) if y.size else float("nan"),
                "median_first_ms": float(np.nanmedian(lat)) if np.isfinite(lat).any() else float("nan"),
                "p_value": p,
            })  # fmt: skip
    q = bh(np.array([r["p_value"] for r in rows]))
    for r, v in zip(rows, q):
        r["q_value"] = float(v)
    return rows


def _channel_table(decodable, channels, start_ms, pulses: Trials) -> list[dict]:
    """How differently each electrode responds to the decodable sites (Kruskal-Wallis on window
    spike counts, pulses outside bursts)."""
    rows = []
    for c, ch in enumerate(channels):
        groups = [pulses.counts[(pulses.label == s) & (pulses.state != "in_burst"), c].sum(axis=1) for s in decodable]
        groups = [g for g in groups if g.size]
        h, p = float("nan"), float("nan")
        if len(groups) >= 2 and np.unique(np.concatenate(groups)).size > 1:
            h, p = (float(v) for v in stats.kruskal(*groups))
        rows.append({"channel": ch, "window_start_ms": float(start_ms[c]), "kruskal_h": h, "p_value": p})
    q = bh(np.array([r["p_value"] for r in rows]))
    for r, v in zip(rows, q):
        r["q_value"] = float(v)
    return rows


# -- decoding -------------------------------------------------------------------------------- #
def _fit_logistic(x: np.ndarray, y: np.ndarray, n_classes: int, l2: float) -> np.ndarray:
    """Multinomial logistic regression with a ridge penalty (not on the intercepts). Returns
    ``(n_features + 1, n_classes)`` weights, the last row being the intercepts."""
    xa = np.column_stack([x, np.ones(len(x))])
    onehot = np.eye(n_classes)[y]
    pen = np.ones((xa.shape[1], 1))
    pen[-1] = 0.0

    def loss(w):
        w = w.reshape(xa.shape[1], n_classes)
        z = xa @ w
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        grad = xa.T @ (np.exp(logp) - onehot) + l2 * pen * w
        return -(onehot * logp).sum() + 0.5 * l2 * float((pen * w * w).sum()), grad.ravel()

    w0 = np.zeros(xa.shape[1] * n_classes)
    return optimize.minimize(loss, w0, jac=True, method="L-BFGS-B").x.reshape(xa.shape[1], n_classes)


def _stratified_folds(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    fold = np.empty(y.size, dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = np.arange(idx.size) % k
    return fold


def _cv_predict(x: np.ndarray, y: np.ndarray, n_classes: int, folds: np.ndarray, l2: float) -> np.ndarray:
    pred = np.empty_like(y)
    for f in np.unique(folds):
        test = folds == f
        mu, sd = x[~test].mean(axis=0), x[~test].std(axis=0)
        sd[sd == 0] = 1.0
        w = _fit_logistic((x[~test] - mu) / sd, y[~test], n_classes, l2)
        z = np.column_stack([(x[test] - mu) / sd, np.ones(test.sum())]) @ w
        pred[test] = np.argmax(z, axis=1)
    return pred


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(n_classes) if (y == c).any()]))


def _repeated_cv(x, y, n_classes, k, cfg: EvokedConfig, rng) -> tuple[float, np.ndarray]:
    """Balanced accuracy averaged over ``n_repeats`` fold splits, and the summed confusion matrix."""
    accs, confusion = [], np.zeros((n_classes, n_classes), dtype=int)
    for _ in range(cfg.n_repeats):
        pred = _cv_predict(x, y, n_classes, _stratified_folds(y, k, rng), cfg.l2)
        accs.append(balanced_accuracy(y, pred, n_classes))
        np.add.at(confusion, (y, pred), 1)
    return float(np.mean(accs)), confusion


def decode(features: np.ndarray, labels: np.ndarray, cfg: EvokedConfig, rng: np.random.Generator) -> dict:
    """Cross-validated balanced accuracy of predicting ``labels`` from ``features`` (repeated
    stratified k-fold), with a label permutation test that repeats the whole procedure. Features
    are sqrt spike counts, standardized within each training fold."""
    classes = sorted(set(labels.tolist()), key=lambda s: (s == "sham", s))
    y = np.array([classes.index(v) for v in labels])
    n_per = np.bincount(y, minlength=len(classes))
    k = int(min(cfg.n_folds, n_per.min()))
    out = {"classes": classes, "n_trials": n_per.tolist(), "chance": 1.0 / len(classes)}
    if len(classes) < 2 or k < 2:
        return {**out, "skipped": "needs at least 2 classes with at least 2 trials each"}
    x = np.sqrt(features.reshape(len(features), -1).astype(float))
    n = len(classes)
    acc, confusion = _repeated_cv(x, y, n, k, cfg, rng)
    null = np.array([_repeated_cv(x, rng.permutation(y), n, k, cfg, rng)[0] for _ in range(cfg.n_permutations)])
    return {
        **out, "n_folds": k, "n_repeats": cfg.n_repeats, "balanced_accuracy": acc,
        "null_mean": float(null.mean()) if null.size else float("nan"),
        "p_value": float((1 + (null >= acc).sum()) / (1 + null.size)) if null.size else float("nan"),
        "recall": {c: float(confusion[i, i] / confusion[i].sum()) for i, c in enumerate(classes)},
        "confusion": confusion.tolist(),
    }  # fmt: skip


# -- analysis -------------------------------------------------------------------------------- #
def evoked_responses(
    trains: SpikeTrains,
    pulses: Sequence[StimEvents],
    bursts: np.ndarray,
    start_ms: Sequence[float] | None = None,
    config: EvokedConfig | None = None,
) -> EvokedResult:
    """Evoked responses of ``trains`` (the readout electrodes) to every pulse in ``pulses``.

    A pulse's class is its site; pulses of an output with no known site are classed by the output
    ("STG 1"). ``start_ms`` is where each electrode's window starts (after blanking); by default
    the first bin edge. ``bursts`` are network bursts on the whole array.
    """
    cfg = config or EvokedConfig()
    start = np.full(trains.n_units, cfg.window_ms[0]) if start_ms is None else np.maximum(np.asarray(start_ms, float), cfg.window_ms[0])
    times, labels, sources = [], [], []
    for s in pulses:
        sites = s.sites if s.sites is not None else np.full(s.n, "", dtype=object)
        times.append(s.onsets_s)
        labels += [site or s.source for site in sites]
        sources += [s.source] * s.n
    times = np.concatenate(times) if times else np.zeros(0)
    order = np.argsort(times, kind="stable")
    times, labels, sources = times[order], np.array(labels, dtype=object)[order], np.array(sources, dtype=object)[order]
    keep = (times + cfg.window_ms[1] / 1e3 <= trains.t_stop_s) & (times >= trains.t_start_s)
    times, labels, sources = times[keep], labels[keep], sources[keep]
    bursts = np.asarray(bursts, dtype=float).reshape(-1, 2)

    pulse_trials = _trials(times, labels, sources, trains, start, bursts, cfg)
    grid = sham_times(times, trains.t_start_s, trains.t_stop_s, cfg)
    sham_trials = _trials(grid, ["sham"] * grid.size, [""] * grid.size, trains, start, bursts, cfg)

    sites = tuple(dict.fromkeys(labels.tolist()))
    channels = trains.unit_ids
    n_site = {s: int((labels == s).sum()) for s in sites}
    decodable = [s for s in sites if n_site[s] >= cfg.min_trials]

    rng = np.random.default_rng(cfg.seed)
    decoding = {}
    use = np.isin(pulse_trials.label, decodable)
    decoding["sites"] = decode(pulse_trials.counts[use], pulse_trials.label[use], cfg, rng)
    if decodable and sham_trials.onset_s.size:
        n_sham = min(int(np.median([n_site[s] for s in decodable])), sham_trials.onset_s.size)
        pick = rng.choice(sham_trials.onset_s.size, n_sham, replace=False)
        feats = np.concatenate([pulse_trials.counts[use], sham_trials.counts[pick]])
        labs = np.concatenate([pulse_trials.label[use], sham_trials.label[pick]])
        decoding["sites_and_sham"] = decode(feats, labs, cfg, rng)
    # Readouts that were never stimulated: a stimulated electrode often fires after its own pulses
    # without a network burst, which identifies that site but needs the input electrode as a readout.
    offsite = np.array([c not in set(sites) for c in channels])
    if offsite.any() and not offsite.all():
        d = decode(pulse_trials.counts[use][:, offsite], pulse_trials.label[use], cfg, rng)
        decoding["sites_unstimulated_readouts"] = {**d, "readouts": [c for c, k in zip(channels, offsite) if k]}
    # Which pairs of sites can be told apart (on unstimulated readouts where there are any): for a
    # set of distinguishable inputs, every pair in it should decode above chance.
    feats = pulse_trials.counts[:, offsite] if offsite.any() and not offsite.all() else pulse_trials.counts
    pair_cfg = cfg.model_copy(update={"n_permutations": cfg.n_pair_permutations})
    pairs = []
    for i, a in enumerate(decodable):
        for b in decodable[i + 1 :]:
            m = np.isin(pulse_trials.label, [a, b])
            d = decode(feats[m], pulse_trials.label[m], pair_cfg, rng)
            pairs.append({"a": a, "b": b, "balanced_accuracy": d.get("balanced_accuracy", float("nan")),
                          "p_value": d.get("p_value", float("nan"))})  # fmt: skip
    q = bh(np.array([r["p_value"] for r in pairs]))
    for r, v in zip(pairs, q):
        r["q_value"] = float(v)
    decoding["site_pairs"] = pairs

    edges = np.arange(cfg.psth_ms[0], cfg.psth_ms[1] + cfg.psth_bin_ms / 2, cfg.psth_bin_ms)
    psth = {"edges_ms": edges}
    for label in (*sites, "sham"):
        on = (sham_trials if label == "sham" else pulse_trials).onset_s
        on = on if label == "sham" else on[pulse_trials.label == label]
        rates = np.zeros((trains.n_units, edges.size - 1))
        for c, t in enumerate(trains.times_s):
            rel = (t[None, :] - on[:, None]).ravel() * 1e3 if on.size else np.zeros(0)
            rates[c] = np.histogram(rel, edges)[0] / max(on.size, 1) / (cfg.psth_bin_ms / 1e3)
        psth[label] = rates

    return EvokedResult(
        config=cfg, channels=channels, start_ms=start, sites=sites, pulses=pulse_trials, shams=sham_trials,
        bursts=bursts, n_dropped_pre=0, site_table=_site_table(sites, pulse_trials, sham_trials),
        response_table=_response_table(sites, channels, pulse_trials, sham_trials),
        channel_table=_channel_table(decodable, channels, start, pulse_trials), decoding=decoding, psth=psth,
    )  # fmt: skip


def blanked_until_ms(detection: DetectionResult, channels: Sequence[str]) -> np.ndarray:
    """Per channel, the latest time after pulse onset (ms) at which detection may still have
    removed its spikes, over every output and site: pulse duration + artifact recovery (or the
    maximum, for unrecovered channels and for stimulated electrodes) + guard."""
    cfg = detection.config
    pulses = pulse_events(detection.stim)
    out = np.zeros(len(channels))
    for s in pulses:
        dur_ms = float(np.max(s.durations_s)) * 1e3 if s.durations_s is not None and s.n else 0.0
        recovery = detection.recovery_ms.get(s.source, {})
        sites = set(s.sites.tolist()) if s.sites is not None else set()
        for c, ch in enumerate(channels):
            r = recovery.get(ch, float("nan"))
            post = cfg.recovery_max_post_ms if (ch in sites or not np.isfinite(r)) else max(r, cfg.recovery_min_post_ms)
            out[c] = max(out[c], dur_ms + post + cfg.guard_ms)
    return out


def evoked_from_detection(detection: DetectionResult, config: EvokedConfig | None = None) -> EvokedResult:
    """Evoked responses on a detection result's readout channels. Network bursts are detected on
    every non-excluded channel, as in :func:`meagraph.connectivity.pipeline.graphs_from_detection`,
    but on all spikes: stimulation periods are kept, since evoked bursts are the point here."""
    cfg = config or EvokedConfig()
    usable = [c for c in detection.trains.unit_ids if c not in detection.excluded]
    chosen = list(detection.active_channels) if cfg.channels == "active" else usable
    pulses = pulse_events(detection.stim)
    if not pulses:
        raise ValueError("no stimulation pulses in this detection result")
    base = detection.trains.select(usable)
    dropped = 0
    if cfg.drop_pre_ms > 0:
        on = np.concatenate([s.onsets_s for s in pulses])
        pre = np.column_stack([on - cfg.drop_pre_ms / 1e3, on - 1e-9])
        cleaned = base.without_periods(pre)
        dropped = int(base.n_spikes().sum() - cleaned.n_spikes().sum())
        base = cleaned
    bursts = network_bursts(base, cfg.bursts)
    res = evoked_responses(base.select(chosen), pulses, bursts, blanked_until_ms(detection, chosen), cfg)
    return replace(res, n_dropped_pre=dropped)


# -- files ----------------------------------------------------------------------------------- #
def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def save_evoked(result: EvokedResult, folder: str | Path, inputs=(), overwrite: bool = False) -> Path:
    """``folder/``: ``sites.csv``, ``responses.csv``, ``channels.csv``, ``trials.csv``,
    ``decoding.json``, ``psth.npz``, ``network_bursts.csv``, ``config.yaml``, ``provenance.json``."""
    out = write_run_folder(folder, result.config, inputs=inputs, overwrite=overwrite)
    _write_csv(out / "sites.csv", result.site_table)
    _write_csv(out / "responses.csv", result.response_table)
    _write_csv(out / "channels.csv", result.channel_table)
    p = result.pulses
    total = p.counts.sum(axis=2)
    _write_csv(out / "trials.csv", [
        {"onset_s": float(p.onset_s[i]), "site": p.label[i], "source": p.source[i], "state": p.state[i],
         "since_burst_s": float(p.since_burst_s[i]), "burst_latency_ms": float(p.burst_latency_ms[i]),
         **{f"spikes_{ch}": int(total[i, c]) for c, ch in enumerate(result.channels)}}
        for i in range(len(p.onset_s))
    ])  # fmt: skip
    (out / "decoding.json").write_text(json.dumps(result.decoding, indent=2) + "\n")
    np.savez_compressed(out / "psth.npz", channels=np.array(result.channels, dtype=str),
                        labels=np.array([k for k in result.psth if k != "edges_ms"], dtype=str),
                        **{k if k == "edges_ms" else f"rate_{k}": v for k, v in result.psth.items()})  # fmt: skip
    np.savetxt(out / "network_bursts.csv", result.bursts, delimiter=",", header="start_s,stop_s", comments="", fmt="%.4f")
    return out
