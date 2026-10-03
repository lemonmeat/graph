"""Stimulation artifacts: blanking windows, per-channel recovery, stimulated-site inference.

All windows are ``[start, stop)`` sample indices from the recording's first sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from spikeinterface.core import BaseRecording

from meagraph.io.mcs_events import StimEvents
from meagraph.preprocess import detection_band, merge_windows


def _to_samples(times_s: np.ndarray, recording: BaseRecording) -> np.ndarray:
    return (np.asarray(times_s) - recording.get_start_time()) * recording.get_sampling_frequency()


def fixed_windows(recording: BaseRecording, stim: Sequence[StimEvents], pre_ms: float, post_ms: float) -> np.ndarray:
    """Legacy ``spikes.py`` rule: ``[t - pre, t + post)`` around every Start and every Stop event."""
    fs = recording.get_sampling_frequency()
    times = [s.onsets_s for s in stim] + [s.offsets_s for s in stim if s.offsets_s is not None]
    t = np.sort(np.concatenate(times)) if times else np.zeros(0)
    a = np.round(_to_samples(t, recording) - pre_ms * 1e-3 * fs).astype(np.int64)
    b = np.round(_to_samples(t, recording) + post_ms * 1e-3 * fs).astype(np.int64)
    return merge_windows(np.column_stack([a, b]), recording.get_num_samples())


def pulse_windows(recording: BaseRecording, stim: StimEvents, pre_ms: float, post_ms: float | np.ndarray) -> np.ndarray:
    """``[onset - pre, offset + post)`` for every pulse; ``post_ms`` may be one value or one per channel.

    Returns ``(n_pulses, 2)`` for a scalar ``post_ms`` and ``(n_channels, n_pulses, 2)`` otherwise.
    """
    fs = recording.get_sampling_frequency()
    offsets = stim.offsets_s if stim.offsets_s is not None else stim.onsets_s
    a = np.round(_to_samples(stim.onsets_s, recording) - pre_ms * 1e-3 * fs).astype(np.int64)
    end = _to_samples(offsets, recording)
    post = np.asarray(post_ms, dtype=np.float64)
    if post.ndim == 0:
        return np.column_stack([a, np.round(end + post * 1e-3 * fs).astype(np.int64)])
    b = np.round(end[None, :] + post[:, None] * 1e-3 * fs).astype(np.int64)
    return np.stack([np.broadcast_to(a, b.shape), b], axis=-1)


@dataclass(frozen=True, eq=False)
class Recovery:
    """Residual artifact after a minimal blank, per channel (DECISIONS.md D10).

    ``recovery_ms[c]`` is measured from pulse offset; NaN means the residual did not settle
    within ``max_post_ms``.
    """

    channel_ids: tuple[str, ...]
    recovery_ms: np.ndarray
    n_epochs: int
    min_post_ms: float
    max_post_ms: float

    def post_ms(self) -> np.ndarray:
        """Per-channel blanking after pulse offset: the recovery time, at least ``min_post_ms``."""
        r = np.where(np.isnan(self.recovery_ms), self.max_post_ms, self.recovery_ms)
        return np.maximum(r, self.min_post_ms)


def measure_recovery(
    recording: BaseRecording,
    stim: StimEvents,
    *,
    pre_ms: float = 1.0,
    min_post_ms: float = 1.0,
    max_post_ms: float = 50.0,
    threshold_sigma: float = 1.0,
    hold_ms: float = 1.0,
    baseline_ms: float = 20.0,
    max_epochs: int = 200,
    band_hz: tuple[float, float] = (300.0, 3000.0),
    filter_order: int = 3,
) -> Recovery:
    """How long the artifact stays visible in the detection band after a minimal blank.

    1. Bridge ``[onset - pre, offset + min_post)`` for every pulse and band-pass (the detection chain).
    2. Take epochs after the offsets of pulses not followed by another pulse within ``max_post_ms``.
    3. The trial *median* is the deterministic artifact; evoked spikes are jittered and absent on
       many trials, so they barely move it.
    4. Recovery is the first time after ``min_post_ms`` from which |median| stays below
       ``threshold_sigma`` single-trial noise sigmas for ``hold_ms``.
    """
    fs = recording.get_sampling_frequency()
    n = recording.get_num_samples()
    filt = detection_band(recording, pulse_windows(recording, stim, pre_ms, min_post_ms), band_hz, filter_order)
    offsets = stim.offsets_s if stim.offsets_s is not None else stim.onsets_s
    next_onset = np.append(stim.onsets_s[1:], np.inf)
    isolated = next_onset - offsets > max_post_ms * 1e-3
    post = int(round(max_post_ms * 1e-3 * fs))
    base = int(round(baseline_ms * 1e-3 * fs))
    off_idx = np.round(_to_samples(offsets, recording)).astype(np.int64)
    on_idx = np.round(_to_samples(stim.onsets_s, recording)).astype(np.int64)
    ok = isolated & (on_idx - base - int(pre_ms * 1e-3 * fs) >= 0) & (off_idx + post <= n)
    chosen = np.flatnonzero(ok)
    if chosen.size > max_epochs:
        chosen = chosen[np.linspace(0, chosen.size - 1, max_epochs).round().astype(int)]
    ids = tuple(str(c) for c in recording.channel_ids)
    if chosen.size < 3:
        return Recovery(ids, np.full(len(ids), np.nan), int(chosen.size), min_post_ms, max_post_ms)

    pre_gap = int(round(pre_ms * 1e-3 * fs)) + 1
    epochs, baselines = [], []
    for i in chosen:
        epochs.append(filt.get_traces(start_frame=off_idx[i], end_frame=off_idx[i] + post))
        baselines.append(filt.get_traces(start_frame=on_idx[i] - pre_gap - base, end_frame=on_idx[i] - pre_gap))
    med = np.abs(np.median(np.stack(epochs), axis=0))  # (samples, channels)
    sigma = np.median(np.abs(np.concatenate(baselines)), axis=0) / 0.6745

    quiet = med <= threshold_sigma * sigma[None, :]
    hold = max(int(round(hold_ms * 1e-3 * fs)), 1)
    first = int(round(min_post_ms * 1e-3 * fs))
    recovery = np.full(len(ids), np.nan)
    for c in range(len(ids)):
        run = np.convolve(quiet[first:, c].astype(np.int64), np.ones(hold, np.int64), mode="valid")
        hits = np.flatnonzero(run == hold)
        if hits.size:
            recovery[c] = (first + hits[0]) / fs * 1e3
    return Recovery(ids, recovery, int(chosen.size), min_post_ms, max_post_ms)


@dataclass(frozen=True)
class SiteInference:
    site: str | None
    reason: str
    peak_uv: dict[str, float]
    saturated: tuple[str, ...]


def infer_site(recording: BaseRecording, stim: StimEvents, n_trials: int = 40, ratio: float = 1.5) -> SiteInference:
    """The stimulating electrode, from artifact amplitude during the pulse.

    Accepted only when it is unambiguous: exactly one channel reaches the ADC rail, or (if none
    does) the largest artifact is at least ``ratio`` times the second largest. Otherwise ``site``
    is None and ``reason`` says why.
    """
    fs = recording.get_sampling_frequency()
    offsets = stim.offsets_s if stim.offsets_s is not None else stim.onsets_s
    keep = np.flatnonzero(stim.onsets_s - recording.get_start_time() > 0.002)[:n_trials]
    ids = [str(c) for c in recording.channel_ids]
    if keep.size == 0:
        return SiteInference(None, "no pulse with a pre-stimulus baseline", {}, ())
    peak = np.zeros(len(ids))
    railed = np.zeros(len(ids), dtype=bool)
    rail = recording.get_property("adc_rail_uv")
    for i in keep:
        a = int(round((stim.onsets_s[i] - recording.get_start_time() - 0.002) * fs))
        b = int(round((offsets[i] - recording.get_start_time()) * fs)) + 2
        x = recording.get_traces(start_frame=a, end_frame=b, return_in_uV=True)
        dev = np.abs(x - np.median(x[: int(0.0015 * fs)], axis=0))
        peak = np.maximum(peak, dev.max(axis=0))
        if rail is not None:
            railed |= (np.abs(x) >= 0.999 * rail[None, :]).any(axis=0)
    order = np.argsort(-peak)
    peaks = {ids[i]: float(peak[i]) for i in order}
    saturated = tuple(ids[i] for i in order if railed[i])
    if len(saturated) == 1:
        return SiteInference(saturated[0], "only channel that reaches the ADC rail", peaks, saturated)
    if len(saturated) > 1:
        return SiteInference(None, f"{len(saturated)} of {len(ids)} channels reach the ADC rail", peaks, saturated)
    if peak[order[0]] >= ratio * peak[order[1]]:
        return SiteInference(ids[order[0]], f"artifact {peak[order[0]] / peak[order[1]]:.1f}x the next largest", peaks, ())
    return SiteInference(None, "no channel's artifact clearly dominates", peaks, ())
