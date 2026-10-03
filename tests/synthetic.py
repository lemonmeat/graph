"""Synthetic recordings with known spikes and stimulation artifacts, for detection tests."""

from __future__ import annotations

import numpy as np
from spikeinterface.core import NumpyRecording

from meagraph.io.mcs_events import StimEvents

FS = 10_000.0
SPIKE = -np.array([0.1, 0.3, 0.7, 1.0, 0.8, 0.4, 0.1, -0.15, -0.25, -0.2, -0.1, -0.05])  # trough at index 3


def make_recording(
    n_channels: int = 3,
    duration_s: float = 20.0,
    noise_uv: float = 5.0,
    spike_channel: int = 0,
    spike_uv: float = 60.0,
    spike_rate_hz: float = 5.0,
    stim_onsets_s=(),
    pulse_ms: float = 2.0,
    artifact_uv: float = 20_000.0,
    tail_uv: float = 3_000.0,
    tail_ms: float = 3.0,
    seed: int = 0,
):
    """Gaussian noise, one channel with spikes of known times, optional stimulation artifacts.

    Returns ``(recording, spike_samples, stim)``: spike troughs as sample indices on
    ``spike_channel``, and a ``StimEvents`` (or None) for the pulses.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    x = rng.normal(0.0, noise_uv, size=(n, n_channels))
    gap = int(FS / spike_rate_hz)
    spikes = np.arange(gap // 2, n - 50, gap) + rng.integers(-gap // 4, gap // 4, size=len(range(gap // 2, n - 50, gap)))
    for s in spikes:
        x[s - 3 : s - 3 + SPIKE.size, spike_channel] += spike_uv * SPIKE
    stim = None
    if len(stim_onsets_s):
        onsets = np.asarray(stim_onsets_s, dtype=float)
        p = int(pulse_ms * 1e-3 * FS)
        t = np.arange(int(10 * tail_ms * 1e-3 * FS))
        tail = tail_uv * np.exp(-t / (tail_ms * 1e-3 * FS))
        for on in onsets:
            i = int(round(on * FS))
            x[i : i + p // 2] -= artifact_uv
            x[i + p // 2 : i + p] += artifact_uv
            x[i + p : i + p + t.size] += tail[: max(0, min(t.size, n - i - p))][:, None]
        stim = StimEvents("STG 1", "Single Pulse", onsets, onsets + pulse_ms * 1e-3)
        # Spikes inside the artifact are unrecoverable; keep only those well clear of pulses.
        clear = np.all(np.abs(spikes[:, None] / FS - onsets[None, :] - 0.01) > 0.04, axis=1)
        spikes = spikes[clear]
    rec = NumpyRecording([x.astype(np.float32)], sampling_frequency=FS, channel_ids=[str(c) for c in range(n_channels)])
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)
    return rec, np.sort(spikes), stim


def match_fraction(found: np.ndarray, truth: np.ndarray, tol: int = 1) -> float:
    """Fraction of ``truth`` samples with a ``found`` sample within ``tol``."""
    if truth.size == 0:
        return 1.0
    if found.size == 0:
        return 0.0
    i = np.clip(np.searchsorted(found, truth), 1, found.size - 1)
    d = np.minimum(np.abs(found[i - 1] - truth), np.abs(found[i] - truth))
    return float(np.mean(d <= tol))
