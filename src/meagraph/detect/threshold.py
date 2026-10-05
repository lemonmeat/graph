"""Threshold spike detection on SpikeInterface recordings.

The detection settings are those of the retired ``spikes.py`` (Q8). Stimulation is blanked per
channel from the measured artifact recovery, and the stimulating electrode is dropped (Q9;
DECISIONS.md D10).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from scipy.stats import binomtest
from spikeinterface.core import BaseRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from meagraph.detect.noise import median_abs_noise_uv
from meagraph.io.mcs_events import StimEvents
from meagraph.preprocess import detection_band, merge_windows
from meagraph.probe.build import channel_positions_um
from meagraph.spiketrains import SpikeTrains
from meagraph.stimulation.artifacts import Recovery, measure_recovery, pulse_windows


class DetectionConfig(BaseModel):
    """Every parameter of detection. Saved with each result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Detection (Q8: the spikes.py settings)
    threshold_sigma: PositiveFloat = 5.0
    band_hz: tuple[PositiveFloat, PositiveFloat] = (300.0, 3000.0)
    filter_order: PositiveInt = 3
    refractory_ms: PositiveFloat = 1.0
    max_amplitude_uv: PositiveFloat = 1000.0
    cutout_ms: tuple[PositiveFloat, PositiveFloat] = (1.0, 2.0)
    reject_rebound: bool = True
    # Stimulation (Q9): blank each channel from pulse onset - blank_pre_ms until its artifact has
    # recovered, at least recovery_min_post_ms and at most recovery_max_post_ms after pulse offset.
    # The stimulated electrode (StimEvents.site) is excluded.
    blank_pre_ms: PositiveFloat = 1.0
    guard_ms: float = 1.0  # peaks this soon after a window are dropped
    recovery_min_post_ms: PositiveFloat = 1.0
    recovery_max_post_ms: PositiveFloat = 50.0
    recovery_threshold_sigma: PositiveFloat = 1.0
    # Signal-quality control: a channel is "active" when negative peaks outnumber positive ones.
    # Counted outside stimulation and the qc_exclude_post_ms after each pulse, where evoked
    # responses and artifact residue are not spontaneous-like.
    qc_alpha: PositiveFloat = 1e-3
    qc_min_spikes: PositiveInt = 20
    qc_exclude_post_ms: float = 50.0
    # Execution
    n_jobs: PositiveInt = 1
    chunk_duration_s: PositiveFloat = 1.0


@dataclass(frozen=True, eq=False)
class ChannelQC:
    """Polarity control per channel: noise crossings are symmetric, extracellular spikes are negative."""

    channel_ids: tuple[str, ...]
    n_negative: np.ndarray
    n_positive: np.ndarray
    p_value: np.ndarray
    active: np.ndarray


@dataclass(frozen=True, eq=False)
class DetectionResult:
    """Everything later steps need from a recording: spikes (with electrode positions when the
    recording has a probe), per-channel QC, and the stimulation events, so analyses of a saved
    result never have to reopen the raw file."""

    trains: SpikeTrains
    amplitudes_uv: dict[str, np.ndarray]
    waveforms_uv: dict[str, np.ndarray]  # (n_spikes, n_samples), filtered
    noise_uv: dict[str, float]
    qc: ChannelQC
    excluded: dict[str, str]  # channel id -> reason; no spikes are reported for these
    recovery_ms: dict[str, dict[str, float]]  # STG source -> channel -> ms (adaptive blanking only)
    config: DetectionConfig
    stim: tuple[StimEvents, ...] = ()  # the events that were blanked, with their sites

    @property
    def active_channels(self) -> tuple[str, ...]:
        return tuple(c for c, a in zip(self.qc.channel_ids, self.qc.active) if a and c not in self.excluded)


def _blanking(recording: BaseRecording, stim: Sequence[StimEvents], cfg: DetectionConfig):
    """Per-channel windows, plus recovery measurements for the record."""
    ids = [str(c) for c in recording.channel_ids]
    n = recording.get_num_samples()
    if not stim:
        return None, {}
    per_channel: dict[str, list[np.ndarray]] = {c: [] for c in ids}
    recoveries: dict[str, dict[str, float]] = {}
    for s in stim:
        rec: Recovery = measure_recovery(
            recording, s, pre_ms=cfg.blank_pre_ms, min_post_ms=cfg.recovery_min_post_ms,
            max_post_ms=cfg.recovery_max_post_ms, threshold_sigma=cfg.recovery_threshold_sigma,
            band_hz=cfg.band_hz, filter_order=cfg.filter_order,
        )  # fmt: skip
        recoveries[s.source] = dict(zip(rec.channel_ids, rec.recovery_ms.tolist()))
        windows = pulse_windows(recording, s, cfg.blank_pre_ms, rec.post_ms())
        for k, c in enumerate(ids):
            per_channel[c].append(windows[k])
    return {c: merge_windows(np.concatenate(w), n) for c, w in per_channel.items()}, recoveries


def _polarity_control(recording, stim, sample, chan, amp, ids, event_span, cfg: DetectionConfig) -> ChannelQC:
    """Are a channel's threshold crossings dominated by negative events? (DECISIONS.md D10)

    Crossings of either sign closer than ``event_span`` samples on one channel form one event,
    whose polarity is that of its largest peak. A spike's trough outweighs its overshoot, so a
    spike counts as negative; symmetric noise, including ringing bursts, is equally likely to
    count either way, so under the null the two counts are binomial with p = 1/2.
    Events within ``qc_exclude_post_ms`` after a pulse are not counted.
    """
    n_ch = len(ids)
    order = np.lexsort((sample, chan))
    s, c, a = sample[order], chan[order], amp[order]
    if s.size == 0:
        zeros = np.zeros(n_ch, dtype=np.int64)
        return ChannelQC(tuple(ids), zeros, zeros, np.ones(n_ch), np.zeros(n_ch, dtype=bool))
    event = np.cumsum(np.r_[True, (np.diff(s) >= event_span) | (np.diff(c) != 0)]) - 1
    by_size = np.lexsort((-np.abs(a), event))
    dominant = by_size[np.r_[True, np.diff(event[by_size]) != 0]]  # largest peak of each event
    s, c, negative = s[dominant], c[dominant], a[dominant] < 0
    if stim:
        n = recording.get_num_samples()
        windows = merge_windows(np.concatenate([pulse_windows(recording, st, cfg.blank_pre_ms, cfg.qc_exclude_post_ms) for st in stim]), n)
        k = np.searchsorted(windows[:, 0], s, side="right") - 1
        counted = ~((k >= 0) & (s < windows[np.maximum(k, 0), 1]))
        s, c, negative = s[counted], c[counted], negative[counted]
    n_neg = np.bincount(c[negative], minlength=n_ch)
    n_pos = np.bincount(c[~negative], minlength=n_ch)
    p = np.array([binomtest(int(x), int(x + y), 0.5, alternative="greater").pvalue if x + y else 1.0 for x, y in zip(n_neg, n_pos)])
    return ChannelQC(tuple(ids), n_neg, n_pos, p, (p < cfg.qc_alpha) & (n_neg >= cfg.qc_min_spikes))


def _cutouts(filt: BaseRecording, samples: np.ndarray, channels: np.ndarray, pre: int, post: int) -> np.ndarray:
    """(n_peaks, pre + post) snippets, reading nearby peaks in shared blocks."""
    out = np.zeros((samples.size, pre + post), dtype=np.float32)
    if samples.size == 0:
        return out
    order = np.argsort(samples)
    gap = int(0.1 * filt.get_sampling_frequency())
    block_max = int(10 * filt.get_sampling_frequency())
    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and samples[order[j + 1]] - samples[order[j]] < gap and samples[order[j + 1]] - samples[order[i]] < block_max:
            j += 1
        a, b = samples[order[i]] - pre, samples[order[j]] + post
        block = filt.get_traces(start_frame=a, end_frame=b)
        for k in order[i : j + 1]:
            s = samples[k] - a
            out[k] = block[s - pre : s + post, channels[k]]
        i = j + 1
    return out


def detect_spikes(
    recording: BaseRecording,
    stim: Sequence[StimEvents] = (),
    config: DetectionConfig | None = None,
) -> DetectionResult:
    """Detect negative threshold crossings on every channel.

    ``recording`` is a session recording (raw units, probe attached). ``stim`` gives the
    stimulation events to blank; a ``site`` on an event marks that electrode for exclusion.
    """
    cfg = config or DetectionConfig()
    fs = recording.get_sampling_frequency()
    n = recording.get_num_samples()
    ids = [str(c) for c in recording.channel_ids]
    windows, recoveries = _blanking(recording, stim, cfg)
    filt = detection_band(recording, windows, cfg.band_hz, cfg.filter_order)
    noise = median_abs_noise_uv(filt)

    # One pass finds both polarities: negative peaks are spikes, positive peaks feed the QC.
    peaks = detect_peaks(
        filt, method="by_channel",
        method_kwargs=dict(peak_sign="both", detect_threshold=cfg.threshold_sigma,
                           exclude_sweep_ms=cfg.refractory_ms, noise_levels=noise),
        job_kwargs=dict(n_jobs=cfg.n_jobs, chunk_duration=f"{cfg.chunk_duration_s}s", progress_bar=False),
    )  # fmt: skip
    sample, chan, amp = peaks["sample_index"].astype(np.int64), peaks["channel_index"].astype(np.int64), peaks["amplitude"]

    pre, post = (int(round(ms * 1e-3 * fs)) for ms in cfg.cutout_ms)
    keep = (sample >= pre) & (sample < n - post)
    if windows is not None:
        guard = int(round(cfg.guard_ms * 1e-3 * fs))
        for c, w in windows.items():
            if w.size == 0:
                continue
            on_c = chan == ids.index(c)
            k = np.searchsorted(w[:, 0], sample[on_c], side="right") - 1
            inside = (k >= 0) & (sample[on_c] < w[np.maximum(k, 0), 1] + guard)
            keep[np.flatnonzero(on_c)[inside]] = False
    sample, chan, amp = sample[keep], chan[keep], amp[keep]

    qc = _polarity_control(recording, stim, sample, chan, amp, ids, pre + post, cfg)
    neg = amp < 0
    sample, chan, amp = sample[neg], chan[neg], amp[neg]
    sample, chan, amp = sample[-amp <= cfg.max_amplitude_uv], chan[-amp <= cfg.max_amplitude_uv], amp[-amp <= cfg.max_amplitude_uv]
    wf = _cutouts(filt, sample, chan, pre, post)
    if cfg.reject_rebound and sample.size:
        ok = wf.max(axis=1) < -amp
        sample, chan, amp, wf = sample[ok], chan[ok], amp[ok], wf[ok]

    excluded = {s.site: f"stimulation site ({s.source})" for s in stim if s.site is not None}
    t0 = recording.get_start_time()
    times, amps, wfs = {}, {}, {}
    for k, c in enumerate(ids):
        sel = np.flatnonzero(chan == k) if c not in excluded else np.zeros(0, dtype=np.int64)
        sel = sel[np.argsort(sample[sel], kind="stable")]
        times[c] = t0 + sample[sel] / fs
        amps[c] = amp[sel].astype(np.float32)
        wfs[c] = wf[sel]
    trains = SpikeTrains.from_dict(times, t0, t0 + n / fs)
    if recording.has_probe():
        trains = trains.with_positions(channel_positions_um(recording))
    return DetectionResult(
        trains=trains,
        amplitudes_uv=amps,
        waveforms_uv=wfs,
        noise_uv=dict(zip(ids, noise.astype(float).tolist())),
        qc=qc,
        excluded=excluded,
        recovery_ms=recoveries,
        config=cfg,
        stim=tuple(stim),
    )
