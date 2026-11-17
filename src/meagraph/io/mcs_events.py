"""Stimulation events and MCS online spike streams (DECISIONS.md D2, D4).

Event datasets are ``EventEntity_<EventID>`` and segment datasets ``SegmentData[_ts]_<SegmentID>``,
the McsPy convention. Decoding by table row instead mislabels events (see docs/EXISTING_CODE.md).
All times are seconds on the MCS recording clock.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np

from meagraph.io._mcs_layout import (
    StreamInfo,
    decode,
    electrode_id,
    list_analog_streams,
    match_stream,
    recording_group,
    stream_groups,
)
from meagraph.spiketrains import SpikeTrains

_STG_EVENT = re.compile(r"(STG \d+) (Single Pulse|Marker) (Start|Stop)", re.IGNORECASE)


@dataclass(frozen=True, eq=False)
class EventEntity:
    """One MCS event entity, for example 'STG 1 Single Pulse Start'."""

    event_id: int
    label: str
    stream_label: str
    source_channel_labels: str
    timestamps_s: np.ndarray
    durations_s: np.ndarray


@dataclass(frozen=True, eq=False)
class StimEvents:
    """Stimulation times from one STG output.

    The electrode that was stimulated is not stored in MCS files. It is supplied from metadata,
    never read from the file: either one ``site`` for every pulse (``--stim-site``), or one site
    per pulse in ``pulse_sites`` (from a stimulation log; "" where a pulse's site is unknown).

    A stimulation log also gives each pulse's amplitude (``pulse_amplitudes``, in
    ``amplitude_unit`` as logged: µV in voltage mode, nA in current mode), the duration of its
    waveform's first phase (``pulse_phase_us``) and the waveform's name (``pulse_waveforms``).
    NaN or "" where unknown. Together they form the pulse's :attr:`conditions`.
    """

    source: str
    kind: str
    onsets_s: np.ndarray
    offsets_s: np.ndarray | None
    site: str | None = None
    pulse_sites: tuple[str, ...] | None = None
    pulse_amplitudes: tuple[float, ...] | None = None
    amplitude_unit: str = ""
    pulse_phase_us: tuple[float, ...] | None = None
    pulse_waveforms: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("pulse_sites", "pulse_amplitudes", "pulse_phase_us", "pulse_waveforms"):
            value = getattr(self, name)
            if value is not None and len(value) != self.onsets_s.size:
                raise ValueError(f"{self.source}: {len(value)} {name.replace('_', ' ')} for {self.onsets_s.size} pulses")
        if self.pulse_sites is not None and self.site is not None:
            raise ValueError(f"{self.source}: give one site for all pulses or one per pulse, not both")

    @property
    def n(self) -> int:
        return int(self.onsets_s.size)

    @property
    def durations_s(self) -> np.ndarray | None:
        return None if self.offsets_s is None else self.offsets_s - self.onsets_s

    @property
    def ends_s(self) -> np.ndarray:
        """Pulse offsets, or the onsets when the file has no Stop events."""
        return self.onsets_s if self.offsets_s is None else self.offsets_s

    @property
    def sites(self) -> np.ndarray | None:
        """The stimulated electrode of every pulse ("" if unknown), or None if no site is known."""
        if self.pulse_sites is not None:
            return np.array(self.pulse_sites, dtype=object)
        if self.site is not None:
            return np.full(self.n, self.site, dtype=object)
        return None

    @property
    def conditions(self) -> np.ndarray:
        """Each pulse's stimulus condition, e.g. "1000mV/500us/bi500" (see :func:`condition_label`);
        "" where the log gave no parameters (or there is no log)."""
        n = self.n
        amps = self.pulse_amplitudes or (float("nan"),) * n
        phases = self.pulse_phase_us or (float("nan"),) * n
        waves = self.pulse_waveforms or ("",) * n
        return np.array([condition_label(a, self.amplitude_unit, p, w) for a, p, w in zip(amps, phases, waves)], dtype=object)

    def with_site(self, site: str | None) -> StimEvents:
        return replace(self, site=None if site is None else str(site), pulse_sites=None)

    def with_pulse_sites(self, sites) -> StimEvents:
        return replace(self, site=None, pulse_sites=tuple("" if s is None else str(s) for s in sites))

    def with_pulse_params(self, amplitudes=None, unit: str = "", phase_us=None, waveforms=None) -> StimEvents:
        """Per-pulse stimulus parameters from a stimulation log (None: not logged)."""

        def floats(v):
            return None if v is None else tuple(float("nan") if x is None else float(x) for x in v)

        return replace(self, pulse_amplitudes=floats(amplitudes), amplitude_unit=unit or "", pulse_phase_us=floats(phase_us),
                       pulse_waveforms=None if waveforms is None else tuple("" if w is None else str(w) for w in waveforms))  # fmt: skip

    def select(self, index) -> StimEvents:
        """The pulses at ``index`` (indices or a boolean mask)."""
        idx = np.arange(self.n)[index]

        def pick(v):
            return None if v is None else tuple(v[i] for i in idx)

        return replace(
            self, onsets_s=self.onsets_s[idx], offsets_s=None if self.offsets_s is None else self.offsets_s[idx],
            pulse_sites=pick(self.pulse_sites), pulse_amplitudes=pick(self.pulse_amplitudes),
            pulse_phase_us=pick(self.pulse_phase_us), pulse_waveforms=pick(self.pulse_waveforms),
        )  # fmt: skip


def condition_label(amplitude: float, unit: str, phase_us: float, waveform: str = "") -> str:
    """A pulse's stimulus condition as text: amplitude, first-phase duration and waveform name,
    e.g. "1000mV/500us/bi500". Amplitudes logged in µV are shown in mV and nA in µA. Unknown
    parts are left out; "" if nothing is known."""
    parts = []
    if not math.isnan(amplitude):
        scale = {"uv": (1e-3, "mV"), "µv": (1e-3, "mV"), "na": (1e-3, "uA")}.get(unit.lower())
        value, shown = (amplitude * scale[0], scale[1]) if scale else (amplitude, unit)
        parts.append(f"{value:g}{shown}")
    if not math.isnan(phase_us):
        parts.append(f"{phase_us:g}us")
    if waveform:
        parts.append(waveform)
    return "/".join(parts)


def mixed_conditions(stim) -> bool:
    """Whether the pulses of ``stim`` (StimEvents, or a sequence of them) were given more than one
    stimulus condition. Only then do analyses split by condition; otherwise they run as before."""
    events = [stim] if isinstance(stim, StimEvents) else list(stim)
    known = {c for s in events for c in s.conditions.tolist() if c}
    return len(known) > 1


def read_event_entities(path: str | Path, recording_index: int = 0) -> list[EventEntity]:
    """Every event entity that has data, from every EventStream."""
    out = []
    with h5py.File(path, "r") as f:
        rec = recording_group(f, recording_index)
        for grp in stream_groups(rec, "EventStream"):
            stream_label = StreamInfo.from_group(grp).label
            for row in grp["InfoEvent"][:]:
                event_id = int(row["EventID"])
                name = f"EventEntity_{event_id}"
                if name not in grp:
                    continue
                data = grp[name][:]
                out.append(
                    EventEntity(
                        event_id=event_id,
                        label=" ".join(decode(row["Label"]).split()),
                        stream_label=stream_label,
                        source_channel_labels=decode(row["SourceChannelLabels"]),
                        timestamps_s=data[0].astype(np.float64) * 1e-6,
                        durations_s=(data[1].astype(np.float64) * 1e-6) if data.shape[0] > 1 else np.zeros(data.shape[1]),
                    )
                )
    return out


def _pair_offsets(onsets: np.ndarray, offsets: np.ndarray) -> np.ndarray | None:
    """Match each onset to the first offset at or after it and before the next onset."""
    idx = np.searchsorted(offsets, onsets, side="left")
    if np.any(idx >= offsets.size):
        return None
    paired = offsets[idx]
    next_onset = np.append(onsets[1:], np.inf)
    if np.any(paired >= next_onset) or np.unique(idx).size != idx.size:
        return None
    return paired


def read_stim_events(path: str | Path, recording_index: int = 0) -> list[StimEvents]:
    """STG stimulation events, one entry per (STG output, kind) that has Start events.

    Returns an empty list for recordings without stimulation.
    """
    starts: dict[tuple[str, str], np.ndarray] = {}
    stops: dict[tuple[str, str], np.ndarray] = {}
    for ent in read_event_entities(path, recording_index):
        m = _STG_EVENT.search(ent.label)
        if not m:
            continue
        key = (m.group(1).upper(), m.group(2).title())
        target = starts if m.group(3).lower() == "start" else stops
        if key in target:
            raise ValueError(f"duplicate event entity for {key} {m.group(3)} in {path}")
        target[key] = np.sort(ent.timestamps_s)
    out = []
    for key in sorted(starts):
        onsets = starts[key]
        offsets = None
        if key in stops:
            offsets = _pair_offsets(onsets, stops[key])
            if offsets is None:
                warnings.warn(f"{key}: Stop events could not be paired with Start events; offsets dropped", stacklevel=2)
        out.append(StimEvents(source=key[0], kind=key[1], onsets_s=onsets, offsets_s=offsets))
    return out


def read_mcs_spikes(
    path: str | Path,
    stream: str = "Spike Detector",
    recording_index: int = 0,
    merge_units: bool = True,
) -> SpikeTrains:
    """Timestamps from an MCS SegmentStream ('Spike Detector' or 'Spike Sorter').

    These are MCS's online detections, read for comparison only (D2). With ``merge_units``,
    sorter units on one electrode are merged and unit ids are electrode ids. Otherwise they
    are ``'<electrode>#<sorter unit>'``.

    MCS can log detections from before the saved analog data starts (seen in the beforestim
    file). Events outside the raw stream's time span are dropped with a warning.
    """
    with h5py.File(path, "r") as f:
        rec = recording_group(f, recording_index)
        groups = stream_groups(rec, "SegmentStream")
        infos = [StreamInfo.from_group(g) for g in groups]
        chosen = match_stream(infos, stream, kind="segment stream")
        grp = groups[infos.index(chosen)]
        analog = list_analog_streams(rec)
        raw = [a for a in analog if a.is_raw]
        span = raw[0] if raw else analog[0]
        t_start, t_stop = span.t_start_s, span.t_start_s + span.duration_s

        times: dict[str, list[np.ndarray]] = {}
        channels: dict[str, str] = {}
        for row in grp["InfoSegment"][:]:
            seg_id = int(row["SegmentID"])
            name = f"SegmentData_ts_{seg_id}"
            if name not in grp:
                continue
            electrode = electrode_id(row["Label"])
            unit = electrode if merge_units else f"{electrode}#{int(row['Sorter Unit'])}"
            times.setdefault(unit, []).append(grp[name][:].ravel().astype(np.float64) * 1e-6)
            channels[unit] = electrode

    merged = {u: np.sort(np.concatenate(ts)) for u, ts in times.items()}
    outside = sum(int(np.sum((t < t_start) | (t > t_stop))) for t in merged.values())
    if outside:
        warnings.warn(
            f"{outside} MCS segment events lie outside the analog data [{t_start:g}, {t_stop:g}] s and were dropped",
            stacklevel=2,
        )
        merged = {u: t[(t >= t_start) & (t <= t_stop)] for u, t in merged.items()}
    return SpikeTrains.from_dict(merged, t_start, t_stop, channel_ids=channels)
