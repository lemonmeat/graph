"""Low-level helpers for the MCS HDF5 layout. The layout is documented in docs/DATA_FORMAT.md.

Rules enforced here:
- analog streams are chosen by label or role, never by HDF5 index (stream index != processing order)
- event and segment entities are decoded by ID, never by table row (DECISIONS.md D4)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import h5py
import numpy as np

NULL_GUID = "00000000-0000-0000-0000-000000000000"
RAW = "raw"


def decode(value) -> str:
    """HDF5 bytes, numpy bytes or str -> str."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    return str(value)


def electrode_id(label) -> str:
    """'E-00303 47' -> '47'. The electrode id is the last whitespace token of an MCS label."""
    tokens = decode(label).replace(";", " ").split()
    if not tokens:
        raise ValueError(f"empty MCS channel label {label!r}")
    return tokens[-1]


def short_label(label: str) -> str:
    """'Filter (3);Filter; Filter Data3' -> 'Filter (3)'."""
    return label.split(";")[0].strip() or label


def recording_group(f: h5py.File, recording_index: int = 0) -> h5py.Group:
    name = f"Data/Recording_{recording_index}"
    if name not in f:
        available = sorted(f["Data"]) if "Data" in f else []
        raise KeyError(f"{f.filename}: no {name}; available: {available}")
    return f[name]


def stream_groups(rec: h5py.Group, kind: str) -> list[h5py.Group]:
    """Stream groups of one kind ('AnalogStream', 'EventStream', 'SegmentStream'), by numeric suffix."""
    if kind not in rec:
        return []
    groups = rec[kind]
    return [groups[k] for k in sorted(groups, key=lambda s: int(s.rsplit("_", 1)[1]))]


@dataclass(frozen=True)
class StreamInfo:
    """Identity of an MCS stream: its HDF5 name, labels and lineage GUIDs."""

    name: str
    label: str
    guid: str
    source_guid: str
    data_subtype: str

    @property
    def short_label(self) -> str:
        return short_label(self.label)

    @classmethod
    def from_group(cls, grp: h5py.Group) -> StreamInfo:
        a = grp.attrs
        return cls(
            name=grp.name.rsplit("/", 1)[1],
            label=decode(a.get("Label", grp.name.rsplit("/", 1)[1])),
            guid=decode(a.get("StreamGUID", "")),
            source_guid=decode(a.get("SourceStreamGUID", NULL_GUID)),
            data_subtype=decode(a.get("DataSubType", "")),
        )


@dataclass(frozen=True)
class AnalogStreamInfo(StreamInfo):
    n_channels: int = 0
    n_samples: int = 0
    sampling_frequency_hz: float = 0.0
    t_start_s: float = 0.0

    @property
    def is_raw(self) -> bool:
        return self.source_guid == NULL_GUID

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sampling_frequency_hz

    @classmethod
    def from_group(cls, grp: h5py.Group) -> AnalogStreamInfo:
        base = StreamInfo.from_group(grp)
        info = grp["InfoChannel"][:]
        n_channels, n_samples = grp["ChannelData"].shape
        tick_us = channel_tick_us(info)
        return cls(
            **base.__dict__,
            n_channels=int(n_channels),
            n_samples=int(n_samples),
            sampling_frequency_hz=1e6 / tick_us,
            t_start_s=first_timestamp_s(grp["ChannelDataTimeStamps"][:], tick_us, int(n_samples)),
        )


def channel_tick_us(info: np.ndarray) -> float:
    ticks = np.unique(info["Tick"])
    if ticks.size != 1:
        raise ValueError(f"channels in one stream have different sampling intervals: {ticks.tolist()} µs")
    return float(ticks[0])


def first_timestamp_s(timestamps: np.ndarray, tick_us: float, n_samples: int) -> float:
    """First-sample time from ChannelDataTimeStamps rows [FirstTimeStamp (µs), FirstIndex, LastIndex].

    Several rows mean the stream was written in blocks. They are accepted only when the blocks
    are contiguous in time, since a gap would break the sample-index-to-time mapping.
    """
    ts = np.atleast_2d(np.asarray(timestamps, dtype=np.int64))
    if ts.shape[1] != 3:
        raise ValueError(f"unexpected ChannelDataTimeStamps shape {ts.shape}")
    ts = ts[np.argsort(ts[:, 1])]
    for prev, cur in zip(ts[:-1], ts[1:]):
        expected = prev[0] + (cur[1] - prev[1]) * tick_us
        if cur[1] != prev[2] + 1 or abs(cur[0] - expected) > tick_us / 2:
            raise NotImplementedError("analog stream has gaps between blocks; segmented recordings are not supported")
    if ts[0, 1] != 0 or ts[-1, 2] != n_samples - 1:
        raise ValueError(f"ChannelDataTimeStamps indices {ts[[0, -1], [1, 2]].tolist()} do not cover {n_samples} samples")
    return float(ts[0, 0]) * 1e-6


def lineage_order(streams: Sequence[StreamInfo]) -> list[StreamInfo]:
    """Order streams so every stream comes after its source (raw first, then filter chain)."""
    by_guid = {s.guid: s for s in streams}
    children: dict[str, list[StreamInfo]] = {}
    roots = []
    for s in streams:
        if s.source_guid in by_guid and s.source_guid != s.guid:
            children.setdefault(s.source_guid, []).append(s)
        else:
            roots.append(s)
    ordered: list[StreamInfo] = []
    queue = sorted(roots, key=lambda s: s.name)
    while queue:
        s = queue.pop(0)
        ordered.append(s)
        queue[:0] = sorted(children.get(s.guid, []), key=lambda c: c.name)
    return ordered


def list_analog_streams(rec: h5py.Group) -> list[AnalogStreamInfo]:
    """Analog streams in processing order (raw first, then each filter stage)."""
    return lineage_order([AnalogStreamInfo.from_group(g) for g in stream_groups(rec, "AnalogStream")])


def _norm(label: str) -> str:
    return " ".join(label.lower().split())


def match_stream(streams: Sequence[StreamInfo], query: str, kind: str = "stream") -> StreamInfo:
    """Pick a stream by full label, short label ('Filter (3)'), or short label without the
    instance number ('Spike Detector'). The match must be unique."""
    q = _norm(query)
    for key in (
        lambda s: _norm(s.label),
        lambda s: _norm(s.short_label),
        lambda s: _norm(re.sub(r"\s*\(\d+\)$", "", s.short_label)),
    ):
        hits = [s for s in streams if key(s) == q]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"{kind} {query!r} is ambiguous: {[s.short_label for s in hits]}")
    raise KeyError(f"no {kind} matching {query!r}; available: {[s.short_label for s in streams]}")


def resolve_analog_stream(rec: h5py.Group, stream: str = RAW) -> AnalogStreamInfo:
    """``'raw'`` selects the hardware stream (no source stream); anything else is matched by label."""
    streams = list_analog_streams(rec)
    if not isinstance(stream, str):
        raise TypeError("select analog streams by label or 'raw', not by index (stream order != processing order)")
    if _norm(stream) == RAW:
        raws = [s for s in streams if s.is_raw]
        if len(raws) != 1:
            raise ValueError(f"expected exactly one raw analog stream, found {[s.short_label for s in raws]}")
        return raws[0]
    return match_stream(streams, stream, kind="analog stream")
