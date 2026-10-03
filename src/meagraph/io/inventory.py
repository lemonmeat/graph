"""Structured summary of an MCS HDF5 file: what is in it, without loading the data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from meagraph.io._mcs_layout import (
    AnalogStreamInfo,
    StreamInfo,
    decode,
    list_analog_streams,
    recording_group,
    stream_groups,
)
from meagraph.io.mcs_events import EventEntity, read_event_entities


@dataclass(frozen=True)
class SegmentStreamInfo(StreamInfo):
    n_entities: int = 0
    n_events: int = 0


@dataclass(frozen=True)
class FileInventory:
    path: Path
    size_bytes: int
    generator: str
    protocol: str
    program: str
    mea_layout: str
    date: str
    file_guid: str
    duration_s: float
    analog_streams: tuple[AnalogStreamInfo, ...]
    event_entities: tuple[EventEntity, ...]
    segment_streams: tuple[SegmentStreamInfo, ...]

    @property
    def raw_stream(self) -> AnalogStreamInfo | None:
        raws = [s for s in self.analog_streams if s.is_raw]
        return raws[0] if len(raws) == 1 else None


def inspect_file(path: str | Path, recording_index: int = 0) -> FileInventory:
    path = Path(path)
    with h5py.File(path, "r") as f:
        root, data = f.attrs, f["Data"].attrs
        rec = recording_group(f, recording_index)
        analog = tuple(list_analog_streams(rec))
        segments = []
        for grp in stream_groups(rec, "SegmentStream"):
            ts_names = [k for k in grp if k.startswith("SegmentData_ts_")]
            segments.append(
                SegmentStreamInfo(
                    **StreamInfo.from_group(grp).__dict__,
                    n_entities=int(grp["InfoSegment"].shape[0]),
                    n_events=int(sum(grp[k].shape[-1] for k in ts_names)),
                )
            )
        inv = dict(
            path=path,
            size_bytes=path.stat().st_size,
            generator=f"{decode(root.get('GeneratingApplicationName', '?'))} {decode(root.get('GeneratingApplicationVersion', ''))}".strip(),
            protocol=f"{decode(root.get('McsHdf5ProtocolType', '?'))} v{decode(root.get('McsHdf5ProtocolVersion', '?'))}",
            program=f"{decode(data.get('ProgramName', '?'))} {decode(data.get('ProgramVersion', ''))}".strip(),
            mea_layout=decode(data.get("MeaLayout", "")),
            date=decode(data.get("Date", "")),
            file_guid=decode(data.get("FileGUID", "")),
            duration_s=float(rec.attrs.get("Duration", 0)) * 1e-6,
        )
    return FileInventory(
        **inv,
        analog_streams=analog,
        event_entities=tuple(read_event_entities(path, recording_index)),
        segment_streams=tuple(segments),
    )


def format_inventory(inv: FileInventory) -> str:
    lines = [
        f"{inv.path.name}  ({inv.size_bytes / 1e6:.0f} MB)",
        f"  written by {inv.generator}, {inv.program}; protocol {inv.protocol}",
        f"  MEA layout {inv.mea_layout!r}, {inv.date}, recording clock ends at {inv.duration_s:.3f} s",
        "  analog streams (processing order):",
    ]
    for s in inv.analog_streams:
        role = "raw" if s.is_raw else "derived"
        lines.append(
            f"    {s.short_label:<22} [{s.name}, {role}] {s.n_channels} ch @ {s.sampling_frequency_hz:g} Hz, "
            f"{s.duration_s:.3f} s from t = {s.t_start_s:.3f} s"
        )
    if inv.event_entities:
        lines.append("  event entities:")
        for e in inv.event_entities:
            ts = e.timestamps_s
            isi = f", median interval {np.median(np.diff(ts)):.4f} s" if ts.size > 1 else ""
            lines.append(f"    #{e.event_id} {e.label!r}: {ts.size} events{isi}")
    else:
        lines.append("  event entities: none")
    for s in inv.segment_streams:
        lines.append(f"  segment stream {s.short_label!r}: {s.n_entities} entities, {s.n_events} events")
    return "\n".join(lines)
