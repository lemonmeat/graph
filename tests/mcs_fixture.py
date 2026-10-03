"""Write small files in the MCS HDF5 layout observed in docs/DATA_FORMAT.md.

The defaults deliberately exercise the traps found in Phase 0: RowIndex is permuted, ADZero is
non-zero, the filter stream sits at a lower HDF5 index than the raw stream, InfoEvent rows are
not in EventID order, and sorter SegmentIDs differ from their table rows.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

VSTR = h5py.string_dtype("utf-8")
NULL_GUID = "00000000-0000-0000-0000-000000000000"
RAW_GUID = "5c5b0f4b-0000-0000-0000-000000000000"
FILTER_GUID = "3369051c-0000-0000-0000-000000000000"
DETECTOR_GUID = "56dddeec-0000-0000-0000-000000000000"
SORTER_GUID = "354415e9-0000-0000-0000-000000000000"
STG_GUID = "6402936b-0000-0000-0000-000000000000"

INFO_CHANNEL = np.dtype(
    [
        ("ChannelID", "<i4"), ("RowIndex", "<i4"), ("GroupID", "<i4"), ("ElectrodeGroup", "<i4"),
        ("Label", VSTR), ("RawDataType", VSTR), ("Unit", VSTR), ("Exponent", "<i4"), ("ADZero", "<i4"),
        ("Tick", "<i8"), ("ConversionFactor", "<i8"), ("ADCBits", "<i4"),
        ("HighPassFilterType", VSTR), ("HighPassFilterCutOffFrequency", VSTR), ("HighPassFilterOrder", "<i4"),
        ("LowPassFilterType", VSTR), ("LowPassFilterCutOffFrequency", VSTR), ("LowPassFilterOrder", "<i4"),
    ]
)  # fmt: skip
INFO_EVENT = np.dtype(
    [("EventID", "<i4"), ("GroupID", "<i4"), ("Label", VSTR), ("RawDataType", VSTR), ("RawDataBytes", "<i4"),
     ("SourceChannelIDs", VSTR), ("SourceChannelLabels", VSTR)]
)  # fmt: skip
INFO_SEGMENT = np.dtype(
    [("SegmentID", "<i4"), ("Sorter Unit", "<i4"), ("GroupID", "<i4"), ("Label", VSTR), ("PreInterval", "<i8"),
     ("PostInterval", "<i8"), ("SegmentType", VSTR), ("SourceChannelIDs", VSTR)]
)  # fmt: skip
STG_EVENT_LABELS = [
    "STG 1 Single Pulse Start", "STG 1 Single Pulse Stop", "STG 1 Marker Start", "STG 1 Marker Stop",
    "STG 2 Single Pulse Start", "STG 2 Single Pulse Stop", "STG 2 Marker Start", "STG 2 Marker Stop",
]  # fmt: skip


def _stream_attrs(grp, label, guid, source, subtype, stream_type):
    grp.attrs["Label"] = np.bytes_(label)
    grp.attrs["StreamGUID"] = np.bytes_(guid)
    grp.attrs["SourceStreamGUID"] = np.bytes_(source)
    grp.attrs["DataSubType"] = np.bytes_(subtype)
    grp.attrs["StreamType"] = np.bytes_(stream_type)


def _analog(grp, labels, data_by_channel, row_index, adzero, conversion, exponent, tick_us, first_us, unit):
    n_ch, n = data_by_channel.shape
    table = np.zeros(n_ch, INFO_CHANNEL)
    for k, lbl in enumerate(labels):
        table[k] = (k, row_index[k], 0, 0, f"E-00303 {lbl}".encode(), b"Int", unit.encode(), exponent, adzero[k],
                    tick_us, conversion[k], 24, b"", b"-1", -1, b"", b"-1", -1)  # fmt: skip
    rows = np.empty_like(data_by_channel)
    rows[row_index] = data_by_channel
    grp.create_dataset("ChannelData", data=rows, chunks=(n_ch, min(n, 1092)), compression="gzip")
    grp.create_dataset("ChannelDataTimeStamps", data=np.array([[first_us, 0, n - 1]], np.int64))
    grp.create_dataset("InfoChannel", data=table)


def write_mcs_h5(
    path: str | Path,
    *,
    labels=("47", "12", "33", "15"),
    n_samples: int = 2000,
    tick_us: int = 100,
    first_timestamp_us: int = 500_000,
    row_index=None,
    adzero=None,
    conversion=None,
    exponent: int = -12,
    unit: str = "V",
    stim_onsets_us=None,
    stim_offset_delay_us: int = 3000,
    info_event_order=None,
    detector_spikes_us=None,
) -> dict:
    """Write a fixture file and return the ground truth needed to check readers against it."""
    labels = list(labels)
    n_ch = len(labels)
    row_index = np.array(row_index if row_index is not None else list(range(n_ch))[::-1])
    adzero = np.array(adzero if adzero is not None else [100 * (k + 1) for k in range(n_ch)])
    conversion = np.array(conversion if conversion is not None else [8670 + 10 * k for k in range(n_ch)])
    t = np.arange(n_samples)
    raw = np.stack([(k + 1) * 1000 + (t % 97) - 50 * k for k in range(n_ch)]).astype(np.int32)
    filtered = (-raw).astype(np.int32)
    gains_uv = conversion * 10.0**exponent * 1e6
    truth_uv = ((raw - adzero[:, None]) * gains_uv[:, None]).T  # (samples, channels), InfoChannel order

    with h5py.File(path, "w") as f:
        f.attrs["GeneratingApplicationName"] = np.bytes_("Multi Channel DataManager")
        f.attrs["McsHdf5ProtocolType"] = np.bytes_("RawData")
        f.attrs["McsHdf5ProtocolVersion"] = np.int32(3)
        data = f.create_group("Data")
        data.attrs["MeaLayout"] = np.bytes_("ME21Combi60")
        data.attrs["FileGUID"] = np.bytes_("fixture-guid")
        data.attrs["ProgramName"] = np.bytes_("Multi Channel Experimenter")
        rec = data.create_group("Recording_0")
        rec.attrs["Duration"] = np.int64(first_timestamp_us + n_samples * tick_us)
        rec.attrs["TimeStamp"] = np.int64(0)

        analog = rec.create_group("AnalogStream")
        # Filter stream at the lower index on purpose: index order is not processing order.
        g = analog.create_group("Stream_0")
        _stream_attrs(g, "Filter (1);Filter; Filter Data1", FILTER_GUID, RAW_GUID, "Electrode", "Analog")
        _analog(g, labels, filtered, row_index, adzero, conversion, exponent, tick_us, first_timestamp_us, unit)
        g = analog.create_group("Stream_1")
        _stream_attrs(g, "Data Acquisition (1);MEA2100-Mini; Electrode Raw Data1", RAW_GUID, NULL_GUID, "Electrode", "Analog")
        _analog(g, labels, raw, row_index, adzero, conversion, exponent, tick_us, first_timestamp_us, unit)

        events = rec.create_group("EventStream").create_group("Stream_0")
        _stream_attrs(events, "Stimulator (1);Stimulator; STG Events1", STG_GUID, NULL_GUID, "StgSideband", "Event")
        order = list(info_event_order) if info_event_order is not None else list(range(8))
        info = np.zeros(8, INFO_EVENT)
        for row, k in enumerate(order):
            stg = 70 if k < 4 else 71
            info[row] = (k + 1, 0, f"E-00303  {STG_EVENT_LABELS[k]}".encode(), b"Int", 4, str(stg).encode(),
                         f"E-00303 Sideband Data {stg - 70}".encode())  # fmt: skip
        events.create_dataset("InfoEvent", data=info)
        if stim_onsets_us is not None:
            onsets = np.asarray(stim_onsets_us, np.int64)
            for event_id, ts in ((1, onsets), (2, onsets + stim_offset_delay_us)):
                block = np.zeros((5, ts.size), np.int64)
                block[0] = ts
                events.create_dataset(f"EventEntity_{event_id}", data=block)

        segments = rec.create_group("SegmentStream")
        sorter = segments.create_group("Stream_0")
        _stream_attrs(sorter, "Spike Sorter (1);Spike Sorter; Spike Data1", SORTER_GUID, DETECTOR_GUID, "Spike", "Segment")
        detector = segments.create_group("Stream_1")
        _stream_attrs(detector, "Spike Detector (1);Spike Detector; Spike Data1", DETECTOR_GUID, FILTER_GUID, "Spike", "Segment")
        spikes = {k: np.asarray(v, np.int64) for k, v in (detector_spikes_us or {}).items()}
        det_info = np.zeros(len(spikes), INFO_SEGMENT)
        sort_rows = []
        for row, (lbl, ts) in enumerate(spikes.items()):
            det_info[row] = (row, 0, 0, f"E-00303 {lbl}".encode(), 1000, 2000, b"Cutout", str(row).encode())
            detector.create_dataset(f"SegmentData_{row}", data=np.zeros((31, ts.size), np.int32))
            detector.create_dataset(f"SegmentData_ts_{row}", data=ts[None, :])
            for unit, part in ((1, ts[0::2]), (2, ts[1::2])):
                sort_rows.append((lbl, unit, part))
        detector.create_dataset("InfoSegment", data=det_info)
        # Sorter SegmentIDs run backwards relative to the table rows: decoding must use the ID.
        n_sort = len(sort_rows)
        sort_info = np.zeros(n_sort, INFO_SEGMENT)
        for row, (lbl, unit, part) in enumerate(sort_rows):
            seg_id = n_sort - 1 - row
            sort_info[row] = (seg_id, unit, 0, f"E-00303 {lbl}".encode(), 1000, 2000, b"Cutout", b"0")
            sorter.create_dataset(f"SegmentData_{seg_id}", data=np.zeros((31, part.size), np.int32))
            sorter.create_dataset(f"SegmentData_ts_{seg_id}", data=part[None, :])
        sorter.create_dataset("InfoSegment", data=sort_info)

    return {
        "labels": labels,
        "uv": truth_uv,
        "filtered_uv": ((filtered - adzero[:, None]) * gains_uv[:, None]).T,
        "fs": 1e6 / tick_us,
        "t_start_s": first_timestamp_us * 1e-6,
        "t_stop_s": (first_timestamp_us + n_samples * tick_us) * 1e-6,
    }
