"""Minimal reader for Multi Channel Systems (MCS) HDF5 recordings."""
import re

import h5py
import numpy as np

_REC = "Data/Recording_0"


def _s(x):
    """Decode HDF5 bytes / numpy scalars to a plain str."""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", "replace")
    return str(x)


def electrode_number(label):
    """'E-00303 47' -> '47'; '47' -> '47'. The electrode id is the last token."""
    toks = _s(label).replace(";", " ").split()
    return toks[-1] if toks else ""


def mea_geometry(labels, pitch=200.0):
    """(n, 2) x/y in um for MCS 8x8 MEA labels where label = <column><row> (e.g. '47').

    Columns run left to right, rows top to bottom (the viewer inverts y for display).
    """
    xy = np.full((len(labels), 2), np.nan)
    for i, l in enumerate(labels):
        m = re.fullmatch(r"(\d)(\d)", electrode_number(l))
        if m:
            xy[i] = ((int(m.group(1)) - 1) * pitch, (int(m.group(2)) - 1) * pitch)
    return xy


# Electrode number = (row, col, layer) in the 4x4x4 MEA
MEA_CUBE = {
    # top edge, left to right
    "33": (4, 3, 2), "21": (4, 2, 2), "32": (4, 1, 2), "31": (4, 4, 1), "44": (4, 3, 1),
    "43": (4, 2, 1), "41": (4, 1, 1), "42": (3, 4, 4), "52": (3, 3, 4), "51": (3, 2, 4),
    "53": (3, 1, 4), "54": (3, 4, 3), "61": (3, 3, 3), "62": (3, 2, 3), "71": (3, 1, 3),
    # right edge, top to bottom
    "63": (3, 4, 2), "72": (3, 3, 2), "82": (3, 2, 2), "73": (3, 1, 2), "83": (3, 4, 1),
    "64": (3, 3, 1), "74": (3, 2, 1), "84": (3, 1, 1), "85": (2, 1, 1), "75": (2, 2, 1),
    "65": (2, 3, 1), "86": (2, 4, 1), "76": (2, 1, 2), "87": (2, 2, 2), "77": (2, 3, 2),
    "66": (2, 4, 2),
    # bottom edge, left to right
    "36": (1, 3, 2), "28": (1, 2, 2), "37": (1, 1, 2), "38": (1, 4, 1), "45": (1, 3, 1),
    "46": (1, 2, 1), "48": (1, 1, 1), "47": (2, 4, 4), "57": (2, 3, 4), "58": (2, 2, 4),
    "56": (2, 1, 4), "55": (2, 4, 3), "68": (2, 3, 3), "67": (2, 2, 3), "78": (2, 1, 3),
    # left edge, top to bottom
    "22": (4, 4, 2), "12": (4, 1, 3), "23": (4, 2, 3), "13": (4, 3, 3), "34": (4, 4, 3),
    "24": (4, 1, 4), "14": (4, 2, 4), "25": (1, 1, 4), "35": (1, 4, 3), "16": (1, 3, 3),
    "26": (1, 2, 3), "17": (1, 1, 3), "27": (1, 4, 2),
}
assert len(set(MEA_CUBE.values())) == len(MEA_CUBE)   # no two electrodes share a cell


def cube_geometry(labels):
    """(n, 3) array of (row, col, layer), NaN for electrodes not in the cube (e.g. REF 15)."""
    out = np.full((len(labels), 3), np.nan)
    for i, l in enumerate(labels):
        rcl = MEA_CUBE.get(electrode_number(l))
        if rcl:
            out[i] = rcl
    return out


class Channel:
    def __init__(self, label, row, scale_uv, zero):
        self.label, self.row, self.scale_uv, self.zero = label, row, scale_uv, zero


class AnalogStream:
    def __init__(self, grp, name):
        self.grp = grp
        self.data = grp["ChannelData"]
        info = grp["InfoChannel"][:]
        self.label = name
        self.channels = []
        for r in info:
            scale = float(r["ConversionFactor"]) * 10.0 ** int(r["Exponent"]) * 1e6   # -> uV
            self.channels.append(Channel(_s(r["Label"]), int(r["RowIndex"]), scale, int(r["ADZero"])))
        self.labels = [c.label for c in self.channels]
        self.n_channels, self.n_samples = self.data.shape
        tick_us = float(info[0]["Tick"])
        self.fs = 1e6 / tick_us
        ts = grp["ChannelDataTimeStamps"][:]
        self.t_start = float(ts[0, 0]) * 1e-6          # first sample's timestamp (s)
        self.duration = self.n_samples / self.fs
        # Rows in the dataset are not guaranteed to be in InfoChannel order.
        self._by_label = {c.label: i for i, c in enumerate(self.channels)}
        self._by_elec = {electrode_number(c.label): i for i, c in enumerate(self.channels)}

    def index_of(self, label):
        """Channel index from a full label, an electrode number, or an int."""
        if isinstance(label, (int, np.integer)):
            return int(label)
        label = _s(label)
        if label in self._by_label:
            return self._by_label[label]
        e = electrode_number(label)
        if e in self._by_elec:
            return self._by_elec[e]
        raise KeyError(label)

    def read_uv(self, t0=0.0, t1=None, channels=None):
        """Samples in [t0, t1) seconds as float32 uV, shape (n_channels, n_samples)."""
        t1 = self.duration if t1 is None else t1
        i0 = max(int(round(t0 * self.fs)), 0)
        i1 = min(int(round(t1 * self.fs)), self.n_samples)
        idx = list(range(self.n_channels)) if channels is None else [self.index_of(c) for c in channels]
        out = np.empty((len(idx), max(i1 - i0, 0)), np.float32)
        if i1 <= i0:
            return out
        # h5py wants increasing indices for fancy selection; read each row directly.
        for k, ci in enumerate(idx):
            c = self.channels[ci]
            raw = self.data[c.row, i0:i1].astype(np.float32)
            out[k] = (raw - c.zero) * np.float32(c.scale_uv)
        return out


class McsFile:
    def __init__(self, path):
        self.path = path
        self.h5 = h5py.File(path, "r")
        self.rec = self.h5[_REC]

    def close(self):
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -- analog ------------------------------------------------------------- #
    def _streams(self, kind):
        if kind not in self.rec:
            return []
        g = self.rec[kind]
        return [g[k] for k in sorted(g, key=lambda s: int(s.rsplit("_", 1)[1]))]

    def n_analog_streams(self):
        return len(self._streams("AnalogStream"))

    def analog_stream(self, i=0):
        streams = self._streams("AnalogStream")
        if not streams:
            raise ValueError("no AnalogStream in file")
        grp = streams[i]
        # The per-stream label lives in the InfoChannel 'RawDataType'/'GroupID'; files
        # from DataManager use the HDF5 attr 'Label' when present, else the stream name.
        label = _s(grp.attrs["Label"]) if "Label" in grp.attrs else grp.name.rsplit("/", 1)[1]
        return AnalogStream(grp, label)

    # -- events ------------------------------------------------------------- #
    def triggers(self):
        """{event label: times in s}. EventEntity_K takes its label from InfoEvent row K."""
        out = {}
        for grp in self._streams("EventStream"):
            info = grp["InfoEvent"][:]
            for name in grp:
                if not name.startswith("EventEntity_"):
                    continue
                k = int(name.rsplit("_", 1)[1])
                label = " ".join(_s(info[k]["Label"]).split()) if k < len(info) else name
                t = np.asarray(grp[name][0], dtype=np.float64) * 1e-6
                key = label
                while key in out:          # same label in several event streams
                    key += "'"
                out[key] = t
        return out

    def segment_triggers(self):
        """{'Stream_i/SegmentData_ts_k': (label, times in s)} for every segment entity."""
        out = {}
        for si, grp in enumerate(self._streams("SegmentStream")):
            info = grp["InfoSegment"][:]
            for k in range(len(info)):
                name = f"SegmentData_ts_{k}"
                if name in grp:
                    out[f"Stream_{si}/{name}"] = (_s(info[k]["Label"]),
                                                  np.asarray(grp[name], dtype=np.float64).ravel() * 1e-6)
        return out

    # -- segment events ---------------------------------------------------- #
    def segment_events(self, stream=0):
        """{electrode number: event times in s} from SegmentStream `stream`.

        Entities from Stream_0 can be several per electrode (sorted units); they are merged.
        """
        streams = self._streams("SegmentStream")
        if stream >= len(streams):
            raise ValueError(f"file has {len(streams)} SegmentStreams, no stream {stream}")
        grp = streams[stream]
        info = grp["InfoSegment"][:]
        out = {}
        for k in range(len(info)):
            name = f"SegmentData_ts_{k}"
            if name not in grp:
                continue
            e = electrode_number(_s(info[k]["Label"]))
            t = np.asarray(grp[name], dtype=np.float64).ravel() * 1e-6
            out[e] = np.sort(np.concatenate([out[e], t])) if e in out else t
        return out
