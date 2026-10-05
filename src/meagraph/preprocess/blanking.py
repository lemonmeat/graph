"""Linear bridging of stimulation-artifact windows before filtering (DECISIONS.md D9).

Each window [start, stop) is replaced by a straight line from the last sample before it to the
first sample after it, the rule used by the legacy ``spikes.py``. Bridging before the band-pass
filter stops the artifact from ringing through the filter into neighbouring samples.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike
from spikeinterface.core import BaseRecording
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from meagraph.intervals import merge_intervals


def merge_windows(windows: ArrayLike, n_samples: int) -> np.ndarray:
    """Clip ``[start, stop)`` sample windows to the recording, sort them, merge overlapping or touching ones."""
    return merge_intervals(np.clip(np.asarray(windows, dtype=np.int64).reshape(-1, 2), 0, n_samples))


class InterpolateWindowsRecording(BasePreprocessor):
    """Bridge sample windows with straight lines, per channel.

    Parameters
    ----------
    recording
        Single-segment recording.
    windows
        One ``(n, 2)`` array of ``[start, stop)`` sample indices shared by all channels, or a
        mapping ``{channel_id: (n, 2) array}`` for per-channel windows. Channels missing from
        the mapping are left untouched.
    """

    def __init__(self, recording: BaseRecording, windows: Mapping[str, ArrayLike] | ArrayLike):
        if recording.get_num_segments() != 1:
            raise ValueError("InterpolateWindowsRecording supports single-segment recordings")
        BasePreprocessor.__init__(self, recording, dtype="float32")
        n = recording.get_num_samples(0)
        ids = [str(c) for c in recording.channel_ids]
        if isinstance(windows, Mapping):
            per_channel = {str(k): merge_windows(v, n) for k, v in windows.items()}
            unknown = set(per_channel) - set(ids)
            if unknown:
                raise KeyError(f"windows given for unknown channels: {sorted(unknown)}")
        else:
            shared = merge_windows(windows, n)
            per_channel = {c: shared for c in ids}
        empty = np.zeros((0, 2), dtype=np.int64)
        ordered = [per_channel.get(c, empty) for c in ids]
        self.add_recording_segment(InterpolateWindowsSegment(recording._recording_segments[0], ordered, n))
        self._kwargs = {"recording": recording, "windows": {c: w.tolist() for c, w in per_channel.items()}}


class InterpolateWindowsSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, windows: list[np.ndarray], num_samples: int):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self._windows = windows
        self._n = num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        start = 0 if start_frame is None else int(start_frame)
        end = self._n if end_frame is None else int(end_frame)
        channels = np.arange(len(self._windows))[channel_indices if channel_indices is not None else slice(None)]

        # Find windows touching [start, end) and widen the read to include their anchor samples.
        lo_read, hi_read = start, end
        relevant = []
        for k, ch in enumerate(np.atleast_1d(channels)):
            w = self._windows[ch]
            if w.size == 0:
                continue
            hit = w[(w[:, 1] > start) & (w[:, 0] < end)]
            if hit.size:
                relevant.append((k, hit))
                lo_read = min(lo_read, int(hit[0, 0]) - 1)
                hi_read = max(hi_read, int(hit[-1, 1]) + 1)
        lo_read, hi_read = max(lo_read, 0), min(hi_read, self._n)

        traces = self.parent_recording_segment.get_traces(lo_read, hi_read, channel_indices).astype(np.float32)
        for k, hit in relevant:
            x = traces[:, k]
            for lo, hi in hit:
                a, b = lo - lo_read, hi - lo_read
                left = x[a - 1] if lo > 0 else None
                right = x[b] if hi < self._n else None
                if left is None and right is None:
                    x[a:b] = 0.0
                    continue
                left = right if left is None else left
                right = left if right is None else right
                x[a:b] = np.linspace(left, right, b - a)
        return traces[start - lo_read : end - lo_read]
