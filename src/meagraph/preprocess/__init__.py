"""Preprocessing chains built from SpikeInterface steps plus stimulation-window bridging."""

from __future__ import annotations

from collections.abc import Mapping

from numpy.typing import ArrayLike
from spikeinterface.core import BaseRecording
import spikeinterface.preprocessing as spre

from meagraph.preprocess.blanking import InterpolateWindowsRecording, merge_windows


def detection_band(
    recording: BaseRecording,
    windows: Mapping[str, ArrayLike] | ArrayLike | None = None,
    band_hz: tuple[float, float] = (300.0, 3000.0),
    filter_order: int = 3,
) -> BaseRecording:
    """µV traces with stimulation windows bridged, then a zero-phase Butterworth band-pass.

    This is the legacy ``spikes.py`` chain. The upper edge is capped at fs/2 - 1 Hz, as there.
    Lazy and chunked: nothing is computed until traces are requested.
    """
    rec = spre.scale_to_uV(recording)
    if windows is not None:
        rec = InterpolateWindowsRecording(rec, windows)
    low, high = band_hz
    high = min(high, rec.get_sampling_frequency() / 2 - 1)
    return spre.bandpass_filter(
        rec, freq_min=low, freq_max=high, filter_order=filter_order, ftype="butter",
        direction="forward-backward", dtype="float32",
    )  # fmt: skip


__all__ = ["InterpolateWindowsRecording", "detection_band", "merge_windows"]
