"""Whole-recording robust noise level in one chunked pass."""

from __future__ import annotations

import numpy as np
from spikeinterface.core import BaseRecording


def median_abs_noise_uv(
    recording: BaseRecording,
    chunk_duration_s: float = 10.0,
    relative_precision: float = 5e-4,
    min_uv: float = 1e-3,
    max_uv: float = 1e5,
) -> np.ndarray:
    """sigma = median(|x|) / 0.6745 per channel over every sample, as in the legacy ``spikes.py``.

    ``recording`` must already be in µV (e.g. from :func:`meagraph.preprocess.detection_band`).
    The median comes from a histogram of |x| with logarithmic bins, so its relative error is
    below ``relative_precision / 2`` on every channel while memory stays constant. Unlike
    SpikeInterface's ``get_noise_levels``, which samples random chunks, this uses all samples
    and is deterministic.
    """
    n = recording.get_num_samples()
    n_ch = recording.get_num_channels()
    log_step = np.log1p(relative_precision)
    n_bins = int(np.ceil(np.log(max_uv / min_uv) / log_step)) + 2
    counts = np.zeros((n_ch, n_bins), dtype=np.int64)
    step = max(int(chunk_duration_s * recording.get_sampling_frequency()), 1)
    for start in range(0, n, step):
        x = np.abs(recording.get_traces(start_frame=start, end_frame=min(start + step, n))).astype(np.float64)
        q = np.floor(np.log(np.maximum(x, min_uv) / min_uv) / log_step).astype(np.int64) + 1
        q[x < min_uv] = 0
        np.minimum(q, n_bins - 1, out=q)
        for c in range(n_ch):
            counts[c] += np.bincount(q[:, c], minlength=n_bins)
    median_bin = np.argmax(np.cumsum(counts, axis=1) >= (n + 1) / 2, axis=1)
    centre = min_uv * np.exp((median_bin - 0.5) * log_step)  # geometric centre of bin [k-1, k) steps above min_uv
    return np.where(median_bin == 0, 0.0, centre) / 0.6745
