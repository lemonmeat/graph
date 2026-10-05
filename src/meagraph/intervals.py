"""Time intervals as ``(n, 2)`` arrays of ``[start, stop]`` rows: blanking windows in samples,
stimulation and burst periods in seconds."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def merge_intervals(intervals: ArrayLike) -> np.ndarray:
    """Sort intervals and merge those that overlap or touch; empty intervals are dropped."""
    a = np.asarray(intervals).reshape(-1, 2)
    a = a[a[:, 1] > a[:, 0]]
    if a.size == 0:
        return a.copy()
    a = a[np.argsort(a[:, 0], kind="stable")]
    merged = [a[0].copy()]
    for lo, hi in a[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append(np.array([lo, hi], dtype=a.dtype))
    return np.array(merged, dtype=a.dtype)


def inside(points: ArrayLike, intervals: np.ndarray) -> np.ndarray:
    """Boolean mask of ``points`` within any closed interval of a merged ``(n, 2)`` array."""
    p = np.asarray(points)
    if intervals.size == 0:
        return np.zeros(p.shape, dtype=bool)
    k = np.searchsorted(intervals[:, 0], p, side="right") - 1
    return (k >= 0) & (p <= intervals[np.maximum(k, 0), 1])
