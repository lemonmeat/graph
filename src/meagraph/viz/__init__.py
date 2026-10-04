"""Plotting functions: data in, matplotlib artists out, no file I/O.

Each function draws into an ``ax`` (created when omitted) so it works in the viewer, in
notebooks, and in report figures alike. Requires matplotlib (the ``viewer`` extra).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

INK, MUTED, GRID = "#1c1c1f", "#8a8a94", "#e6e6ea"
ACCENT, WARM = "#2f6f9f", "#b4552d"
STIM_COLORS = (WARM, "#7a4fa3", "#3c8c5a", "#a38a1f")  # one per STG output


def _ax(ax, **subplot_kw):
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(subplot_kw=subplot_kw)
    return ax


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_cube_map(
    grid: np.ndarray,
    values: np.ndarray,
    *,
    active: Sequence[bool] | None = None,
    extent: tuple[int, int, int] = (4, 4, 4),
    cmap: str = "Blues",
    ax=None,
):
    """Electrodes at grid positions (row, col, layer), coloured by ``values``.

    Axes: x = column, y = row (row 1 at the front, as in the pad map), z = layer. QC-active
    electrodes get a dark outline. Returns the scatter artist (for a colourbar).
    """
    ax = _ax(ax, projection="3d")
    rows, cols, layers = extent
    gx, gy, gz = (a.ravel() for a in np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1), np.arange(1, layers + 1)))
    ax.scatter(gx, gy, gz, s=6, color=GRID, depthshade=False)
    edge = [INK if a else MUTED for a in active] if active is not None else MUTED
    width = [1.8 if a else 0.6 for a in active] if active is not None else 0.6
    sc = ax.scatter(
        grid[:, 1], grid[:, 0], grid[:, 2], c=values, s=130, cmap=cmap, vmin=0, vmax=max(float(np.max(values, initial=0)), 1e-9),
        edgecolors=edge, linewidths=width, depthshade=False,
    )  # fmt: skip
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_zlabel("layer")
    ax.set_xticks(range(1, cols + 1))
    ax.set_yticks(range(1, rows + 1))
    ax.set_zticks(range(1, layers + 1))
    ax.set_xlim(0.6, cols + 0.4)
    ax.set_ylim(rows + 0.4, 0.6)
    ax.set_zlim(0.6, layers + 0.4)
    ax.set_box_aspect((1, 1, 2.0))  # tall, so back electrodes stay clickable
    ax.view_init(elev=20, azim=-30)
    return sc


def plot_raster(
    spikes: Mapping[str, np.ndarray],
    order: Sequence[str],
    *,
    stim_onsets: Mapping[str, np.ndarray] | None = None,
    t_range: tuple[float, float] | None = None,
    ax=None,
):
    """One row per electrode in ``order`` (row 0 at the top); stimulation onsets as vertical lines."""
    ax = _ax(ax)
    for i, e in enumerate(order):
        t = spikes.get(e)
        if t is not None and t.size:
            ax.vlines(t, i - 0.4, i + 0.4, color=INK, linewidth=0.5)
    for k, (source, onsets) in enumerate((stim_onsets or {}).items()):
        color = STIM_COLORS[k % len(STIM_COLORS)]
        for t in onsets:
            ax.axvline(t, color=color, linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_ylim(len(order) - 0.5, -0.5)
    if t_range is not None:
        ax.set_xlim(*t_range)
    ax.set_yticks(range(0, len(order), 5))
    ax.set_yticklabels(list(order)[::5], fontsize=7)
    return ax


def plot_trace(
    t_s: np.ndarray,
    y_uv: np.ndarray,
    *,
    spikes_s: np.ndarray | None = None,
    stim_onsets: Mapping[str, np.ndarray] | None = None,
    ax=None,
):
    """A trace with spike markers along the top. Y limits use the 0.5-99.5 percentiles, so
    stimulation artifacts are clipped on purpose."""
    ax = _ax(ax)
    if y_uv.size:
        ax.plot(t_s, y_uv, color=INK, linewidth=0.6)
        lo, hi = np.percentile(y_uv, [0.5, 99.5])
        pad = (hi - lo) * 0.25 + 1e-6
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(t_s[0], t_s[-1])
    t0, t1 = (t_s[0], t_s[-1]) if t_s.size else (0.0, 0.0)
    for k, (source, onsets) in enumerate((stim_onsets or {}).items()):
        color = STIM_COLORS[k % len(STIM_COLORS)]
        for s in onsets[(onsets >= t0) & (onsets <= t1)]:
            ax.axvline(s, color=color, linewidth=1.0, alpha=0.7)
    if spikes_s is not None and y_uv.size:
        sp = spikes_s[(spikes_s >= t0) & (spikes_s <= t1)]
        ax.scatter(sp, np.full(sp.size, ax.get_ylim()[1]), marker="v", s=22, color=ACCENT, zorder=3, clip_on=False)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("µV")
    _clean(ax)
    return ax


def plot_waveforms(waveforms_uv: np.ndarray, cutout_ms: tuple[float, float], *, limit: int = 300, seed: int = 0, ax=None):
    """Up to ``limit`` cutouts (random subset, fixed seed) and their mean. ``waveforms_uv`` is (n, samples)."""
    ax = _ax(ax)
    w = np.asarray(waveforms_uv)
    if w.shape[0] > limit:
        w = w[np.sort(np.random.default_rng(seed).choice(w.shape[0], limit, replace=False))]
    if w.size:
        tw = np.linspace(-cutout_ms[0], cutout_ms[1], w.shape[1])
        ax.plot(tw, w.T, color=ACCENT, alpha=0.08, linewidth=0.6)
        ax.plot(tw, w.mean(axis=0), color=WARM, linewidth=1.8)
    ax.set_xlabel("ms from detection")
    ax.set_ylabel("µV")
    _clean(ax)
    return ax


def plot_ccg(counts: np.ndarray, lag_edges_ms: np.ndarray, *, window_ms: tuple[float, float] | None = None,
             baseline: np.ndarray | None = None, ax=None):  # fmt: skip
    """A cross-correlogram (target spikes at each lag after source spikes), with the synaptic
    window shaded and an optional baseline curve."""
    ax = _ax(ax)
    centres = (lag_edges_ms[:-1] + lag_edges_ms[1:]) / 2
    ax.bar(centres, counts, width=np.diff(lag_edges_ms), color=INK, linewidth=0)
    if baseline is not None:
        ax.plot(centres, baseline, color=WARM, linewidth=1.2)
    if window_ms is not None:
        ax.axvspan(*window_ms, color=ACCENT, alpha=0.15, zorder=0)
    ax.axvline(0, color=MUTED, linewidth=0.6)
    ax.set_xlabel("lag (ms), target after source")
    ax.set_ylabel("count")
    _clean(ax)
    return ax


__all__ = [
    "ACCENT", "GRID", "INK", "MUTED", "STIM_COLORS", "WARM",
    "plot_ccg", "plot_cube_map", "plot_raster", "plot_trace", "plot_waveforms",
]  # fmt: skip
