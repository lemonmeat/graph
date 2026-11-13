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


def plot_graph_3d(positions_um: np.ndarray, node_ids: Sequence[str], edges: Sequence[tuple[int, int]], *,
                  directed: bool = True, values: np.ndarray | None = None, background_um: np.ndarray | None = None,
                  inhibitory: Sequence[bool] | None = None, ax=None):  # fmt: skip
    """Nodes at their 3D positions, coloured by ``values`` (e.g. firing rate), with ``edges``
    as (source index, target index) arrows (lines if undirected); edges flagged ``inhibitory``
    are blue, the others orange. ``background_um`` draws the other electrodes faintly, for
    context. Works for any probe; planar probes have z = 0."""
    ax = _ax(ax, projection="3d")
    pos = np.asarray(positions_um, dtype=np.float64)
    if background_um is not None and len(background_um):
        bg = np.asarray(background_um, dtype=np.float64)
        ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=10, color=GRID, depthshade=False, gid="background")
    vals = np.zeros(len(pos)) if values is None else np.asarray(values, dtype=np.float64)
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=vals, s=110, cmap="Blues", vmin=0,
                    vmax=max(float(np.max(vals, initial=0)), 1e-9), edgecolors=INK, linewidths=1.2, depthshade=False)  # fmt: skip
    span = np.ptp(np.vstack([pos, background_um]) if background_um is not None and len(background_um) else pos, axis=0)
    lift = 0.06 * max(float(span.max()), 1.0)
    for k, label in enumerate(node_ids):
        ax.text(pos[k, 0], pos[k, 1], pos[k, 2] + lift, str(label), fontsize=8, color=INK)
    for k, (i, j) in enumerate(edges):
        d = pos[j] - pos[i]
        color = ACCENT if inhibitory is not None and inhibitory[k] else WARM
        if directed:
            ax.quiver(*pos[i], *d, color=color, linewidth=2, arrow_length_ratio=0.12)
        else:
            ax.plot(*np.column_stack([pos[i], pos[j]]), color=color, linewidth=2)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    ax.set_box_aspect(np.maximum(span, 0.15 * max(float(span.max()), 1.0)))
    ax.view_init(elev=20, azim=-60)
    return sc


def _pair_rows(result, node: int | None, n: int) -> list[dict]:
    """The ``n`` pairs to show: the strongest significant edges overall (``node`` None), or the
    strongest edges touching ``node``, topped up with its closest tested non-significant pairs
    (smallest p; ``significant`` False) when it has fewer than ``n`` edges."""
    rows = [dict(e, significant=True) for e in result.edges()]
    if node is None:
        return rows[:n]
    uid = result.node_ids[node]
    rows = [r for r in rows if uid in (r["source"], r["target"])][:n]
    if len(rows) < n:
        tested = result.tested & ~result.significant & np.isfinite(result.weights)
        cand = [(i, node) for i in range(len(result.node_ids)) if tested[i, node]]
        cand += [(node, j) for j in range(len(result.node_ids)) if tested[node, j]]
        if not result.directed:  # one entry per unordered pair
            cand = list({tuple(sorted(c)): c for c in cand}.values())
        # smallest p first (methods without p-values: largest |weight| first)
        cand.sort(key=lambda ij: (np.nan_to_num(result.p_values[ij], nan=np.inf), -abs(result.weights[ij])))
        for i, j in cand[: n - len(rows)]:
            rows.append(dict(source=result.node_ids[i], target=result.node_ids[j], weight=float(result.weights[i, j]),
                             delay_ms=float(result.delays_ms[i, j]), p_value=float(result.p_values[i, j]),
                             significant=False))  # fmt: skip
    return rows


def _node_info(result, node: int) -> str:
    uid = result.node_ids[node]
    edges = result.edges()
    out = sum(e["source"] == uid for e in edges)
    inn = sum(e["target"] == uid for e in edges)
    lines = [f"electrode {uid}",
             f"{int(result.n_spikes[node])} spikes, {result.n_spikes[node] / result.duration_s:.2f} Hz"]  # fmt: skip
    if result.positions_um is not None:
        x, y, z = result.positions_um[node]
        lines.append(f"position ({x:.0f}, {y:.0f}, {z:.0f}) µm")
    lines.append(f"edges: {out} out, {inn} in" if result.directed else f"edges: {out + inn}")
    lines.append("click it again (or Esc) to show the whole graph")
    return "\n".join(lines)


def plot_connectivity(result, ccg: np.ndarray | None = None, lag_edges_ms: np.ndarray | None = None, *,
                      background_um: np.ndarray | None = None, background_ids: Sequence[str] | None = None,
                      max_ccgs: int = 4, title: str = "", fig=None):  # fmt: skip
    """Overview figure of one :class:`~meagraph.connectivity.ConnectivityResult`: the graph in 3D,
    and up to ``max_ccgs`` (4) correlograms in a 2 x 2 grid, strongest significant edges first.
    ``ccg`` is ``(n, n, bins)`` source→target, as from ``meagraph.connectivity.cross_correlograms``.

    Interactive (in a window): clicking an electrode shows its label, rate and edge counts, and
    swaps the correlograms for the strongest edges of that electrode; if it has fewer than
    ``max_ccgs`` edges, its tested non-significant pairs with the smallest p fill the rest, marked "n.s.".
    Clicking a background electrode (``background_ids``, aligned with ``background_um``) says
    it is not in the graph. Clicking the selected electrode again, or Esc, restores the overview.
    """
    import matplotlib.pyplot as plt

    edges = result.edges()
    index = {u: k for k, u in enumerate(result.node_ids)}
    pairs = [(index[e["source"]], index[e["target"]]) for e in edges]
    n_cols = 3 if ccg is not None else 1
    fig = fig or plt.figure(figsize=(5.2 + 3.4 * (n_cols - 1), 8.0 if ccg is not None else 5.0))
    grid = fig.add_gridspec(2, n_cols, width_ratios=[1.5] + [1] * (n_cols - 1), height_ratios=[1, 1])
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    sc = None
    if result.positions_um is None:
        ax.text2D(0.5, 0.5, "no electrode positions saved", ha="center", transform=ax.transAxes)
    else:
        sc = plot_graph_3d(result.positions_um, result.node_ids, pairs, directed=result.directed,
                           values=result.n_spikes / result.duration_s, background_um=background_um,
                           inhibitory=[result.signed and e["weight"] < 0 for e in edges], ax=ax)  # fmt: skip
    kinds = "; orange = excitatory, blue = inhibitory" if result.signed else ""
    ax.set_title(f"{title}\n{result.method}: {len(edges)} edge(s); node colour = rate{kinds}\n"
                 "click an electrode for its correlograms", fontsize=9)  # fmt: skip
    info = fig.text(0.01, 0.01, "", fontsize=8, family="monospace", va="bottom", color=INK)
    panels = [fig.add_subplot(grid[k // 2, 1 + k % 2]) for k in range(max_ccgs if ccg is not None else 0)]
    window = result.params.get("window_ms")
    arrow = "→" if result.directed else "–"
    state = {"node": None, "ring": None}

    def draw(node: int | None) -> None:
        state["node"] = node
        rows = _pair_rows(result, node, len(panels))
        for a in panels:
            a.clear()
            a.set_visible(False)
        for a, e in zip(panels, rows):
            a.set_visible(True)
            i, j = index[e["source"]], index[e["target"]]
            plot_ccg(ccg[i, j], lag_edges_ms, window_ms=tuple(window) if window else None, ax=a)
            delay = f", {e['delay_ms']:.2f} ms" if np.isfinite(e["delay_ms"]) else ""
            kind = (" inh" if e["weight"] < 0 else " exc") if result.signed and e["significant"] else ""
            stat = f"p {e['p_value']:.1g}" if np.isfinite(e["p_value"]) else f"w {e['weight']:.3g}"
            ns = "" if e["significant"] else " (n.s.)"
            a.set_title(f"{e['source']} {arrow} {e['target']}{kind}{ns}\n{stat}{delay}", fontsize=9,
                        color=INK if e["significant"] else MUTED)  # fmt: skip
        if panels and not rows:
            panels[0].set_visible(True)
            panels[0].set_axis_off()
            panels[0].text(0.5, 0.5, "no tested pairs" if node is not None else "no significant edges",
                           ha="center", va="center", color=MUTED, transform=panels[0].transAxes)  # fmt: skip
        if state["ring"] is not None:
            state["ring"].remove()
            state["ring"] = None
        if node is None:
            info.set_text(f"top {len(rows)} of {len(edges)} edge(s), strongest first" if rows else "")
        else:
            info.set_text(_node_info(result, node))
            if result.positions_um is not None:
                p = result.positions_um[node]
                state["ring"] = ax.scatter([p[0]], [p[1]], [p[2]], s=320, facecolors="none", edgecolors=WARM,
                                           linewidths=2.5, depthshade=False)  # fmt: skip
        fig.canvas.draw_idle()

    def on_pick(event) -> None:
        idx = getattr(event, "ind", None)
        if idx is None or not len(idx):
            return
        if event.artist is sc:
            node = int(idx[0])
            draw(None if node == state["node"] else node)
        elif event.artist.get_gid() == "background" and background_ids is not None:
            uid = str(background_ids[int(idx[0])])
            if uid in index:  # drawn under a graph node
                draw(None if index[uid] == state["node"] else index[uid])
            else:
                draw(None)
                info.set_text(f"electrode {uid}: not in this graph (not QC-active or too few spikes)")
                fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key == "escape":
            draw(None)

    if sc is not None:
        sc.set_picker(6)
        for coll in ax.collections:
            if coll.get_gid() == "background":
                coll.set_picker(4)
    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig._meagraph_select = draw  # keeps the callbacks reachable; lets scripts and tests select a node
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.14, wspace=0.3, hspace=0.55)
    draw(None)
    return fig


__all__ = [
    "ACCENT", "GRID", "INK", "MUTED", "STIM_COLORS", "WARM",
    "plot_ccg", "plot_connectivity", "plot_cube_map", "plot_graph_3d", "plot_raster", "plot_trace", "plot_waveforms",
]  # fmt: skip
