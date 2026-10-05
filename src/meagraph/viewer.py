"""Interactive viewer: 4x4x4 electrode selector, spike raster, trace and waveforms.

Rebuilt from the legacy ``visualize.py`` on the meagraph API; the behaviours it preserves are
listed in docs/EXISTING_CODE.md. Launch with ``meagraph view <file.h5>``. Nothing in the core
library imports this module (it needs matplotlib, the ``viewer`` extra).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meagraph import viz
from meagraph.detect.store import find_detection, load_detection
from meagraph.io import McsH5Recording, Session, load_session, read_spikes_sidecar, sidecar_path
from meagraph.probe import DEFAULT_PROBE


@dataclass
class ViewerData:
    """Everything the viewer shows, loaded through the package API."""

    session: Session
    electrodes: tuple[str, ...]  # recording channel order
    grid: np.ndarray  # (n, 3) row, col, layer
    spikes: dict[str, np.ndarray]  # s, recording clock
    waveforms: dict[str, np.ndarray]  # (n_spikes, samples)
    cutout_ms: tuple[float, float]
    spike_source: str
    active: frozenset[str] = frozenset()
    streams: tuple[str, ...] = ()  # analog stream labels in processing order
    _stream_cache: dict[str, McsH5Recording] = field(default_factory=dict)

    @property
    def t_start(self) -> float:
        return self.session.recording.get_start_time()

    @property
    def t_stop(self) -> float:
        rec = self.session.recording
        return rec.get_start_time() + rec.get_total_duration()

    @property
    def stim_onsets(self) -> dict[str, np.ndarray]:
        return {s.source: s.onsets_s for s in self.session.stim}

    def rates_hz(self) -> np.ndarray:
        duration = self.t_stop - self.t_start
        return np.array([self.spikes.get(e, np.zeros(0)).size / duration for e in self.electrodes])

    def stream(self, label: str) -> McsH5Recording:
        if label not in self._stream_cache:
            self._stream_cache[label] = McsH5Recording(self.session.path, stream=label)
        return self._stream_cache[label]


def load_viewer_data(path, probe=DEFAULT_PROBE, stim_site=None, spikes_dir=None) -> ViewerData:
    """Spikes come from ``spikes_dir``, else the newest ``meagraph detect`` result, else a legacy sidecar."""
    session = load_session(path, probe=probe, stim_site=stim_site)
    rec = session.recording
    electrodes = tuple(str(c) for c in rec.channel_ids)
    grid = np.column_stack([rec.get_property(k) for k in ("grid_row", "grid_col", "grid_layer")])
    folder = Path(spikes_dir) if spikes_dir else find_detection(path)
    spikes, waveforms, cutout, source, active = {}, {}, (1.0, 2.0), "no spikes (run: meagraph detect)", frozenset()
    if folder is not None:
        det = load_detection(folder)
        spikes, waveforms, cutout = det.trains.as_dict(), det.waveforms_uv, det.config.cutout_ms
        source = f"meagraph detect ({det.config.profile}, {det.config.threshold_sigma:g} sigma)"
        active = frozenset(det.active_channels)
    elif sidecar_path(path).exists():
        legacy = read_spikes_sidecar(sidecar_path(path))
        spikes = legacy.trains.as_dict()
        waveforms = {e: w.T for e, w in legacy.waveforms_uv.items()}
        cutout, source = legacy.cutout_ms, f"legacy spikes.py ({legacy.params['k']:g} sigma)"
    streams = tuple(s.short_label for s in session.inventory.analog_streams)
    return ViewerData(session, electrodes, grid, spikes, waveforms, cutout, source, active, streams)


def nearest_on_screen(ax3d, xyz: np.ndarray, x_px: float, y_px: float, max_px: float = 25.0) -> int | None:
    """Index of the 3D point drawn closest to a display position, if within ``max_px``."""
    from mpl_toolkits.mplot3d import proj3d

    xs, ys, _ = proj3d.proj_transform(*xyz.T, ax3d.get_proj())
    disp = ax3d.transData.transform(np.column_stack([xs, ys]))
    d = np.hypot(disp[:, 0] - x_px, disp[:, 1] - y_px)
    k = int(np.argmin(d))
    return k if d[k] < max_px else None


class Viewer:
    def __init__(self, data: ViewerData, t0=None, win=2.0, electrode=None, stream="raw", bandpass=False):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons, RadioButtons, Slider

        self.plt, self.data = plt, data
        self.elec = electrode if electrode in data.electrodes else data.electrodes[0]
        self.stream = data.streams[0] if stream == "raw" else stream
        self.bandpass = bandpass
        self.win = win
        latest = max(data.t_stop - win, data.t_start)
        self.t0 = float(np.clip(data.t_start if t0 is None else t0, data.t_start, latest))

        plt.rcParams.update({"font.size": 9, "axes.edgecolor": viz.MUTED, "axes.labelcolor": viz.INK,
                             "xtick.color": viz.MUTED, "ytick.color": viz.MUTED, "text.color": viz.INK})  # fmt: skip
        fig = self.fig = plt.figure(figsize=(15, 9.2))
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(data.session.path.name)
        gs = fig.add_gridspec(3, 3, left=0.05, right=0.985, top=0.93, bottom=0.15, width_ratios=[1.0, 1.35, 1.0],
                              height_ratios=[1.25, 1.0, 0.0001], hspace=0.35, wspace=0.22)  # fmt: skip
        self.ax_map = fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_raster = fig.add_subplot(gs[0, 1:])
        self.ax_trace = fig.add_subplot(gs[1, :2])
        self.ax_wave = fig.add_subplot(gs[1, 2])
        fig.suptitle(data.session.path.name, x=0.05, ha="left", fontsize=10, color=viz.MUTED)
        self._draw_map()
        self._draw_raster()

        self.s_t = Slider(fig.add_axes([0.08, 0.095, 0.50, 0.025]), "start (s)", data.t_start,
                          max(data.t_stop - 0.05, data.t_start + 0.1), valinit=self.t0, color=viz.ACCENT)  # fmt: skip
        self.s_w = Slider(fig.add_axes([0.08, 0.055, 0.50, 0.025]), "window (s)", 0.05, 20, valinit=win, valstep=0.05, color=viz.ACCENT)
        self.s_t.on_changed(self._on_slider)
        self.s_w.on_changed(self._on_slider)
        self.radio = RadioButtons(fig.add_axes([0.63, 0.02, 0.22, 0.11], frameon=False), list(data.streams),
                                  active=data.streams.index(self.stream), activecolor=viz.ACCENT)  # fmt: skip
        self.radio.on_clicked(lambda label: self._set(stream=label))
        self.check = CheckButtons(fig.add_axes([0.87, 0.06, 0.12, 0.06], frameon=False), ["band-pass\n300-3000 Hz"], [bandpass])
        self.check.on_clicked(lambda _: self._set(bandpass=not self.bandpass))
        fig.text(0.63, 0.135, "analog stream", color=viz.MUTED, fontsize=8)

        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw_dynamic()

    # -- static panels ------------------------------------------------------------------- #
    def _draw_map(self):
        d, ax = self.data, self.ax_map
        rates = d.rates_hz()
        sc = viz.plot_cube_map(d.grid, rates, active=[e in d.active for e in d.electrodes], ax=ax)
        self.cube_xyz = d.grid[:, [1, 0, 2]].astype(float)  # x = col, y = row, z = layer
        self.sel_ring = ax.scatter([], [], [], s=280, facecolors="none", edgecolors=viz.WARM, linewidths=2.2, depthshade=False)
        self.map_label = ax.text2D(0.02, 0.92, "", transform=ax.transAxes, fontsize=9, color=viz.WARM)
        ax.disable_mouse_rotation()
        qc = f", QC-active (dark rim): {len(d.active)}" if d.active else ""
        ax.set_title(f"4x4x4 map: spikes/s, mean {rates.mean():.2f}{qc}\n(click to select)", loc="left", fontsize=9)
        cb = self.fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01, shrink=0.7)
        cb.outline.set_visible(False)

    def _draw_raster(self):
        d, ax = self.data, self.ax_raster
        viz.plot_raster(d.spikes, d.electrodes, stim_onsets=d.stim_onsets, t_range=(d.t_start, d.t_stop), ax=ax)
        n_stim = sum(o.size for o in d.stim_onsets.values())
        stim = f";  stim lines: {n_stim} pulses ({', '.join(d.stim_onsets)})" if n_stim else ""
        ax.set_xlabel("time (s)  - click to move window")
        ax.set_title(f"Spike raster ({sum(v.size for v in d.spikes.values())} spikes, {d.spike_source}){stim}", loc="left", fontsize=9)
        self.win_patch = ax.axvspan(self.t0, self.t0 + self.win, color=viz.ACCENT, alpha=0.25, zorder=0)
        self.sel_line = ax.axhline(0, color=viz.WARM, linewidth=1.0, alpha=0.6)

    # -- dynamic panels ------------------------------------------------------------------ #
    def _read_trace(self):
        rec = self.data.stream(self.stream)
        fs = rec.get_sampling_frequency()
        start = max(int(round((self.t0 - rec.get_start_time()) * fs)), 0)
        end = min(int(round((self.t0 + self.win - rec.get_start_time()) * fs)), rec.get_num_samples())
        if end <= start:
            return np.zeros(0), np.zeros(0)
        y = rec.get_traces(start_frame=start, end_frame=end, channel_ids=[self.elec], return_in_uV=True)[:, 0].astype(np.float64)
        t = rec.get_start_time() + np.arange(start, end) / fs
        if self.bandpass and y.size > 50:  # display filter only, as in the legacy viewer
            from scipy.signal import butter, sosfiltfilt

            sos = butter(3, [300, min(3000, fs / 2 - 1)], btype="band", fs=fs, output="sos")
            y = sosfiltfilt(sos, y - y.mean())
        return t, y

    def _draw_dynamic(self):
        d = self.data
        t, y = self._read_trace()
        sp = d.spikes.get(self.elec, np.zeros(0))
        self.ax_trace.clear()
        viz.plot_trace(t, y, spikes_s=sp, stim_onsets=d.stim_onsets, ax=self.ax_trace)
        n_win = int(((sp >= self.t0) & (sp <= self.t0 + self.win)).sum())
        self.ax_trace.set_title(f"Electrode {self.elec} - {self.stream}{' (band-passed)' if self.bandpass else ''}"
                                f" - {n_win} spikes in window", loc="left", fontsize=9)  # fmt: skip

        self.ax_wave.clear()
        w = d.waveforms.get(self.elec, np.zeros((0, 0)))
        viz.plot_waveforms(w, d.cutout_ms, ax=self.ax_wave)
        title = f"Spike waveforms (band-passed), electrode {self.elec} (n={w.shape[0]})" if w.size else f"No spikes for {self.elec}"
        self.ax_wave.set_title(title, loc="left", fontsize=9)

        self.win_patch.remove()
        self.win_patch = self.ax_raster.axvspan(self.t0, self.t0 + self.win, color=viz.ACCENT, alpha=0.25, zorder=0)
        i = d.electrodes.index(self.elec)
        self.sel_line.set_ydata([i, i])
        r, c, layer = d.grid[i]
        self.sel_ring._offsets3d = ([c], [r], [layer])
        qc = ", QC-active" if self.elec in d.active else ""
        self.map_label.set_text(f"electrode {self.elec}  (row {r}, col {c}, layer {layer}){qc}")
        self.fig.canvas.draw_idle()

    # -- interaction --------------------------------------------------------------------- #
    def _set(self, **changes):
        for k, v in changes.items():
            setattr(self, k, v)
        self._draw_dynamic()

    def _clamp(self, t0):
        return float(np.clip(t0, self.data.t_start, max(self.data.t_stop - self.win, self.data.t_start)))

    def _on_slider(self, _):
        self.t0, self.win = float(self.s_t.val), float(self.s_w.val)
        self._draw_dynamic()

    def _on_release(self, event):
        """Left click in the fixed-view 3D map selects the electrode nearest on screen."""
        if event.inaxes is not self.ax_map or event.button != 1:
            return
        k = nearest_on_screen(self.ax_map, self.cube_xyz, event.x, event.y)
        if k is not None:
            self._set(elec=self.data.electrodes[k])

    def _on_click(self, event):
        if event.inaxes is self.ax_raster and event.xdata is not None:
            self.s_t.set_val(self._clamp(event.xdata - self.win / 2))  # triggers _on_slider
            if event.ydata is not None:
                i = int(round(event.ydata))
                if 0 <= i < len(self.data.electrodes):
                    self._set(elec=self.data.electrodes[i])

    def _on_key(self, event):
        if event.key in ("left", "right"):
            step = self.win / 2 * (-1 if event.key == "left" else 1)
            self.s_t.set_val(self._clamp(self.t0 + step))


def main(args: argparse.Namespace) -> int:
    """Entry point for ``meagraph view`` (arguments are parsed in meagraph.cli)."""
    import matplotlib

    from meagraph.cli import _stim_site

    if args.save:
        matplotlib.use("Agg")
    data = load_viewer_data(args.path, probe=args.probe, stim_site=_stim_site(args.stim_site), spikes_dir=args.spikes)
    viewer = Viewer(data, t0=args.t0, win=args.win, electrode=args.electrode, stream=args.stream, bandpass=args.bandpass)
    if args.save:
        viewer.fig.savefig(args.save, dpi=120)
        print("saved", args.save)
    else:
        viewer.plt.show()
    return 0
