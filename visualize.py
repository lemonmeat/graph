import argparse
import glob
import os

import h5py
import matplotlib
import numpy as np

from mcs import McsFile, electrode_number, cube_geometry, _s
from spikes import load_spikes

INK, MUTED, GRID = "#1c1c1f", "#8a8a94", "#e6e6ea"
ACCENT, WARM = "#2f6f9f", "#b4552d"


def print_tree(path):
    """Print groups, dataset shapes and attributes -- the 'what is in this file' view."""
    def visit(name, obj):
        depth = name.count("/")
        if isinstance(obj, h5py.Dataset):
            # Segment/event entities repeat hundreds of times; show only a few.
            base = name.rsplit("/", 1)[1]
            if base.startswith(("SegmentData_", "EventEntity_")) and not base.endswith(("_0", "_1", "_2")):
                return
            print("  " * depth + f"{base}  {obj.shape} {obj.dtype}")
        else:
            print("  " * depth + name.rsplit("/", 1)[-1] + "/")
    with h5py.File(path, "r") as f:
        print("root attributes:", {k: _s(v) for k, v in f.attrs.items()})
        f.visititems(visit)
        print("(SegmentData_*/EventEntity_* beyond the first few are hidden)")


class Recording:
    """Everything the viewer needs"""

    def __init__(self, path):
        self.path = path
        self.mcs = McsFile(path)
        m = self.mcs
        self.n_streams = m.n_analog_streams()
        self.streams = [m.analog_stream(i) for i in range(self.n_streams)]
        self.stream_names = [self._short(s.label) for s in self.streams]
        self.fs = self.streams[0].fs
        self.duration = max(s.duration for s in self.streams)

        # Electrode order / geometry come from stream 0.
        self.labels = [electrode_number(l) for l in self.streams[0].labels]
        self.pos = cube_geometry(self.labels)   # (row, col, layer) per electrode
        self.row_of = {e: i for i, e in enumerate(self.labels)}

        self.events = {k: v for k, v in m.triggers().items() if v.size}
        self.seg_src, self.seg_events, self.cutouts = self._load_seg_events()
        self.stim_times = self._stim_times()

    @staticmethod
    def _short(label):
        # 'Filter (3);Filter; Filter Data3' -> 'Filter (3)'
        return label.split(";")[0] or label

    def _load_seg_events(self):
        """Spikes from the `<recording>.spikes.h5` sidecar written by spikes.py."""
        got = load_spikes(self.path)
        if got is None:
            print("no spike sidecar found -- run: python spikes.py", os.path.basename(self.path))
            self.cutout_ms = (-1.0, 2.0)
            return "no spikes (run spikes.py)", {}, {}
        ts, wf, self.cutout_ms, params = got
        return f"spikes.py, {params['k']:g} sigma", ts, wf

    def _stim_times(self):
        """Union of stimulation event times; 'Marker Start' events mark pulse onsets."""
        pick = [v for k, v in self.events.items() if "Start" in k] or list(self.events.values())
        return np.sort(np.concatenate(pick)) if pick else np.array([])

    def cutout_uv(self, elec, limit=300):
        w = self.cutouts.get(elec)
        if w is None or w.shape[1] == 0:
            return None
        n = w.shape[1]
        return w if n <= limit else w[:, np.sort(np.random.default_rng(0).choice(n, limit, replace=False))]

    def rates(self):
        return np.array([self.seg_events.get(e, np.array([])).size / self.duration for e in self.labels])

class Viewer:
    def __init__(self, rec, t0=0.0, win=2.0, electrode=None, stream=0, bandpass=False):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons, RadioButtons, Slider
        self.plt, self.rec = plt, rec
        self.elec = electrode if electrode in rec.row_of else rec.labels[0]
        self.stream = stream
        self.bandpass = bandpass
        self.win = win
        self.t0 = float(np.clip(t0, 0, max(rec.duration - win, 0)))

        plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                             "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
        fig = self.fig = plt.figure(figsize=(15, 9.2))
        fig.canvas.manager.set_window_title(os.path.basename(rec.path)) if fig.canvas.manager else None
        gs = fig.add_gridspec(3, 3, left=0.05, right=0.985, top=0.93, bottom=0.15,
                              width_ratios=[1.0, 1.35, 1.0], height_ratios=[1.25, 1.0, 0.0001],
                              hspace=0.35, wspace=0.22)
        self.ax_map = fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_raster = fig.add_subplot(gs[0, 1:])
        self.ax_trace = fig.add_subplot(gs[1, :2])
        self.ax_wave = fig.add_subplot(gs[1, 2])

        fig.suptitle(os.path.basename(rec.path), x=0.05, ha="left", fontsize=10, color=MUTED)
        self._draw_map()
        self._draw_raster()

        # Controls along the bottom.
        self.s_t = Slider(fig.add_axes([0.08, 0.095, 0.50, 0.025]), "start (s)", 0, max(rec.duration - 0.05, 0.1),
                          valinit=self.t0, color=ACCENT)
        self.s_w = Slider(fig.add_axes([0.08, 0.055, 0.50, 0.025]), "window (s)", 0.05, 20, valinit=win,
                          valstep=0.05, color=ACCENT)
        self.s_t.on_changed(self._on_slider)
        self.s_w.on_changed(self._on_slider)
        rax = fig.add_axes([0.63, 0.02, 0.22, 0.11], frameon=False)
        self.radio = RadioButtons(rax, rec.stream_names, active=stream, activecolor=ACCENT)
        self.radio.on_clicked(lambda lab: self._set(stream=rec.stream_names.index(lab)))
        cax = fig.add_axes([0.87, 0.06, 0.12, 0.06], frameon=False)
        self.check = CheckButtons(cax, ["band-pass\n300-3000 Hz"], [bandpass])
        self.check.on_clicked(lambda _: self._set(bandpass=not self.bandpass))
        fig.text(0.63, 0.135, "analog stream", color=MUTED, fontsize=8)

        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw_dynamic()

    def _draw_map(self):
        ax, rec = self.ax_map, self.rec
        rates = rec.rates()
        ok = ~np.isnan(rec.pos[:, 0])
        rcl = rec.pos[ok]
        x, y, z = rcl[:, 1], rcl[:, 0], rcl[:, 2]     # x = col, y = row, z = layer
        ax.clear()
        # Faint lattice so empty cells of the 4x4x4 cube are visible.
        g = np.arange(1, 5)
        gx, gy, gz = (a.ravel() for a in np.meshgrid(g, g, g))
        ax.scatter(gx, gy, gz, s=6, color=GRID, depthshade=False, zorder=1)
        sc = ax.scatter(x, y, z, c=rates[ok], s=130, cmap="Blues", vmin=0, vmax=max(rates.max(), 1e-9),
                        edgecolors=MUTED, linewidths=0.6, depthshade=False)
        self.cube_xyz = np.column_stack([x, y, z])
        self.cube_idx = np.flatnonzero(ok)
        self.sel_ring = ax.scatter([], [], [], s=280, facecolors="none", edgecolors=WARM, linewidths=2.2,
                                   depthshade=False)
        self.map_label = ax.text2D(0.02, 0.92, "", transform=ax.transAxes, fontsize=9, color=WARM)
        ax.set_xlabel("col"); ax.set_ylabel("row"); ax.set_zlabel("layer")
        for setter in (ax.set_xticks, ax.set_yticks, ax.set_zticks):
            setter(g)
        ax.set_xlim(0.6, 4.4); ax.set_ylim(4.4, 0.6); ax.set_zlim(0.6, 4.4)   # row 1 at the front/top, as in the pad map
        ax.set_box_aspect((1, 1, 2.0))     # tall: layers spread apart so back electrodes stay reachable
        ax.view_init(elev=20, azim=-30)
        ax.disable_mouse_rotation()
        ax.set_title(f"4x4x4 map: spikes/s, mean {rates.mean():.2f}\n(click to select)",
                     loc="left", fontsize=9)
        cb = self.fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01, shrink=0.7)
        cb.outline.set_visible(False)

    def _draw_raster(self):
        ax, rec = self.ax_raster, self.rec
        ax.clear()
        order = rec.labels
        for i, e in enumerate(order):
            t = rec.seg_events.get(e)
            if t is not None and t.size:
                ax.vlines(t, i - 0.4, i + 0.4, color=INK, linewidth=0.5)
        for t in rec.stim_times:
            ax.axvline(t, color=WARM, linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlim(0, rec.duration)
        ax.set_yticks(range(0, len(order), 5)); ax.set_yticklabels(order[::5], fontsize=7)
        ax.set_xlabel("time (s)  - click to move window")
        nsp = sum(v.size for v in rec.seg_events.values())
        ax.set_title(f"Spike raster ({nsp} events, {rec.seg_src});  red = {len(rec.stim_times)} stim events",
                     loc="left", fontsize=9)
        self.win_patch = ax.axvspan(0, 1, color=ACCENT, alpha=0.25, zorder=0)
        self.sel_line = ax.axhline(0, color=WARM, linewidth=1.0, alpha=0.6)

    def _read_trace(self):
        rec = self.rec
        st = rec.streams[self.stream]
        t1 = min(self.t0 + self.win, st.duration)
        if t1 <= self.t0:
            return np.array([]), np.array([])
        row = st.index_of(next(l for l in st.labels if electrode_number(l) == self.elec))
        y = st.read_uv(self.t0, t1, channels=[row])[0]
        t = self.t0 + np.arange(y.size) / st.fs
        if self.bandpass and y.size > 50:
            from scipy.signal import butter, sosfiltfilt
            sos = butter(3, [300, min(3000, st.fs / 2 - 1)], btype="band", fs=st.fs, output="sos")
            y = sosfiltfilt(sos, y - y.mean())
        return t, y

    def _draw_dynamic(self):
        rec, ax = self.rec, self.ax_trace
        t, y = self._read_trace()
        ax.clear()
        t_hi = self.t0 + self.win
        if y.size:
            ax.plot(t, y, color=INK, linewidth=0.6)
            lo, hi = np.percentile(y, [0.5, 99.5])
            pad = (hi - lo) * 0.25 + 1e-6
            ax.set_ylim(lo - pad, hi + pad)   # rare stim artefacts are clipped on purpose
        for s in rec.stim_times[(rec.stim_times >= self.t0) & (rec.stim_times <= t_hi)]:
            ax.axvline(s, color=WARM, linewidth=1.0, alpha=0.7)
        sp = rec.seg_events.get(self.elec, np.array([]))
        sp = sp[(sp >= self.t0) & (sp <= t_hi)]
        if sp.size and y.size:
            ax.scatter(sp, np.full(sp.size, ax.get_ylim()[1]), marker="v", s=22, color=ACCENT, zorder=3,
                       clip_on=False)
        ax.set_xlim(self.t0, t_hi)
        ax.set_xlabel("time (s)"); ax.set_ylabel("uV")
        ax.set_title(f"Electrode {self.elec} - {rec.stream_names[self.stream]}"
                     f"{' (band-passed)' if self.bandpass else ''} - {sp.size} spikes in window",
                     loc="left", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # Waveforms
        aw = self.ax_wave
        aw.clear()
        w = rec.cutout_uv(self.elec)
        if w is not None and w.size:
            tw = np.linspace(*rec.cutout_ms, w.shape[0])
            aw.plot(tw, w, color=ACCENT, alpha=0.08, linewidth=0.6)
            aw.plot(tw, w.mean(axis=1), color=WARM, linewidth=1.8)
            aw.set_title(f"Spike waveforms (band-passed), electrode {self.elec} (n={w.shape[1]})",
                         loc="left", fontsize=9)
        else:
            aw.set_title(f"No spikes for {self.elec}", loc="left", fontsize=9)
        aw.set_xlabel("ms from detection"); aw.set_ylabel("uV")
        for s in ("top", "right"):
            aw.spines[s].set_visible(False)

        # markers on the other panels
        self.win_patch.remove()
        self.win_patch = self.ax_raster.axvspan(self.t0, t_hi, color=ACCENT, alpha=0.25, zorder=0)
        self.sel_line.set_ydata([rec.row_of[self.elec]] * 2)
        i = rec.row_of[self.elec]
        r, c, l = rec.pos[i]
        self.sel_ring._offsets3d = ([c], [r], [l])
        self.map_label.set_text(f"electrode {self.elec}  (row {r:.0f}, col {c:.0f}, layer {l:.0f})"
                                if not np.isnan(r) else f"electrode {self.elec}  (not in cube)")
        self.fig.canvas.draw_idle()

    def _set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self._draw_dynamic()

    def _on_slider(self, _):
        self.t0, self.win = float(self.s_t.val), float(self.s_w.val)
        self._draw_dynamic()

    def _on_release(self, ev):
        """A click in the fixed-view 3D map selects the nearest electrode on screen."""
        from mpl_toolkits.mplot3d import proj3d
        if ev.inaxes is not self.ax_map or ev.button != 1:
            return
        ax = self.ax_map
        xs, ys, _ = proj3d.proj_transform(*self.cube_xyz.T, ax.get_proj())
        disp = ax.transData.transform(np.column_stack([xs, ys]))
        d = np.hypot(disp[:, 0] - ev.x, disp[:, 1] - ev.y)
        k = int(np.argmin(d))
        if d[k] < 25:
            self._set(elec=self.rec.labels[self.cube_idx[k]])

    def _on_click(self, ev):
        if ev.inaxes is self.ax_raster and ev.xdata is not None:
            t0 = float(np.clip(ev.xdata - self.win / 2, 0, max(self.rec.duration - self.win, 0)))
            self.s_t.set_val(t0)       # triggers _on_slider
            if ev.ydata is not None:
                i = int(round(ev.ydata))
                if 0 <= i < len(self.rec.labels):
                    self._set(elec=self.rec.labels[i])

    def _on_key(self, ev):
        if ev.key in ("left", "right"):
            step = self.win / 2 * (-1 if ev.key == "left" else 1)
            self.s_t.set_val(float(np.clip(self.t0 + step, 0, max(self.rec.duration - self.win, 0))))


def choose_file(folder):
    files = [f for f in sorted(glob.glob(os.path.join(folder, "*.h5"))) if not f.endswith(".spikes.h5")]
    if not files:
        raise SystemExit(f"no .h5 files in {folder}")
    if len(files) == 1:
        return files[0]
    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)}  ({os.path.getsize(f) / 1e6:.0f} MB)")
    return files[int(input("open which? "))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--tree", action="store_true", help="print the HDF5 structure and exit")
    ap.add_argument("--save", help="render to an image instead of opening a window")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--electrode", default=None, help="electrode number, e.g. 47")
    ap.add_argument("--stream", type=int, default=0, help="analog stream index")
    ap.add_argument("--bandpass", action="store_true")
    a = ap.parse_args()

    path = a.path or choose_file(os.path.dirname(os.path.abspath(__file__)))
    if a.tree:
        print_tree(path)
        return
    if a.save:
        matplotlib.use("Agg")
    rec = Recording(path)
    v = Viewer(rec, a.t0, a.win, a.electrode, a.stream, a.bandpass)
    if a.save:
        v.fig.savefig(a.save, dpi=120)
        print("saved", a.save)
    else:
        v.plt.show()


if __name__ == "__main__":
    main()
