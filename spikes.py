"""Spike detection for MCS recordings; writes a SpikeStream sidecar next to each .h5.

Per electrode, on the raw analog stream (Stream 0):
  1. blank stimulation artefacts in the raw trace (linear interpolation) so the filter does not ring,
  2. zero-phase band-pass (default 300-3000 Hz),
  3. noise sigma = median(|x|)/0.6745 (robust to spikes),
  4. negative-going peaks below -k*sigma, at least `refractory` ms apart,
  5. drop peaks inside the stimulation blanking windows, larger than `max_uv`, or whose positive
     rebound within the cutout exceeds the negative peak (ringing artefact, not a spike).

Sidecar `<recording>.spikes.h5` (the recording itself is never modified):
  SpikeStream/Electrode_<n>/ts          spike times in s, on the same clock as the analog stream
  SpikeStream/Electrode_<n>/amp         negative-peak amplitude of the filtered trace (uV)
  SpikeStream/Electrode_<n>/waveforms   (samples, spikes) filtered cutouts (uV)
  attributes: sigma_uv per electrode; detection parameters and the source file on the root.

Usage:  python spikes.py [file.h5 ...]      (no args: every .h5 next to this script)
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from mcs import McsFile, electrode_number

SUFFIX = ".spikes.h5"


def spikes_path(h5_path):
    return os.path.splitext(h5_path)[0] + SUFFIX


def stim_windows(mcs, t_start, fs, n, pre_ms, post_ms):
    """Sample-index [a, b) blanking windows around every stimulation event (both Start and Stop)."""
    times = np.sort(np.concatenate([v for v in mcs.triggers().values()] or [np.array([])]))
    a = np.round((times - t_start) * fs - pre_ms * 1e-3 * fs).astype(int)
    b = np.round((times - t_start) * fs + post_ms * 1e-3 * fs).astype(int)
    keep = (b > 0) & (a < n)
    a, b = np.clip(a[keep], 0, n), np.clip(b[keep], 0, n)
    # merge overlapping windows
    out = []
    for lo, hi in zip(a, b):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def detect_channel(x_uv, fs, windows, sos, k, refractory_ms, max_uv, guard_ms, pre, post):
    x = x_uv - np.median(x_uv)
    for lo, hi in windows:                       # bridge the artefact so it does not excite the filter
        l, r = x[lo - 1] if lo > 0 else 0.0, x[hi] if hi < x.size else 0.0
        x[lo:hi] = np.linspace(l, r, hi - lo)
    y = sosfiltfilt(sos, x)
    sigma = float(np.median(np.abs(y)) / 0.6745)
    pk, _ = find_peaks(-y, height=k * sigma, distance=max(int(round(refractory_ms * 1e-3 * fs)), 1))
    guard = int(round(guard_ms * 1e-3 * fs))
    ok = (pk >= pre) & (pk < y.size - post) & (-y[pk] <= max_uv)
    for lo, hi in windows:
        ok &= ~((pk >= lo) & (pk < hi + guard))
    pk = pk[ok]
    # Artefacts ring: a real spike's negative peak dominates its window, an artefact's rebound does not.
    if pk.size:
        rebound = np.array([y[p - pre:p + post].max() for p in pk])
        pk = pk[rebound < -y[pk]]
    wf = np.stack([y[p - pre:p + post] for p in pk], axis=1) if pk.size else np.empty((pre + post, 0))
    return pk, y[pk].astype(np.float32), wf.astype(np.float32), sigma


def detect_file(path, k=5.0, band=(300.0, 3000.0), refractory_ms=1.0, max_uv=1000.0,
                blank_pre_ms=1.0, blank_post_ms=6.0, guard_ms=1.0, cutout_ms=(1.0, 2.0), stream=0):
    mcs = McsFile(path)
    st = mcs.analog_stream(stream)
    fs = st.fs
    sos = butter(3, [band[0], min(band[1], fs / 2 - 1)], btype="band", fs=fs, output="sos")
    windows = stim_windows(mcs, st.t_start, fs, st.n_samples, blank_pre_ms, blank_post_ms)
    pre, post = int(round(cutout_ms[0] * 1e-3 * fs)), int(round(cutout_ms[1] * 1e-3 * fs))
    res = {}
    for ci, label in enumerate(st.labels):
        e = electrode_number(label)
        x = st.read_uv(0.0, None, channels=[ci])[0].astype(np.float64)
        pk, amp, wf, sigma = detect_channel(x, fs, windows, sos, k, refractory_ms, max_uv, guard_ms, pre, post)
        res[e] = dict(ts=pk / fs, amp=amp, wf=wf, sigma=sigma)
        print(f"  electrode {e:>3}: {pk.size:6d} spikes ({pk.size / st.duration:5.2f}/s)  sigma {sigma:5.1f} uV", flush=True)
    params = dict(k=k, band=list(band), refractory_ms=refractory_ms, max_uv=max_uv, blank_pre_ms=blank_pre_ms,
                  blank_post_ms=blank_post_ms, guard_ms=guard_ms, cutout_ms=list(cutout_ms), stream=stream,
                  fs=fs, t_start=st.t_start, duration=st.duration, n_stim_windows=len(windows))
    mcs.close()
    return res, params


def write_spikes(h5_path, res, params):
    out = spikes_path(h5_path)
    with h5py.File(out, "w") as f:
        f.attrs["source"] = os.path.basename(h5_path)
        f.attrs["params"] = json.dumps(params)
        g = f.create_group("SpikeStream")
        for e, r in res.items():
            eg = g.create_group(f"Electrode_{e}")
            eg.create_dataset("ts", data=r["ts"])
            eg.create_dataset("amp", data=r["amp"])
            eg.create_dataset("waveforms", data=r["wf"], compression="gzip")
            eg.attrs["sigma_uv"] = r["sigma"]
    return out


def load_spikes(h5_path):
    """(spikes {electrode: ts in s}, cutouts {electrode: (samples, n) uV}, cutout_ms) or None if no sidecar."""
    p = spikes_path(h5_path)
    if not os.path.exists(p):
        return None
    with h5py.File(p, "r") as f:
        params = json.loads(f.attrs["params"])
        ts, wf = {}, {}
        for name, eg in f["SpikeStream"].items():
            e = name.split("_", 1)[1]
            ts[e] = eg["ts"][:]
            wf[e] = eg["waveforms"][:]
    return ts, wf, (-params["cutout_ms"][0], params["cutout_ms"][1]), params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("-k", type=float, default=5.0, help="threshold in noise sigmas")
    ap.add_argument("--blank-post", type=float, default=6.0, help="ms blanked after each stim event")
    ap.add_argument("--max-uv", type=float, default=1000.0, help="reject peaks larger than this (artefact)")
    a = ap.parse_args()
    paths = a.paths or [p for p in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.h5")))
                        if not p.endswith(SUFFIX)]
    for p in paths:
        print(os.path.basename(p))
        res, params = detect_file(p, k=a.k, blank_post_ms=a.blank_post, max_uv=a.max_uv)
        print("  wrote", write_spikes(p, res, params), f"({sum(r['ts'].size for r in res.values())} spikes)")


if __name__ == "__main__":
    main()
