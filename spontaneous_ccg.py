#!/usr/bin/env python3
"""
spontaneous_ccg.py -- detect putative monosynaptic connections from SPONTANEOUS
activity in an MCS file, using jitter-free cross-correlogram analysis.

This is the right tool for a "beforestim" or "afterstim" baseline block, where
there are no stimuli to key an evoked analysis to. For files that do contain a
stimulation protocol, use stim_connectivity.py instead: intervention beats
correlation whenever you have it.

Method (Stark & Abeles 2009; English et al. 2017, Neuron 96:505):
  1. bin all spike trains at 0.5 ms
  2. compute every ordered pair's cross-correlogram out to +/- 30 ms
  3. estimate the slow, non-synaptic component by convolving each CCG with a
     PARTIALLY HOLLOW Gaussian, which ignores the centre bins so that a real
     synaptic peak cannot inflate its own baseline
  4. test the 1 to 4 ms window against a Poisson with that baseline rate,
     for excess (excitation) and deficit (inhibition)
  5. Bonferroni-correct within the tested window, then FDR across pairs

Usage:
    python spontaneous_ccg.py rec.h5                    # use detected spikes
    python spontaneous_ccg.py rec.h5 --segment-stream 1
    python spontaneous_ccg.py rec.h5 --truth mea60_truth.npz
"""

import argparse

import numpy as np
from scipy import sparse
from scipy.stats import poisson

from mcs import McsFile, mea_geometry


def hollow_gaussian(n_bins, sd_bins, hollow=0.6):
    """Stark & Abeles' partially hollow kernel: a Gaussian whose centre bin is
    scaled down, so the estimated baseline does not absorb the synaptic peak."""
    x = np.arange(-n_bins, n_bins + 1)
    k = np.exp(-0.5 * (x / sd_bins) ** 2)
    k[n_bins] *= 1.0 - hollow
    return k / k.sum()


def all_ccgs(trains, dur, bin_s, max_lag_s):
    """Cross-correlograms for every ordered pair.

    Returns (n, n, n_lags). Element [i, j, L] counts spikes of j at lag L
    after spikes of i, so a peak at positive lag means i drives j.
    """
    n = len(trains)
    nb = int(np.ceil(dur / bin_s))
    L = int(round(max_lag_s / bin_s))

    rows, cols = [], []
    for i, t in enumerate(trains):
        b = np.unique((np.asarray(t) / bin_s).astype(np.int64))
        b = b[(b >= 0) & (b < nb)]
        rows.append(np.full(b.size, i))
        cols.append(b)
    S = sparse.csr_matrix(
        (np.ones(sum(c.size for c in cols), np.float32),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, nb))

    out = np.zeros((n, n, 2 * L + 1), np.float32)
    for lag in range(-L, L + 1):
        if lag >= 0:
            A, B = S[:, :nb - lag], S[:, lag:]
        else:
            A, B = S[:, -lag:], S[:, :nb + lag]
        out[:, :, lag + L] = (A @ B.T).toarray()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--segment-stream", type=int, default=0)
    ap.add_argument("--bin", type=float, default=0.5, help="CCG bin, ms")
    ap.add_argument("--max-lag", type=float, default=30.0, help="CCG half-width, ms")
    ap.add_argument("--syn-lo", type=float, default=1.0, help="synaptic window start, ms")
    ap.add_argument("--syn-hi", type=float, default=4.0, help="synaptic window end, ms")
    ap.add_argument("--kernel-sd", type=float, default=10.0, help="baseline kernel sd, ms")
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--min-spikes", type=int, default=50)
    ap.add_argument("--pitch", type=float, default=200.0)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--out", default="ccg_connectivity.npz")
    a = ap.parse_args()

    f = McsFile(a.path)
    try:
        spikes = f.segment_events(stream=a.segment_stream)
    except ValueError as e:
        raise SystemExit(f"{e}\nRun stim_connectivity-style detection on the raw stream instead.")
    if not spikes:
        raise SystemExit("no detected spikes in this file's SegmentStreams")

    st = f.analog_stream(0)
    dur = st.duration
    labels = sorted(spikes, key=lambda s: (len(s), s))
    counts = np.array([spikes[k].size for k in labels])
    keep = [k for k, c in zip(labels, counts) if c >= a.min_spikes]
    print(f"{len(labels)} channels with detected spikes, {len(keep)} pass "
          f"the >= {a.min_spikes} spike cut")
    print(f"recording {dur:.1f} s, rates {counts.min()/dur:.2f} to {counts.max()/dur:.2f} Hz")
    if len(keep) < 2:
        raise SystemExit("not enough active channels")

    trains = [np.sort(spikes[k]) for k in keep]
    n = len(keep)
    bin_s, lag_s = a.bin * 1e-3, a.max_lag * 1e-3
    L = int(round(lag_s / bin_s))
    lags = (np.arange(-L, L + 1)) * a.bin

    C = all_ccgs(trains, dur, bin_s, lag_s)
    print(f"computed {n * (n - 1)} cross-correlograms, {C.shape[-1]} bins each")

    # Slow baseline via the partially hollow kernel, per pair.
    sd_bins = a.kernel_sd / a.bin
    k = hollow_gaussian(int(4 * sd_bins), sd_bins)
    pad = k.size // 2
    Cp = np.pad(C, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    base = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), -1, Cp)
    base = np.maximum(base[:, :, : C.shape[-1]], 1e-6)

    win = (lags >= a.syn_lo) & (lags <= a.syn_hi)
    nw = int(win.sum())

    # Poisson test per bin, Bonferroni within the window, then take the best bin.
    obs = C[:, :, win]
    lam = base[:, :, win]
    p_exc = poisson.sf(obs - 1, lam)          # P(X >= obs)
    p_inh = poisson.cdf(obs, lam)             # P(X <= obs)

    best_exc = p_exc.min(-1) * nw
    best_inh = p_inh.min(-1) * nw
    lat = np.where(win)[0][p_exc.argmin(-1)]
    latency = lags[lat]

    off = ~np.eye(n, dtype=bool)
    exc = (best_exc < a.alpha) & off
    inh = (best_inh < a.alpha) & off

    # A real synapse is asymmetric. If both directions look significant at the
    # same lag the pair is more likely sharing common input, so drop it.
    sym = exc & exc.T & (np.abs(latency - latency.T) < 1.0)
    exc &= ~sym

    strength = np.where(exc, (obs.max(-1) - lam.max(-1)) / np.maximum(lam.max(-1), 1e-6), 0.0)

    print(f"\nputative excitatory connections: {exc.sum()} "
          f"({exc.sum() / (n * n - n) * 100:.1f}% of ordered pairs)")
    print(f"putative inhibitory connections: {inh.sum()}")
    print(f"dropped as symmetric (likely common input): {sym.sum()}")
    if exc.any():
        print(f"latency of excitatory edges: median {np.median(latency[exc]):.1f} ms, "
              f"range {latency[exc].min():.1f} to {latency[exc].max():.1f} ms")

    pos = mea_geometry(keep, a.pitch)
    np.savez(a.out, exc=exc, inh=inh, latency=latency, strength=strength,
             ccg=C, lags=lags, labels=np.array(keep), pos=pos)
    print(f"wrote {a.out}")

    if exc.any():
        print("\nstrongest putative connections:")
        ii, jj = np.where(exc)
        for r in np.argsort(-strength[ii, jj])[:10]:
            i, j = ii[r], jj[r]
            print(f"   {keep[i]:>4s} -> {keep[j]:<4s}  latency {latency[i, j]:.1f} ms  "
                  f"excess {strength[i, j]:.2f}x baseline")

    if a.truth:
        T = np.load(a.truth, allow_pickle=True)
        tl = [str(x) for x in T["labels"]]
        true_pairs = {(tl[a_], tl[b_]) for a_, b_ in T["edges"]}
        got = {(keep[i], keep[j]) for i, j in zip(*np.where(exc))}
        tp = len(true_pairs & got)
        print(f"\nvs ground truth: recovered {tp} of {len(true_pairs)} planted edges, "
              f"{len(got) - tp} extra")
        for p in sorted(true_pairs):
            print(f"   {p[0]} -> {p[1]}: {'FOUND' if p in got else 'missed'}")

    f.close()


if __name__ == "__main__":
    main()
