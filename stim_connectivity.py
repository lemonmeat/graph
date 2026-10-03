#!/usr/bin/env python3
"""
stim_connectivity.py -- build a directed connectivity graph from an MCS
stimulation experiment where every site was stimulated in turn.

Pipeline, per stimulating site i:
  1. cut epochs around every trigger for that site
  2. subtract the across-trial median epoch, which is the stimulus artefact
     (deterministic every trial) but not the jittered evoked spikes
  3. blank whatever saturation remains, which sets the latency floor
  4. bandpass 300-3000 Hz and detect events at -k sigma, sigma from pre-stim
  5. PSTH per recording site j, permutation-tested against the pre-stim baseline
  6. FDR across all N*(N-1) ordered pairs
  7. strip indirect paths with a network-deconvolution step on the effect matrix
  8. keep edges that are also short-latency and low-jitter

Usage:
    python stim_connectivity.py rec.h5
    python plot_connectivity.py connectivity.npz --grid 4 4 4 --pitch 200
    python stim_connectivity.py rec.h5 --truth mcs_stim_truth.npz     # validation
"""

import argparse

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from mcs import McsFile


def fdr(p, q=0.05):
    """Benjamini-Hochberg. Returns a boolean mask of rejected nulls."""
    p = np.asarray(p)
    flat = p.ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    m = ranked.size
    thresh = q * np.arange(1, m + 1) / m
    below = ranked <= thresh
    k = np.flatnonzero(below).max() + 1 if below.any() else 0
    keep = np.zeros(m, bool)
    keep[order[:k]] = True
    return keep.reshape(p.shape)


def epochs_for(stream, times, pre, post, fs):
    """(n_trials, n_channels, n_samples) in microvolts."""
    npre, npost = int(pre * fs), int(post * fs)
    out = []
    for t in times:
        i0 = int(round(t * fs)) - npre
        if i0 < 0 or i0 + npre + npost > stream.n_samples:
            continue
        out.append(stream.read_uv(i0 / fs, (i0 + npre + npost) / fs))
    if not out:
        return np.empty((0, stream.n_channels, npre + npost), np.float32)
    n = min(e.shape[1] for e in out)
    return np.stack([e[:, :n] for e in out])


def clean(ep, fs, npre, blank_ms, med_ms):
    """Remove the stimulus artefact, then bandpass.

    The artefact is deterministic: for a given (stimulating site, recording
    channel) pair it is near-identical on every trial. Evoked spikes are
    jittered and probabilistic. So the across-trial MEDIAN of the epoch is
    essentially the artefact alone, and subtracting it leaves the spikes.

    The median rather than the mean matters. A response present on a minority
    of trials barely moves the median, so little of the real signal is
    subtracted away. With a response on most trials you do lose some amplitude,
    which costs sensitivity but not specificity. Raise the trial count if that
    bites.

    Anything inside the blanking window is unrecoverable, so `blank_ms` is the
    floor on the latencies this analysis can ever resolve. Report it.
    """
    x = ep.astype(np.float32).copy()
    x -= np.median(x, axis=0, keepdims=True)

    nb = int(blank_ms * 1e-3 * fs)
    if nb > 0:
        x[:, :, npre:npre + nb] = 0.0

    if med_ms > 0:                      # optional residual slow-drift removal
        k = max(3, int(med_ms * 1e-3 * fs) | 1)
        x -= median_filter(x, size=(1, 1, k), mode="nearest")

    hi = min(3000.0, 0.45 * fs)
    sos = butter(3, [300.0, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--stream", type=int, default=0)
    ap.add_argument("--pre", type=float, default=0.030, help="baseline window, s")
    ap.add_argument("--post", type=float, default=0.030, help="response window, s")
    ap.add_argument("--blank", type=float, default=0.6, help="saturation blanking, ms")
    ap.add_argument("--median-ms", type=float, default=0.0, help="artefact-tail median kernel")
    ap.add_argument("--k", type=float, default=4.5, help="detection threshold in sigma")
    ap.add_argument("--resp", type=float, default=12.0, help="response window end, ms")
    ap.add_argument("--direct-max", type=float, default=5.0, help="max latency for a direct edge, ms")
    ap.add_argument("--max-jitter", type=float, default=1.5, help="max onset jitter for a direct edge, ms")
    ap.add_argument("--min-weight", type=float, default=0.03, help="deconvolved weight cutoff")
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--q", type=float, default=0.01, help="FDR level")
    ap.add_argument("--truth", default=None, help="npz with A/pos, for validation")
    ap.add_argument("--out", default="connectivity.npz")
    a = ap.parse_args()

    f = McsFile(a.path)
    st = f.analog_stream(a.stream)
    fs, N = st.fs, st.n_channels
    trig = f.triggers()
    print(f"{N} channels @ {fs:g} Hz, {st.duration:.1f} s, {len(trig)} trigger entities")
    if a.blank * 1e-3 * fs < 1:
        print("warning: blanking window is under one sample")

    # Map each trigger entity to the channel index it stimulated.
    stim_map = {}
    for label, times in trig.items():
        src = label.split("[src", 1)[1].strip(" ]") if "[src" in label else None
        if src is None:
            continue
        try:
            stim_map[st.index_of(src)] = times
        except KeyError:
            pass
    if not stim_map:
        raise SystemExit("could not match trigger entities to channels; inspect f.triggers()")
    print(f"matched {len(stim_map)} stimulating sites, "
          f"{np.median([len(v) for v in stim_map.values()]):.0f} trials each (median)")

    npre = int(a.pre * fs)
    r0, r1 = int(a.blank * 1e-3 * fs), int(a.resp * 1e-3 * fs)

    effect = np.zeros((N, N))       # evoked events per trial, above baseline
    latency = np.full((N, N), np.nan)
    jitter = np.full((N, N), np.nan)
    pval = np.ones((N, N))
    rng = np.random.default_rng(0)

    for src, times in sorted(stim_map.items()):
        ep = epochs_for(st, times, a.pre, a.post, fs)
        if ep.shape[0] < 5:
            continue
        y = clean(ep, fs, npre, a.blank, a.median_ms)

        sigma = np.median(np.abs(y[:, :, : npre - 10]), axis=(0, 2)) / 0.6745
        sigma = np.maximum(sigma, 1e-6)
        below = y < -(a.k * sigma)[None, :, None]
        onset = below[:, :, 1:] & ~below[:, :, :-1]      # (trials, ch, samples)

        post = onset[:, :, npre + r0: npre + r1]
        pre = onset[:, :, : npre - 1]
        n_post = post.sum(-1).astype(float)              # (trials, ch)
        rate_pre = pre.sum(-1).mean(0) / (pre.shape[-1] / fs)
        expected = rate_pre * ((r1 - r0) / fs)

        obs = n_post.mean(0) - expected
        effect[src] = obs

        # Permutation null: shuffle which trial each count came from by
        # resampling matched-length windows from the pre-stimulus period.
        nwin = (npre - 1) // max(1, r1 - r0)
        if nwin >= 2:
            chunks = pre[:, :, : nwin * (r1 - r0)].reshape(pre.shape[0], N, nwin, r1 - r0)
            null_counts = chunks.sum(-1).astype(float)   # (trials, ch, nwin)
            draws = rng.integers(0, nwin, size=(a.nperm, pre.shape[0]))
            # mean over trials for each permutation
            idx_t = np.arange(pre.shape[0])
            null = null_counts[idx_t[None, :], :, draws].mean(1)   # (nperm, ch)
            pval[src] = (1 + (null >= n_post.mean(0)[None, :]).sum(0)) / (a.nperm + 1)

        # Latency from the trial-pooled PSTH, as the first bin whose count
        # exceeds baseline by 3 standard deviations of the baseline counts.
        psth = onset.sum(0).astype(float)                # (ch, samples)
        base_mu = psth[:, : npre - 1].mean(1)
        base_sd = psth[:, : npre - 1].std(1) + 1e-9
        for ch in range(N):
            seg = psth[ch, npre + r0: npre + r1]
            hit = np.flatnonzero(seg > base_mu[ch] + 3 * base_sd[ch])
            if hit.size:
                latency[src, ch] = (r0 + hit[0]) / fs * 1e3
                tr = np.flatnonzero(post[:, ch, :].any(-1))
                if tr.size > 1:
                    firsts = [np.argmax(post[t, ch, :]) for t in tr]
                    jitter[src, ch] = np.std(firsts) / fs * 1e3

    np.fill_diagonal(pval, 1.0)
    np.fill_diagonal(effect, 0.0)

    sig = fdr(pval, a.q) & (effect > 0)
    print(f"\nsignificant evoked responses after FDR q={a.q}: {sig.sum()} "
          f"of {N * (N - 1)} ordered pairs")

    # Deconvolve BEFORE filtering on latency. An indirect effect is roughly the
    # product of the direct effects along its path, and that is exactly the
    # series this inversion sums away. Filtering first throws away the long
    # paths the inversion needs in order to explain the short ones.
    off = ~np.eye(N, dtype=bool)
    R = np.where(sig & off, np.maximum(effect, 0.0), 0.0)
    if R.max() > 0:
        ev = np.max(np.abs(np.linalg.eigvals(R)))
        Rs = R / (ev * 1.05) if ev > 0 else R
        A_dir = np.eye(N) - np.linalg.inv(np.eye(N) + Rs)
        np.fill_diagonal(A_dir, 0.0)
        A_dir[A_dir < 0] = 0.0
    else:
        A_dir = R

    print("\nedges surviving each weight threshold "
          "(look for where the count stops falling steeply):")
    for t in (0.01, 0.02, 0.03, 0.05, 0.08):
        m = (A_dir > t) & off & (latency <= a.direct_max) & (jitter <= a.max_jitter)
        print(f"   weight > {t:<5g} {int(np.nansum(m)):5d} edges")

    direct = sig & (latency <= a.direct_max) & ~np.isnan(latency)
    deconv = (A_dir > a.min_weight) & off & (latency <= a.direct_max) & (jitter <= a.max_jitter)
    deconv = np.where(np.isnan(latency) | np.isnan(jitter), False, deconv)
    print(f"\nsignificant {sig.sum()}  ->  latency-filtered {direct.sum()}  ->  "
          f"final graph {deconv.sum()} edges "
          f"(density {deconv.sum() / (N * N - N):.3f})")

    np.savez(a.out, effect=effect, latency=latency, jitter=jitter, pval=pval,
             sig=sig, direct=direct, deconv=deconv, weights=A_dir,
             labels=np.array(st.labels))
    print(f"wrote {a.out}")

    if a.truth:
        T = np.load(a.truth)["A"].astype(bool)
        for name, est in [("significant", sig), ("+latency", direct), ("+deconv", deconv)]:
            tp = (est & T).sum(); fp = (est & ~T).sum(); fn = (~est & T).sum()
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f"  {name:12s} precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}"
                  f"   (TP {tp}, FP {fp}, FN {fn})")
        lat_true = np.load(a.truth)["DELAY"]
        m = deconv & T
        if m.any():
            print(f"  latency error on true positives: "
                  f"median {np.median(latency[m] - lat_true[m] * 1e3):+.2f} ms")

    f.close()


if __name__ == "__main__":
    main()