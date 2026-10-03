#!/usr/bin/env python3
"""
stim_audit.py -- before you analyse a stimulation file, find out whether it can
answer the question you are asking.

Reports, from the file itself rather than from its name:
  * where the stimulus times actually are, and how regular they are
  * which site was stimulated, inferred from artefact amplitude
  * the per-channel artefact recovery time, which is the FLOOR on any latency
    you can ever measure. If recovery exceeds ~4 ms you cannot see monosynaptic
    responses on that channel, whatever the analysis.
  * how repeatable the artefact is across trials, which decides whether
    trial-median template subtraction will work
  * whether the evoked response drifts across the block, which tells you
    if the protocol is inducing plasticity while you measure

Usage:
    python stim_audit.py stimfile.h5
    python stim_audit.py stimfile.h5 --stream 0 --expect-site 47
"""

import argparse

import numpy as np

from mcs import McsFile, electrode_number


def find_stim_times(f, st, thresh_sd=25.0, min_gap_s=0.05, chunk_s=20.0):
    """Stimulus onsets from the EventStream if it has regular events, else from
    the raw trace, as the times when many channels transient together."""
    # MCS writes both a 'Start' and a 'Stop' entity for each pulse. Latency is
    # only meaningful from pulse ONSET, so prefer Start and say so if we had to
    # settle for Stop.
    cands = []
    for label, t in f.triggers().items():
        if t.size >= 3:
            isi = np.diff(np.sort(t))
            if isi.std() / max(isi.mean(), 1e-12) < 0.25:
                cands.append((label, np.sort(t)))
    if cands:
        starts = [c for c in cands if "start" in c[0].lower()]
        pick = starts[0] if starts else cands[0]
        note = "" if starts else "  [no 'Start' entity found; this marks pulse OFFSET]"
        return pick[1], f"EventStream entity {pick[0]!r}{note}"
    for key, (label, t) in f.segment_triggers().items():
        if t.size >= 3:
            isi = np.diff(np.sort(t))
            if isi.std() / max(isi.mean(), 1e-12) < 0.25:
                return np.sort(t), f"SegmentStream {key} {label!r}"

    fs = st.fs
    onsets = []
    ref = st.read_uv(0.0, min(2.0, st.duration))
    sd = np.median(np.abs(ref), axis=1).mean() / 0.6745
    for t0 in np.arange(0.0, st.duration, chunk_s):
        x = st.read_uv(t0, min(t0 + chunk_s, st.duration))
        loud = (np.abs(x) > thresh_sd * sd).sum(0)
        idx = np.flatnonzero((loud > x.shape[0] * 0.5))
        if idx.size:
            split = np.split(idx, np.flatnonzero(np.diff(idx) > min_gap_s * fs) + 1)
            onsets += [t0 + g[0] / fs for g in split]
    return np.array(sorted(onsets)), "detected from the raw trace"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--stream", type=int, default=0)
    ap.add_argument("--pre", type=float, default=0.050)
    ap.add_argument("--post", type=float, default=0.050)
    ap.add_argument("--recover-k", type=float, default=5.0,
                    help="recovered when |signal| stays under k * baseline sd")
    ap.add_argument("--expect-site", default=None)
    a = ap.parse_args()

    f = McsFile(a.path)
    st = f.analog_stream(a.stream)
    fs, N = st.fs, st.n_channels
    labels = [electrode_number(c.label) for c in st.channels]
    print(f"{a.path}")
    print(f"  stream {a.stream!r}: {N} ch @ {fs:g} Hz, {st.duration:.1f} s")

    ents = {k: v for k, v in f.triggers().items()}
    if ents:
        print(f"\n  EventStream entities ({len(ents)}):")
        for k, v in list(ents.items())[:12]:
            cv = (np.diff(np.sort(v)).std() / max(np.diff(np.sort(v)).mean(), 1e-12)
                  if v.size > 2 else float("nan"))
            print(f"    {k!r}: {v.size} events, interval CV {cv:.3f}")
        if len(ents) > 12:
            print(f"    ... and {len(ents) - 12} more")

    times, how = find_stim_times(f, st)
    if times.size < 2:
        raise SystemExit("  no stimulus train found. Is this a spontaneous file?")
    isi = np.diff(times)
    print(f"\n  stimulus times: {times.size} pulses, {how}")
    print(f"  interval {np.median(isi):.3f} s (CV {isi.std()/isi.mean():.3f}), "
          f"block spans {times[0]:.1f} to {times[-1]:.1f} s")
    if times.size < 20:
        print(f"  WARNING: {times.size} trials is too few for connectivity inference. "
              "Treat this block as a parameter check, not as data.")
    elif times.size < 50:
        print(f"  CAUTION: {times.size} trials is marginal. Expect wide error bars "
              "and a noisy artefact template.")

    npre, npost = int(a.pre * fs), int(a.post * fs)
    ep = []
    for t in times:
        i0 = int(round(t * fs)) - npre
        if i0 >= 0 and i0 + npre + npost <= st.n_samples:
            ep.append(st.read_uv(i0 / fs, (i0 + npre + npost) / fs))
    n = min(e.shape[1] for e in ep)
    ep = np.stack([e[:, :n] for e in ep])
    print(f"  usable epochs: {ep.shape[0]}")

    base = ep[:, :, : npre - 5]
    sd = np.median(np.abs(base), axis=(0, 2)) / 0.6745
    sd = np.maximum(sd, 1e-9)

    peak = np.abs(ep[:, :, npre: npre + int(0.003 * fs)]).max(-1).mean(0)
    site = int(np.argmax(peak))
    print(f"\n  largest artefact on channel {labels[site]!r} "
          f"({peak[site]:.0f} uV peak), most likely the stimulating site")
    ranked = np.argsort(-peak)[:5]
    print("  artefact amplitude ranking: " +
          ", ".join(f"{labels[i]}={peak[i]:.0f}uV" for i in ranked))
    if a.expect_site:
        want = electrode_number(a.expect_site)
        if labels[site] != want:
            print(f"  WARNING: filename says {want}, data says {labels[site]}. "
                  "Trust the data, or check whether the stimulating channel is blanked.")

    # Recovery, measured from the ARTEFACT PEAK rather than from the timestamp.
    # The trigger may mark pulse onset or pulse offset (MCS writes both, named
    # 'Start' and 'Stop'), and an earlier version of this script measured from
    # the timestamp and reported 0 ms for every channel whenever the artefact
    # sat before it. Finding the peak makes the measurement alignment-proof.
    m = np.abs(ep.mean(0))
    search = int(0.010 * fs)
    lo = max(0, npre - search)
    hold = max(2, int(0.002 * fs))          # must stay quiet this long to count

    rec = np.full(N, np.nan)
    peak_off = np.full(N, np.nan)
    for ch in range(N):
        seg = m[ch, lo: npre + search]
        pk = lo + int(np.argmax(seg))
        peak_off[ch] = (pk - npre) / fs * 1e3
        quiet = m[ch, pk:] <= a.recover_k * sd[ch]
        if quiet.size < hold:
            continue
        # First index from which the signal is quiet for `hold` samples running.
        run = np.convolve(quiet.astype(int), np.ones(hold, int), "valid")
        ok = np.flatnonzero(run == hold)
        rec[ch] = (ok[0] / fs * 1e3) if ok.size else (quiet.size / fs * 1e3)

    print(f"\n  artefact peak sits {np.median(peak_off):+.2f} ms from the trigger "
          f"timestamp (so the trigger marks "
          f"{'pulse offset' if np.median(peak_off) < -0.1 else 'pulse onset'})")
    print(f"  artefact recovery time (ms after the artefact peak):")
    print(f"    median {np.median(rec):.2f}, 90th pct {np.percentile(rec, 90):.2f}, "
          f"worst {np.nanmax(rec):.2f} on channel {labels[int(np.nanargmax(rec))]}")
    bad = int((rec > 4.0).sum())
    print(f"    channels still contaminated at 4 ms: {bad} of {N}")
    if bad > N * 0.25:
        print("    WARNING: on these channels the monosynaptic window is buried in "
              "artefact. Shorten the pulse, drop the amplitude, or accept that you "
              "are measuring polysynaptic responses only.")
    print(f"    => minimum resolvable latency is about {np.percentile(rec, 75):.1f} ms")

    # Artefact repeatability: correlation of each trial's artefact to the median.
    w = slice(npre, npre + int(0.005 * fs))
    tmpl = np.median(ep[:, :, w], axis=0)
    cors = []
    for t in range(ep.shape[0]):
        x, y = ep[t, :, w].ravel(), tmpl.ravel()
        cors.append(np.corrcoef(x, y)[0, 1])
    cors = np.array(cors)
    print(f"\n  artefact repeatability across trials: r = {np.median(cors):.4f} "
          f"(min {cors.min():.4f})")
    if np.median(cors) > 0.99:
        print("    highly stereotyped, so trial-median template subtraction will work well")
    elif np.median(cors) > 0.9:
        print("    mostly stereotyped; template subtraction will help but leave residue")
    else:
        print("    WARNING: the artefact varies between trials, so template subtraction "
              "will not clean it. Check for electrode polarisation or drifting contact.")

    # Drift, measured on DETECTED EVENTS rather than on raw amplitude. A slow
    # artefact tail inside the response window would make an amplitude measure
    # drift even when the neural response does not.
    clean = ep - np.median(ep, axis=0, keepdims=True)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(3, [300.0, min(3000.0, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, clean, axis=-1)
    bsd = np.median(np.abs(y[:, :, : npre - 5]), axis=(0, 2)) / 0.6745
    start = npre + int(max(0.005, np.nanpercentile(rec, 75) * 1e-3) * fs)
    w = y[:, :, start: npre + int(0.030 * fs)]
    below = w < -(4.5 * np.maximum(bsd, 1e-9))[None, :, None]
    amp = (below[:, :, 1:] & ~below[:, :, :-1]).sum(-1).sum(-1).astype(float)
    k = max(1, len(amp) // 3)
    first, last = amp[:k].mean(), amp[-k:].mean()
    change = (last - first) / max(first, 1e-9) * 100
    print(f"\n  evoked events per trial, first third {first:.1f} vs last third "
          f"{last:.1f} ({change:+.1f}%)")
    if abs(change) > 25:
        print("    WARNING: the response is drifting across the block. This protocol is "
              "changing the network while measuring it, so do not pool all trials into "
              "one connectivity estimate. Split the block and compare.")

    f.close()


if __name__ == "__main__":
    main()