"""Can this stimulation file answer the question being asked? (port of the legacy ``stim_audit.py``)

Reports, from the data rather than the file name: the pulse and train structure, the
stimulated site (or why it cannot be inferred), the artifact recovery time per channel, which
sets the shortest measurable response latency, and how repeatable the artifact is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from spikeinterface.core import BaseRecording

from meagraph.io.mcs_events import StimEvents
from meagraph.stimulation.artifacts import Recovery, SiteInference, infer_site, measure_recovery


@dataclass(frozen=True, eq=False)
class StimAudit:
    source: str
    n_pulses: int
    pulses_per_train: tuple[int, ...]
    train_interval_s: float | None
    pulse_duration_ms: tuple[float, ...]
    first_s: float
    last_s: float
    site_given: str | None
    site: SiteInference
    recovery: Recovery
    repeatability_r: float | None
    warnings: tuple[str, ...]
    # Per-pulse sites (stimulation log): (logged site, its artifact rank among channels, 1 = largest,
    # 0 = not checkable because several channels saturate; the channel with the largest artifact or
    # "saturated"; number of that site's pulses)
    site_check: tuple[tuple[str, int, str, int], ...] = ()


def check_logged_sites(recording: BaseRecording, stim: StimEvents, n_trials: int = 10) -> tuple[tuple[str, int, str, int], ...]:
    """For each logged site: does its own pulses' artifact peak on that electrode?

    A mismatch points to a wrong electrode table or a wrong log-to-recording match. When every
    channel saturates, the peaks tie and the check cannot decide.
    """
    if stim.pulse_sites is None:
        return ()
    sites = np.array(stim.pulse_sites, dtype=object)
    out = []
    for site in sorted({s for s in stim.pulse_sites if s}):
        mine = stim.select(sites == site)
        inferred = infer_site(recording, mine, n_trials=n_trials)
        if len(inferred.saturated) > 1:  # several channels hit the ADC rail: artifact size cannot point to the site
            out.append((site, 0, "saturated", mine.n))
            continue
        order = list(inferred.peak_uv)
        rank = order.index(site) + 1 if site in order else 0
        out.append((site, rank, order[0] if order else "", mine.n))
    return tuple(out)


def group_trains(onsets_s: np.ndarray, max_gap_s: float = 0.05) -> list[np.ndarray]:
    """Split pulse onsets into trains: a gap longer than ``max_gap_s`` starts a new train."""
    if onsets_s.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(onsets_s) > max_gap_s) + 1
    return np.split(onsets_s, cuts)


def _repeatability(recording: BaseRecording, stim: StimEvents, n_trials: int = 40) -> float | None:
    """Median correlation of each pulse's raw artifact with the across-pulse median artifact."""
    fs = recording.get_sampling_frequency()
    offsets = stim.offsets_s if stim.offsets_s is not None else stim.onsets_s + 0.002
    keep = np.flatnonzero(stim.onsets_s - recording.get_start_time() > 0.001)[:n_trials]
    if keep.size < 3:
        return None
    length = int(round(np.median(offsets[keep] - stim.onsets_s[keep]) * fs)) + 5
    eps = []
    for i in keep:
        a = int(round((stim.onsets_s[i] - recording.get_start_time()) * fs))
        eps.append(recording.get_traces(start_frame=a, end_frame=a + length, return_in_uV=True).ravel())
    eps = np.stack(eps)
    template = np.median(eps, axis=0)
    r = [np.corrcoef(e, template)[0, 1] for e in eps]
    return float(np.median(r))


def audit_stimulation(recording: BaseRecording, stim: StimEvents, **recovery_kwargs) -> StimAudit:
    trains = group_trains(stim.onsets_s)
    starts = np.array([t[0] for t in trains])
    durations = stim.durations_s
    warns = []
    if len(trains) < 20:
        warns.append(f"only {len(trains)} trains: too few for connectivity inference, treat as a parameter check")
    elif len(trains) < 50:
        warns.append(f"{len(trains)} trains is marginal: expect wide error bars")
    if stim.onsets_s.size and stim.onsets_s[0] - recording.get_start_time() < 0.002:
        warns.append("the first pulse is at the start of the recording and has no baseline")

    site = infer_site(recording, stim)
    if stim.site is not None and site.site is not None and site.site != stim.site:
        warns.append(f"site given as {stim.site} but the artifact points to {site.site}")
    site_check = check_logged_sites(recording, stim)
    off = [c for c in site_check if c[1] > 1]
    if any(c[1] == 0 for c in site_check):
        warns.append("the array saturates on some sites' pulses, so the artifact cannot confirm those logged sites")
    if off:
        warns.append(f"{len(off)} of {len(site_check)} logged sites do not have the largest artifact on their own pulses "
                     "(check the electrode table and the log match)")  # fmt: skip
    recovery = measure_recovery(recording, stim, **recovery_kwargs)
    unrecovered = int(np.isnan(recovery.recovery_ms).sum())
    if unrecovered:
        warns.append(f"{unrecovered} channels still show artifact {recovery.max_post_ms:g} ms after the pulse")
    return StimAudit(
        source=stim.source,
        n_pulses=stim.n,
        pulses_per_train=tuple(sorted({len(t) for t in trains})),
        train_interval_s=float(np.median(np.diff(starts))) if starts.size > 1 else None,
        pulse_duration_ms=tuple(np.unique(np.round(durations * 1e3, 2)).tolist()) if durations is not None else (),
        first_s=float(stim.onsets_s[0]) if stim.n else float("nan"),
        last_s=float(stim.onsets_s[-1]) if stim.n else float("nan"),
        site_given=stim.site,
        site=site,
        recovery=recovery,
        repeatability_r=_repeatability(recording, stim),
        warnings=tuple(warns),
        site_check=site_check,
    )


def format_audit(audit: StimAudit) -> str:
    rec = audit.recovery
    r = rec.recovery_ms[~np.isnan(rec.recovery_ms)]
    worst = np.argsort(-np.nan_to_num(rec.recovery_ms, nan=np.inf))[:3]
    top = list(audit.site.peak_uv.items())[:3]
    lines = [
        f"{audit.source}: {audit.n_pulses} pulses, {audit.pulses_per_train} pulses per train, "
        f"train interval {audit.train_interval_s:.3f} s, pulse {audit.pulse_duration_ms} ms, "
        f"from {audit.first_s:.2f} to {audit.last_s:.2f} s"
        if audit.train_interval_s is not None
        else f"{audit.source}: {audit.n_pulses} pulses",
        _site_line(audit),
        "  largest artifacts: " + ", ".join(f"{c} {v / 1000:.1f} mV" for c, v in top),
        f"  recovery after pulse offset ({rec.n_epochs} pulses): median {np.median(r):.1f} ms, "
        f"90th percentile {np.percentile(r, 90):.1f} ms; slowest "
        + ", ".join(f"{rec.channel_ids[i]} {rec.recovery_ms[i]:.1f} ms" for i in worst)
        if r.size
        else f"  recovery: not measurable ({rec.n_epochs} usable pulses)",
    ]
    for site, rank, top, n in audit.site_check:
        if rank > 1:
            lines.append(f"    logged site {site} ({n} pulses): artifact rank {rank or 'n/a'}, largest on {top}")
    if audit.repeatability_r is not None:
        lines.append(f"  artifact repeatability: median r = {audit.repeatability_r:.4f}")
    lines += [f"  WARNING: {w}" for w in audit.warnings]
    return "\n".join(lines)


def _site_line(audit: StimAudit) -> str:
    if audit.site_check:
        agree = sum(1 for c in audit.site_check if c[1] == 1)
        unchecked = sum(1 for c in audit.site_check if c[1] == 0)
        text = (f"  sites: from the stimulation log, {len(audit.site_check)} sites; the logged electrode has the largest "
                f"artifact for {agree} of {len(audit.site_check) - unchecked} checkable")  # fmt: skip
        return text + (f" ({unchecked} not checkable: the array saturates)" if unchecked else "")
    return f"  site: given {audit.site_given or 'not given'}; inferred {audit.site.site or 'none'} ({audit.site.reason})"
