"""Stimulation logs: which electrode each pulse stimulated (DECISIONS.md D25).

MCS files record when each STG output fired, but not which electrode it was routed to. The
stimulation script writes ``<tag>.stimlog.csv`` (one row per pulse) and ``<tag>.notes.yaml`` next
to the recording; ``docs/DATA_FORMAT.md`` documents both.

Matching uses order, not clocks. Within each STG output, the k-th logged pulse is the k-th
recorded ``Single Pulse Start`` event. The PC clock in the log only checks the match: its
intervals between pulses must agree with the recorded ones (they are random, so they form a
fingerprint). If the counts differ, pulses are matched one by one by their intervals instead,
and the unmatched ones are reported.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from meagraph.io.mcs_events import StimEvents

SUFFIX = ".stimlog.csv"
_REQUIRED = ("seq", "stg", "label", "pc_elapsed_s")


@dataclass(frozen=True, eq=False)
class StimLog:
    """A parsed stimulation log, rows in ``seq`` order."""

    path: Path
    seq: np.ndarray
    stg: np.ndarray  # stimulator output, 1 or 2
    label: tuple[str, ...]  # MCS electrode label of each pulse
    time_s: np.ndarray  # PC time of the pulse: trial send time + offset within the trial
    rows: tuple[dict, ...]  # every column as written, for reference
    notes: dict | None = None  # <tag>.notes.yaml, if present

    @property
    def tag(self) -> str:
        return self.path.name[: -len(SUFFIX)]

    def for_output(self, source: str) -> np.ndarray:
        """Row indices of one output ("STG 1"), in order."""
        return np.flatnonzero(self.stg == int(source.split()[-1]))


@dataclass(frozen=True)
class Alignment:
    """How one output's logged pulses matched its recorded pulses."""

    source: str
    n_logged: int
    n_recorded: int
    n_matched: int
    method: str  # "order" (counts equal, intervals agree) or "intervals" (matched one by one)
    max_residual_ms: float  # largest disagreement between logged and recorded timing
    unmatched_logged: tuple[int, ...]  # seq numbers
    unmatched_recorded: tuple[int, ...]  # pulse indices in the recording

    @property
    def complete(self) -> bool:
        return self.n_matched == self.n_logged == self.n_recorded

    def summary(self) -> str:
        text = (f"{self.source}: {self.n_matched} of {self.n_recorded} recorded pulses matched to the log "
                f"({self.n_logged} logged; by {self.method}; timing within {self.max_residual_ms:.1f} ms)")  # fmt: skip
        if self.unmatched_logged:
            text += f"; logged but not recorded: seq {list(self.unmatched_logged)[:10]}"
        if self.unmatched_recorded:
            text += f"; recorded but not logged: {len(self.unmatched_recorded)} pulses (site unknown)"
        return text


def read_stimlog(path: str | Path) -> StimLog:
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path.name}: the stimulation log has no pulses")
    missing = [c for c in _REQUIRED if c not in rows[0]]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing} (see docs/DATA_FORMAT.md)")
    rows.sort(key=lambda r: int(r["seq"]))
    offset = [float(r.get("offset_ms") or 0.0) / 1e3 for r in rows]
    notes_path = path.with_name(path.name[: -len(SUFFIX)] + ".notes.yaml") if path.name.endswith(SUFFIX) else None
    notes = yaml.safe_load(notes_path.read_text()) if notes_path is not None and notes_path.exists() else None
    return StimLog(
        path=path,
        seq=np.array([int(r["seq"]) for r in rows]),
        stg=np.array([int(r["stg"]) for r in rows]),
        label=tuple(r["label"].strip() for r in rows),
        time_s=np.array([float(r["pc_elapsed_s"]) for r in rows]) + np.array(offset),
        rows=tuple(rows),
        notes=notes if isinstance(notes, dict) else None,
    )


def find_stimlog(recording_path: str | Path) -> Path | None:
    """The log for a recording, from the files next to it.

    1. ``<recording name>.stimlog.csv``;
    2. otherwise the log whose tag (its name without ``.stimlog.csv``) appears in the recording's
       name, as Multi Channel Experimenter adds a start time before and the device ID after the
       name typed in. If several tags appear, the longest wins; a tie is an error.
    """
    rec = Path(recording_path)
    exact = rec.with_name(rec.stem + SUFFIX)
    if exact.exists():
        return exact
    hits = [p for p in rec.parent.glob("*" + SUFFIX) if p.name[: -len(SUFFIX)] and p.name[: -len(SUFFIX)] in rec.stem]
    if not hits:
        return None
    longest = max(len(p.name) for p in hits)
    best = [p for p in hits if len(p.name) == longest]
    if len(best) > 1:
        raise ValueError(f"several stimulation logs match {rec.name}: {[p.name for p in best]}; pass one with --stimlog")
    return best[0]


def align_stimlog(stim: Sequence[StimEvents], log: StimLog, tolerance_s: float = 0.1) -> tuple[list[StimEvents], list[Alignment]]:
    """Attach the logged site to every recorded pulse it matches (unmatched pulses get "").

    ``tolerance_s`` is how far a recorded pulse may sit from where the log's intervals predict it.
    Logged intervals are random 2-4 s in the stimulation script, so 0.1 s cannot confuse two pulses
    but allows for the few ms the stimulator takes to start and for PC timing jitter.
    """
    pulses = [s for s in stim if s.kind == "Single Pulse"]
    sources = {s.source for s in pulses}
    logged_outputs = {f"STG {int(k)}" for k in np.unique(log.stg)}
    absent = sorted(logged_outputs - sources)
    if absent:
        raise ValueError(f"{log.path.name} logs pulses on {absent}, but the recording has no events for them")
    out, reports = [], []
    for s in stim:
        if s.kind != "Single Pulse" or s.source not in logged_outputs:
            out.append(s)
            continue
        rows = log.for_output(s.source)
        match, method, residual = _match(s.onsets_s, log.time_s[rows], tolerance_s)
        sites = [""] * s.n
        for r, k in enumerate(match):
            if k >= 0:
                sites[k] = log.label[rows[r]]
        out.append(s.with_pulse_sites(sites))
        matched = {int(k) for k in match if k >= 0}
        reports.append(Alignment(
            source=s.source, n_logged=int(rows.size), n_recorded=s.n, n_matched=len(matched), method=method,
            max_residual_ms=residual * 1e3, unmatched_logged=tuple(int(log.seq[rows[r]]) for r in np.flatnonzero(match < 0)),
            unmatched_recorded=tuple(k for k in range(s.n) if k not in matched),
        ))  # fmt: skip
    return out, reports


def _match(recorded: np.ndarray, logged: np.ndarray, tol: float) -> tuple[np.ndarray, str, float]:
    """For each logged pulse, the index of its recorded pulse (-1 if none)."""
    rec, lg = np.asarray(recorded, float), np.asarray(logged, float)
    if rec.size == lg.size and rec.size:
        # Same counts: match by order, if the intervals agree (a line absorbs offset and clock drift).
        fit = np.polyval(np.polyfit(lg, rec, 1), lg) if rec.size > 1 else rec
        residual = float(np.max(np.abs(rec - fit)))
        if residual <= tol:
            return np.arange(rec.size), "order", residual
    match = np.full(lg.size, -1)
    if rec.size == 0 or lg.size == 0:
        return match, "intervals", float("nan")
    # Clock offset: the shift that lines up the most logged pulses with recorded ones.
    head = lg[: min(lg.size, 50)]
    candidates = (rec[None, :] - head[: min(head.size, 10), None]).ravel()  # pair one of the first logged pulses with any recorded one
    shifted = head[None, :] + candidates[:, None]
    k = np.clip(np.searchsorted(rec, shifted), 1, rec.size - 1) if rec.size > 1 else np.zeros(shifted.shape, int)
    nearest = np.minimum(np.abs(rec[k] - shifted), np.abs(rec[np.maximum(k - 1, 0)] - shifted))
    offset = float(candidates[int(np.argmax(np.sum(nearest <= tol, axis=1)))])
    # Walk in order, predicting each pulse from the last matched one, so slow drift is followed.
    last_rec, last_log, used, residuals = None, None, -1, []
    for i, t in enumerate(lg):
        expected = t + offset if last_rec is None else last_rec + (t - last_log)
        k = int(np.searchsorted(rec, expected))
        best = min((j for j in (k - 1, k) if used < j < rec.size), key=lambda j: abs(rec[j] - expected), default=None)
        if best is not None and abs(rec[best] - expected) <= tol:
            match[i], used = best, best
            residuals.append(abs(rec[best] - expected))
            last_rec, last_log = rec[best], t
    return match, "intervals", float(max(residuals)) if residuals else float("nan")
