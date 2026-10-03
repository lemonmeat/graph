"""Save and load detection results.

A result folder holds:
- ``spikes.npz``: flat arrays ``unit_id``, ``time_s``, ``amplitude_uv``, ``waveform_uv`` (n, samples)
- ``channels.csv``: one row per channel (spike count, rate, noise, QC, exclusion, recovery)
- ``config.yaml`` and ``provenance.json`` (see :func:`meagraph.config.write_run_folder`)
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from meagraph.config import load_yaml, write_run_folder
from meagraph.detect.threshold import ChannelQC, DetectionConfig, DetectionResult
from meagraph.spiketrains import SpikeTrains

_COLUMNS = ["channel_id", "n_spikes", "rate_hz", "noise_uv", "qc_negative", "qc_positive", "qc_p_value",
            "active", "excluded", "recovery_ms"]  # fmt: skip


def default_detection_folder(recording_path: str | Path, profile: str = "default") -> Path:
    """``results/<recording name>/detect_<profile>`` next to the recording."""
    p = Path(recording_path)
    return p.parent / "results" / p.stem / f"detect_{profile}"


def find_detection(recording_path: str | Path) -> Path | None:
    """Most recently written detection folder for a recording, if any."""
    root = Path(recording_path).parent / "results" / Path(recording_path).stem
    found = sorted(root.glob("detect_*/spikes.npz"), key=lambda p: p.stat().st_mtime)
    return found[-1].parent if found else None


def save_detection(result: DetectionResult, folder: str | Path, inputs: Sequence[str | Path] = (), overwrite: bool = False) -> Path:
    out = write_run_folder(folder, result.config, inputs=inputs, overwrite=overwrite)
    tr = result.trains
    if tr.n_units == 0:
        raise ValueError("nothing to save: the result has no channels")
    # Every channel's arrays exist, empty ones included, so concatenation keeps the waveform width.
    ids = np.concatenate([np.full(t.size, u) for u, t in zip(tr.unit_ids, tr.times_s)])
    np.savez_compressed(
        out / "spikes.npz",
        unit_id=ids.astype(str),
        time_s=np.concatenate(tr.times_s),
        amplitude_uv=np.concatenate([result.amplitudes_uv[u] for u in tr.unit_ids]).astype(np.float32),
        waveform_uv=np.concatenate([result.waveforms_uv[u] for u in tr.unit_ids]).astype(np.float32),
        unit_order=np.array(tr.unit_ids, dtype=str),
        t_span_s=np.array([tr.t_start_s, tr.t_stop_s]),
    )
    qc = result.qc
    rates = dict(zip(tr.unit_ids, tr.rates_hz()))
    counts = dict(zip(tr.unit_ids, tr.n_spikes()))
    with open(out / "channels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_COLUMNS)
        for k, c in enumerate(qc.channel_ids):
            recovery = {src: (None if np.isnan(r[c]) else round(r[c], 2)) for src, r in result.recovery_ms.items() if c in r}
            w.writerow([
                c, counts.get(c, 0), f"{rates.get(c, 0.0):.4f}", f"{result.noise_uv[c]:.4f}",
                int(qc.n_negative[k]), int(qc.n_positive[k]), f"{qc.p_value[k]:.3g}", int(bool(qc.active[k])),
                result.excluded.get(c, ""), json.dumps(recovery) if recovery else "",
            ])  # fmt: skip
    return out


def load_detection(folder: str | Path) -> DetectionResult:
    folder = Path(folder)
    cfg = load_yaml(folder / "config.yaml", DetectionConfig)
    z = np.load(folder / "spikes.npz")
    order = [str(u) for u in z["unit_order"]]
    unit = z["unit_id"].astype(str)
    times, amps, wfs = {}, {}, {}
    for u in order:
        sel = unit == u
        times[u], amps[u], wfs[u] = z["time_s"][sel], z["amplitude_uv"][sel], z["waveform_uv"][sel]
    t0, t1 = (float(v) for v in z["t_span_s"])
    with open(folder / "channels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    recovery: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["recovery_ms"]:
            for src, ms in json.loads(r["recovery_ms"]).items():
                recovery.setdefault(src, {})[r["channel_id"]] = float("nan") if ms is None else float(ms)
    qc = ChannelQC(
        channel_ids=tuple(r["channel_id"] for r in rows),
        n_negative=np.array([int(r["qc_negative"]) for r in rows]),
        n_positive=np.array([int(r["qc_positive"]) for r in rows]),
        p_value=np.array([float(r["qc_p_value"]) for r in rows]),
        active=np.array([r["active"] == "1" for r in rows]),
    )
    return DetectionResult(
        trains=SpikeTrains.from_dict(times, t0, t1),
        amplitudes_uv=amps,
        waveforms_uv=wfs,
        noise_uv={r["channel_id"]: float(r["noise_uv"]) for r in rows},
        qc=qc,
        excluded={r["channel_id"]: r["excluded"] for r in rows if r["excluded"]},
        recovery_ms=recovery,
        config=cfg,
    )
