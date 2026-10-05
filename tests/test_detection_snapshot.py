"""Detection output on the real recordings must not change unless on purpose.

The snapshot (tests/data/detection_snapshot.json) fingerprints each recording's detection with
the default settings: spike counts, a hash of every spike's sample index, noise levels, active
and excluded channels. A failure means a code or dependency change altered which spikes are
found. If the change is intended, regenerate the snapshot and commit it with the change:

    MEAGRAPH_UPDATE_SNAPSHOT=1 pytest tests/test_detection_snapshot.py

The 32 min associative recording is left out to keep the suite near a minute.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from meagraph.detect import DetectionConfig, detect_spikes
from meagraph.io import load_session

SNAPSHOT = Path(__file__).parent / "data" / "detection_snapshot.json"
UPDATE = os.environ.get("MEAGRAPH_UPDATE_SNAPSHOT") == "1"
# The stimulated electrode is not in the files (Q6): the sites used for the snapshot.
RECORDINGS = {
    "2026-07-27T10-52-01nic_exp3_plastic_DIV140_beforestim_E-00303.h5": None,
    "2026-07-27T11-15-56nic_exp3_plastic_DIV140_stim_500width_500amplitude_4s_interval_3pulse_E-00303.h5": "12",
    "2026-07-27T12-04-26nic_exp3_plastic_DIV140_stim_500width_500amplitude_4s_interval_3pulse_stim47_E-00303.h5": "47",
    "2026-07-29T15-16-32Flex Electronics Nick Acrylic Day 142_E-00303.h5": None,
}

pytestmark = [pytest.mark.data, pytest.mark.slow, pytest.mark.filterwarnings("ignore::UserWarning")]


def fingerprint(path: Path, site: str | None) -> dict:
    session = load_session(path, stim_site=site)
    res = detect_spikes(session.recording, session.stim, DetectionConfig(n_jobs=4))
    fs, t0 = session.recording.get_sampling_frequency(), res.trains.t_start_s
    digest = hashlib.sha256()
    for u, t in zip(res.trains.unit_ids, res.trains.times_s):
        digest.update(u.encode() + np.round((t - t0) * fs).astype(np.int64).tobytes())
    return {
        "stim_site": site,
        "n_spikes": dict(zip(res.trains.unit_ids, res.trains.n_spikes().tolist())),
        "spike_samples_sha256": digest.hexdigest(),
        "noise_uv": {c: round(v, 3) for c, v in res.noise_uv.items()},
        "active": list(res.active_channels),
        "excluded": res.excluded,
    }


@pytest.mark.parametrize("name", list(RECORDINGS), ids=lambda n: n[:19])
def test_detection_matches_snapshot(real_files, name):
    path = next((p for p in real_files if p.name == name), None)
    if path is None:
        pytest.skip(f"{name} not available")
    got = fingerprint(path, RECORDINGS[name])
    stored = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    if UPDATE:
        stored[name] = got
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n")
        pytest.skip("snapshot updated")
    assert name in stored, "no snapshot for this recording; run with MEAGRAPH_UPDATE_SNAPSHOT=1"
    want = stored[name]
    assert got["n_spikes"] == want["n_spikes"]
    assert got["spike_samples_sha256"] == want["spike_samples_sha256"], "same counts, different spike times"
    assert got["active"] == want["active"] and got["excluded"] == want["excluded"]
    for c, v in want["noise_uv"].items():
        assert got["noise_uv"][c] == pytest.approx(v, abs=2e-3), c
