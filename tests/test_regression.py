"""The refactored pipeline (``legacy`` profile) reproduces the old ``spikes.py`` output.

Baseline: the committed sidecars in tests/data/legacy_baseline (DECISIONS.md D7).
Tolerances (DECISIONS.md D11):
- >= 99 % of spikes matched within +-1 sample, in both directions
- noise sigma within 0.1 %
- identically timed waveforms within 0.01 µV
- spikes in the first 50 ms are ignored: a pulse at t = 0 has no left anchor sample, and the
  legacy code bridged from the channel median while meagraph holds the first sample after it
"""

from pathlib import Path

import numpy as np
import pytest

from conftest import REPO
from meagraph.detect import detect_spikes
from meagraph.io import load_session, read_spikes_sidecar

pytestmark = [pytest.mark.data, pytest.mark.slow, pytest.mark.filterwarnings("ignore::UserWarning")]
BASELINE = REPO / "tests" / "data" / "legacy_baseline"
EDGE_S = 0.05


def _samples(times_s, t_start, fs):
    s = np.round((np.asarray(times_s) - t_start) * fs).astype(np.int64)
    return s[s >= EDGE_S * fs]


def _matched(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each sample in ``a``, whether ``b`` has one within +-1 sample."""
    if a.size == 0 or b.size == 0:
        return np.zeros(a.size, dtype=bool)
    i = np.clip(np.searchsorted(b, a), 1, max(b.size - 1, 1))
    lo, hi = b[np.clip(i - 1, 0, b.size - 1)], b[np.clip(i, 0, b.size - 1)]
    return np.minimum(np.abs(lo - a), np.abs(hi - a)) <= 1


@pytest.mark.parametrize("sidecar", sorted(BASELINE.glob("*.spikes.h5")), ids=lambda p: p.name[:19])
def test_legacy_profile_reproduces_spikes_py(real_files, sidecar: Path):
    recording = next((p for p in real_files if p.name == sidecar.name.replace(".spikes.h5", ".h5")), None)
    if recording is None:
        pytest.skip("recording for this baseline is not present")
    session = load_session(recording)
    rec = session.recording
    fs, t0 = rec.get_sampling_frequency(), rec.get_start_time()
    new = detect_spikes(rec, session.stim, "legacy")
    old = read_spikes_sidecar(sidecar)

    n_old = n_new = hit_old = hit_new = 0
    for unit, old_t in old.trains.as_dict().items():
        new_t = new.trains.as_dict().get(unit, np.zeros(0))
        a, b = _samples(old_t, t0, fs), _samples(new_t, t0, fs)
        n_old, n_new = n_old + a.size, n_new + b.size
        hit_old += int(_matched(a, b).sum())
        hit_new += int(_matched(b, a).sum())
        # Spikes found at exactly the same sample must have the same filtered waveform.
        old_at = {s: i for i, s in enumerate(np.round((old_t - t0) * fs).astype(np.int64))}
        new_at = {s: i for i, s in enumerate(np.round((new_t - t0) * fs).astype(np.int64))}
        for s in np.intersect1d(a, b):
            np.testing.assert_allclose(
                new.waveforms_uv[unit][new_at[s]], old.waveforms_uv[unit][:, old_at[s]], atol=0.01,
                err_msg=f"waveform of the spike at sample {s} on {unit}",
            )  # fmt: skip
        if unit in new.noise_uv:
            assert new.noise_uv[unit] == pytest.approx(old.sigma_uv[unit], rel=1e-3), unit
    assert hit_old / n_old >= 0.99, f"{n_old - hit_old} of {n_old} legacy spikes not reproduced"
    assert hit_new / n_new >= 0.99, f"{n_new - hit_new} of {n_new} new spikes not in the legacy output"
