"""Checks against the real recordings (docs/DATA_FORMAT.md). Skipped when the files are absent."""

import importlib.util

import h5py
import numpy as np
import pytest

from conftest import REPO
from meagraph.io import (
    McsH5Recording,
    inspect_file,
    load_session,
    read_spikes_sidecar,
    read_stim_events,
    sidecar_path,
)

pytestmark = [pytest.mark.data, pytest.mark.filterwarnings("ignore::meagraph.probe.UnverifiedGeometryWarning")]


def _by_tag(files, tag):
    hits = [p for p in files if tag in p.name]
    if not hits:
        pytest.skip(f"no recording matching {tag!r}")
    return hits[0]


def _legacy_mcs():
    path = REPO / "mcs.py"
    if not path.exists():
        pytest.skip("legacy mcs.py not present")
    spec = importlib.util.spec_from_file_location("legacy_mcs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reader_matches_independent_h5py_computation(real_files):
    for path in real_files:
        rec = McsH5Recording(path)
        i0, i1 = 50_000, 60_000
        with h5py.File(path, "r") as f:
            g = f["Data/Recording_0/AnalogStream/Stream_0"]
            info = g["InfoChannel"][:]
            rows = g["ChannelData"][:, i0:i1].astype(np.float64)
        gain = info["ConversionFactor"] * 10.0 ** info["Exponent"] * 1e6
        expected = ((rows[info["RowIndex"]] - info["ADZero"][:, None]) * gain[:, None]).T
        np.testing.assert_allclose(rec.get_traces(start_frame=i0, end_frame=i1, return_in_uV=True), expected, rtol=1e-6, atol=1e-4)


def test_reader_is_bit_identical_to_legacy(real_files):
    mcs = _legacy_mcs()
    for path in real_files:
        rec = McsH5Recording(path)
        with mcs.McsFile(str(path)) as m:
            old = m.analog_stream(0).read_uv(10.0, 11.0, channels=list(rec.channel_ids)).T
        new = rec.get_traces(start_frame=100_000, end_frame=110_000, return_in_uV=True)
        np.testing.assert_array_equal(new, old)


def test_time_base_and_sampling_rate(real_files):
    for path in real_files:
        rec = McsH5Recording(path)
        assert rec.get_sampling_frequency() == 10_000.0
        assert rec.get_start_time() == pytest.approx(0.5 if "beforestim" in path.name else 0.0)


def test_stream_lineage(real_files):
    for path in real_files:
        inv = inspect_file(path)
        assert [s.short_label for s in inv.analog_streams] == ["Data Acquisition (1)", "Filter (1)", "Filter (2)", "Filter (3)"]
        assert [s.name for s in inv.analog_streams] == ["Stream_0", "Stream_3", "Stream_2", "Stream_1"]


def test_stimulation_protocol(real_files):
    for path in real_files:
        stim = read_stim_events(path)
        if "beforestim" in path.name:
            assert stim == []
            continue
        (s,) = stim
        assert (s.source, s.kind, s.n) == ("STG 1", "Single Pulse", 26)
        np.testing.assert_allclose(s.durations_s, 0.003, atol=1e-9)
        np.testing.assert_allclose(np.diff(s.onsets_s), 4.003, atol=1e-9)


def test_artifact_starts_at_pulse_start(real_files):
    """EventEntity_1 is the pulse onset: the artifact rises within 0.3 ms after it, not before."""
    path = _by_tag(real_files, "stim47")
    rec = McsH5Recording(path)
    (stim,) = read_stim_events(path)
    fs = rec.get_sampling_frequency()
    for onset in stim.onsets_s[1:4]:
        i = int(round((onset - rec.get_start_time()) * fs))
        x = rec.get_traces(start_frame=i - 20, end_frame=i + 40, channel_ids=["47"], return_in_uV=True)[:, 0]
        baseline_sd = np.std(x[:15])
        first = np.flatnonzero(np.abs(x - np.median(x[:15])) > 1000 + 20 * baseline_sd)[0] - 20
        assert 0 <= first <= 3


def test_session_on_cube(real_files):
    path = _by_tag(real_files, "stim47")
    session = load_session(path, stim_site="47")
    assert session.recording.get_num_channels() == 59
    assert session.excluded_channel_ids == ("15",)
    assert session.recording.get_probe().ndim == 3
    assert session.stim[0].site == "47"


def test_legacy_sidecars_sit_inside_the_recording(real_files):
    for path in real_files:
        side = sidecar_path(path)
        if not side.exists():
            continue
        legacy = read_spikes_sidecar(side)
        rec = McsH5Recording(path)
        assert legacy.trains.t_start_s == pytest.approx(rec.get_start_time())
        assert legacy.trains.t_stop_s == pytest.approx(rec.get_start_time() + rec.get_total_duration())
