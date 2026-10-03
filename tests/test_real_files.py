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


# Known stimulation per recording (docs/DATA_FORMAT.md): {source: (pulses, pulse duration s)}.
KNOWN_STIM = {
    "2026-07-27T10-52-01": {},
    "2026-07-27T11-15-56": {"STG 1": (26, 0.003)},
    "2026-07-27T12-04-26": {"STG 1": (26, 0.003)},
    "2026-07-29T15-16-32": {},
    "2026-07-29T15-32-23": {"STG 1": (300, None), "STG 2": (300, None)},  # 2.0-2.1 ms pulses
}


def test_stimulation_protocol(real_files):
    for path in real_files:
        expected = KNOWN_STIM.get(path.name[:19])
        if expected is None:
            continue
        stim = {s.source: s for s in read_stim_events(path)}
        assert {k: s.n for k, s in stim.items()} == {k: v[0] for k, v in expected.items()}, path.name
        for source, (_, duration) in expected.items():
            s = stim[source]
            assert s.kind == "Single Pulse" and s.offsets_s is not None
            if duration is not None:
                np.testing.assert_allclose(s.durations_s, duration, atol=1e-9)
    exp3 = [p for p in real_files if p.name.startswith("2026-07-27T11-15-56")]
    if exp3:
        np.testing.assert_allclose(np.diff(read_stim_events(exp3[0])[0].onsets_s), 4.003, atol=1e-9)


def test_associative_protocol_structure(real_files):
    """Trains of 1, 2 or 3 pulses every 5 s, in alternating blocks of 50 trains per STG output."""
    from meagraph.stimulation import group_trains

    path = _by_tag(real_files, "Associative Stimulation 1")
    for s in read_stim_events(path):
        trains = group_trains(s.onsets_s)
        assert len(trains) == 150
        assert sorted({len(t) for t in trains}) == [1, 2, 3]
        starts = np.array([t[0] for t in trains])
        gaps = np.round(np.diff(starts), 3)
        assert set(gaps) == {5.0, 375.0} and (gaps == 375.0).sum() == 2


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


def test_site_inference_on_real_stimulation(real_files):
    from meagraph.stimulation import infer_site

    for tag, site in (("11-15-56", "12"), ("stim47", "47")):
        path = _by_tag(real_files, tag)
        rec = McsH5Recording(path)
        assert infer_site(rec, read_stim_events(path)[0]).site == site
    path = _by_tag(real_files, "Associative Stimulation 1")
    rec = McsH5Recording(path)
    for stim in read_stim_events(path):
        result = infer_site(rec, stim)
        assert result.site is None and "ADC rail" in result.reason  # the whole array saturates


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
