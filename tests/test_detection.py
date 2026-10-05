import numpy as np
import pytest
from spikeinterface.core import NumpyRecording

from meagraph.detect import detect_spikes, load_detection, median_abs_noise_uv, save_detection
from meagraph.io.mcs_events import StimEvents
from meagraph.preprocess import InterpolateWindowsRecording, detection_band, merge_windows
from meagraph.stimulation import group_trains, infer_site, measure_recovery
from synthetic import FS, make_recording, match_fraction


# -- window bridging -------------------------------------------------------------------- #
def test_merge_windows_clips_sorts_and_merges():
    w = merge_windows([[50, 60], [-5, 3], [58, 70], [70, 75], [200, 300]], n_samples=100)
    np.testing.assert_array_equal(w, [[0, 3], [50, 75]])
    assert merge_windows(np.zeros((0, 2)), 10).shape == (0, 2)


def _ramp(n=1000, n_ch=2):
    x = np.tile(np.arange(n, dtype=np.float32)[:, None] ** 1.5, (1, n_ch))
    rec = NumpyRecording([x], sampling_frequency=FS, channel_ids=[str(c) for c in range(n_ch)])
    return rec, x


def test_bridge_is_a_straight_line_between_anchor_samples():
    rec, x = _ramp()
    y = InterpolateWindowsRecording(rec, [[100, 120]]).get_traces()
    np.testing.assert_allclose(y[100:120, 0], np.linspace(x[99, 0], x[120, 0], 20), rtol=1e-6)
    np.testing.assert_array_equal(y[:100], x[:100])
    np.testing.assert_array_equal(y[120:], x[120:])


def test_bridge_is_identical_when_read_in_pieces():
    rec, _ = _ramp()
    blanked = InterpolateWindowsRecording(rec, [[100, 120], [500, 530]])
    full = blanked.get_traces()
    pieces = np.concatenate([blanked.get_traces(start_frame=a, end_frame=a + 7) for a in range(0, 1000, 7)])
    np.testing.assert_array_equal(pieces[:1000], full)


def test_bridge_per_channel_and_at_recording_edges():
    rec, x = _ramp()
    y = InterpolateWindowsRecording(rec, {"1": [[0, 10], [990, 1000]]}).get_traces()
    np.testing.assert_array_equal(y[:, 0], x[:, 0])  # channel 0 untouched
    np.testing.assert_allclose(y[:10, 1], x[10, 1])  # no left anchor: hold the right one
    np.testing.assert_allclose(y[990:, 1], x[989, 1])
    with pytest.raises(KeyError):
        InterpolateWindowsRecording(rec, {"9": [[0, 1]]})


# -- noise ------------------------------------------------------------------------------- #
def test_noise_matches_exact_median():
    rec, _, _ = make_recording(duration_s=5.0, spike_rate_hz=1.0)
    x = rec.get_traces()
    expected = np.median(np.abs(x), axis=0) / 0.6745
    np.testing.assert_allclose(median_abs_noise_uv(rec, chunk_duration_s=0.7), expected, rtol=3e-4)


# -- detection --------------------------------------------------------------------------- #
def test_detects_known_spikes_and_flags_active_channel():
    rec, truth, _ = make_recording(duration_s=20.0, spike_rate_hz=5.0)
    res = detect_spikes(rec)
    found = np.round(res.trains.as_dict()["0"] * FS).astype(np.int64)
    assert match_fraction(found, truth) >= 0.97
    assert match_fraction(truth, found) >= 0.97  # few false positives on the spiking channel
    assert res.trains.n_spikes()[1:].sum() <= 2  # noise-only channels
    assert res.active_channels == ("0",)
    assert res.waveforms_uv["0"].shape[1] == 30
    np.testing.assert_allclose(res.amplitudes_uv["0"], res.waveforms_uv["0"][:, 10])


def test_qc_is_not_fooled_by_symmetric_ringing():
    """Paired +/- crossings from oscillatory bursts must not look like negative spikes."""
    rec, _, _ = make_recording(duration_s=20.0, spike_rate_hz=5.0)
    x = rec.get_traces().copy()
    rng = np.random.default_rng(1)
    t = np.arange(20)
    for start in rng.integers(100, x.shape[0] - 100, size=150):
        phase = rng.uniform(0, 2 * np.pi)
        x[start : start + 20, 1] += 60.0 * np.sin(2 * np.pi * 1500 * t / FS + phase) * np.hanning(20)
    rec2 = NumpyRecording([x], sampling_frequency=FS, channel_ids=rec.channel_ids)
    rec2.set_channel_gains(1.0)
    rec2.set_channel_offsets(0.0)
    res = detect_spikes(rec2)
    assert res.qc.n_negative[1] + res.qc.n_positive[1] > 100  # the bursts do cross threshold
    assert res.active_channels == ("0",)


def test_stimulation_is_blanked_site_excluded_and_recovery_measured():
    onsets = np.arange(0.5, 19.5, 0.5)
    rec, truth, stim = make_recording(duration_s=20.0, stim_onsets_s=onsets, tail_ms=3.0)
    stim = stim.with_site("2")
    rec_ms = measure_recovery(rec, stim)
    assert np.all(rec_ms.recovery_ms > 1.0) and np.all(rec_ms.recovery_ms < rec_ms.max_post_ms)

    res = detect_spikes(rec, [stim])
    t = res.trains.as_dict()
    # no detection inside any pulse (+ 2 ms) on any channel
    for spikes in t.values():
        d = spikes[:, None] - onsets[None, :]
        assert not np.any((d >= -1e-3) & (d < 4e-3))
    assert match_fraction(np.round(t["0"] * FS).astype(np.int64), truth) >= 0.95
    assert t["2"].size == 0 and res.excluded == {"2": "stimulation site (STG 1)"}
    assert res.trains.n_spikes()[1] <= 2  # the artifact did not leak into noise-only channels


def test_detection_result_roundtrip(tmp_path):
    rec, _, stim = make_recording(duration_s=5.0, stim_onsets_s=[1.0, 2.0, 3.0, 4.0])
    res = detect_spikes(rec, [stim])
    back = load_detection(save_detection(res, tmp_path / "det"))
    assert back.config == res.config
    for u in res.trains.unit_ids:
        np.testing.assert_allclose(back.trains.as_dict()[u], res.trains.as_dict()[u])
        np.testing.assert_allclose(back.waveforms_uv[u], res.waveforms_uv[u])
    assert back.active_channels == res.active_channels
    np.testing.assert_allclose(list(back.recovery_ms["STG 1"].values()), list(res.recovery_ms["STG 1"].values()), atol=0.01)
    assert (tmp_path / "det" / "provenance.json").exists() and (tmp_path / "det" / "channels.csv").exists()
    # The stimulation events travel with the spikes, exactly.
    (s0,), (s1,) = res.stim, back.stim
    assert (s1.source, s1.kind, s1.site) == (s0.source, s0.kind, s0.site)
    np.testing.assert_array_equal(s1.onsets_s, s0.onsets_s)
    np.testing.assert_array_equal(s1.offsets_s, s0.offsets_s)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_detection_folder_keeps_positions_and_rejects_old_format(make_mcs_file, tmp_path):
    from meagraph.io import load_session

    path, _ = make_mcs_file(labels=("47", "12", "33", "15"), n_samples=20_000)
    session = load_session(path)
    res = detect_spikes(session.recording, session.stim)
    assert res.stim == ()
    folder = save_detection(res, tmp_path / "det")
    back = load_detection(folder)
    np.testing.assert_allclose(back.trains.positions_um, session.recording.get_channel_locations(axes="xyz"), rtol=1e-6)
    assert back.stim == ()
    (folder / "stimulation.csv").unlink()
    with pytest.raises(ValueError, match="re-run `meagraph detect`"):
        load_detection(folder)


# -- stimulation helpers ------------------------------------------------------------------ #
def test_group_trains():
    t = np.array([0.0, 0.003, 0.006, 5.0, 10.0, 10.003])
    assert [len(g) for g in group_trains(t)] == [3, 1, 2]


def test_site_inference_needs_an_unambiguous_channel():
    rec, _, stim = make_recording(duration_s=3.0, stim_onsets_s=[1.0, 2.0], artifact_uv=500.0)
    x = rec.get_traces().copy()
    for on in (1.0, 2.0):
        i = int(on * FS)
        x[i : i + 20, 1] -= 5_000.0  # channel 1 carries a much larger artifact
    rec2 = NumpyRecording([x], sampling_frequency=FS, channel_ids=rec.channel_ids)
    assert infer_site(rec2, stim).site == "1"
    assert infer_site(rec, stim).site is None  # all channels alike: refuse to guess


def test_detection_band_is_lazy_and_float32():
    rec, _, _ = make_recording(duration_s=1.0)
    filt = detection_band(rec, windows=[[100, 200]])
    assert filt.get_dtype() == np.float32
    assert filt.get_traces(start_frame=0, end_frame=10).shape == (10, 3)


def test_unpaired_events_use_onsets_for_windows():
    rec, _, _ = make_recording(duration_s=2.0)
    stim = StimEvents("STG 1", "Single Pulse", np.array([0.5]), None)
    res = detect_spikes(rec, [stim])
    assert set(res.recovery_ms) == {"STG 1"}
