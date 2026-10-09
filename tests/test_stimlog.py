"""Stimulation logs: reading, finding, matching to recorded pulses, and their use downstream (D25)."""

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording

from meagraph.cli import main
from meagraph.detect import detect_spikes, load_detection, save_detection
from meagraph.io import align_stimlog, find_stimlog, load_session, read_stimlog
from meagraph.io.mcs_events import StimEvents
from meagraph.stimulation.audit import check_logged_sites
from synthetic import FS, make_recording

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

HEADER = "trial,seq,stg,api_index,label,amplitude,unit,phase_us,offset_ms,interval_s,pc_time,pc_elapsed_s"


def write_log(path, pulses, notes=None):
    """``pulses``: (stg, label, pc_elapsed_s, offset_ms) per pulse, in seq order."""
    lines = [HEADER]
    for seq, (stg, label, t, offset) in enumerate(pulses, 1):
        lines.append(f"{seq},{seq},{stg},0,{label},1000,uV,300,{offset},3.0,2026-10-08T12:00:00.000,{t}")
    path.write_text("\n".join(lines) + "\n")
    if notes is not None:
        path.with_name(path.name.replace(".stimlog.csv", ".notes.yaml")).write_text(notes)
    return path


# -- reading and finding --------------------------------------------------------------------- #
def test_read_stimlog(tmp_path):
    p = write_log(tmp_path / "Day 143 Random 1.stimlog.csv", [(1, "47", 10.0, 0), (2, "12", 10.0, 150.0)], notes="div: 143\n")
    log = read_stimlog(p)
    assert log.tag == "Day 143 Random 1" and log.label == ("47", "12")
    np.testing.assert_allclose(log.time_s, [10.0, 10.15])  # trial time + offset within the trial
    assert log.notes == {"div": 143}
    np.testing.assert_array_equal(log.for_output("STG 2"), [1])
    (tmp_path / "bad.stimlog.csv").write_text("seq,stg\n1,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        read_stimlog(tmp_path / "bad.stimlog.csv")


def test_find_stimlog_by_name_and_by_tag(tmp_path):
    rec = tmp_path / "2026-10-08T12-00-00Day 143 Random 1_E-00303.h5"
    rec.write_bytes(b"")
    assert find_stimlog(rec) is None
    write_log(tmp_path / "Day 143 Random 1.stimlog.csv", [(1, "47", 1.0, 0)])
    write_log(tmp_path / "Random 1.stimlog.csv", [(1, "47", 1.0, 0)])
    assert find_stimlog(rec).name == "Day 143 Random 1.stimlog.csv"  # the longest tag in the name wins
    exact = write_log(tmp_path / (rec.stem + ".stimlog.csv"), [(1, "47", 1.0, 0)])
    assert find_stimlog(rec) == exact  # an exact name beats tags
    write_log(tmp_path / "Twin A.stimlog.csv", [(1, "47", 1.0, 0)])
    write_log(tmp_path / "Twin B.stimlog.csv", [(1, "47", 1.0, 0)])
    with pytest.raises(ValueError, match="several stimulation logs"):  # two tags of equal length: ambiguous
        find_stimlog(tmp_path / "2026-10-08T13-00-00Twin A Twin B_E-00303.h5")


# -- matching ---------------------------------------------------------------------------------- #
def _stim(onsets, source="STG 1"):
    onsets = np.asarray(onsets, float)
    return StimEvents(source, "Single Pulse", onsets, onsets + 0.001)


def test_align_by_order_on_both_outputs(tmp_path):
    t1 = np.array([1.0, 3.5, 6.1, 9.0])
    t2 = t1 + 0.2
    log = read_stimlog(write_log(tmp_path / "a.stimlog.csv", [
        (1, "47", 101.0, 0), (2, "12", 101.0, 200), (1, "33", 103.5, 0), (2, "47", 103.5, 200),
        (1, "12", 106.1, 0), (2, "33", 106.1, 200), (1, "47", 109.0, 0), (2, "12", 109.0, 200)]))  # fmt: skip
    stim, reports = align_stimlog([_stim(t1), _stim(t2, "STG 2")], log)
    assert stim[0].pulse_sites == ("47", "33", "12", "47") and stim[1].pulse_sites == ("12", "47", "33", "12")
    assert all(r.complete and r.method == "order" for r in reports)


def test_align_reports_a_pulse_missing_from_the_recording(tmp_path):
    rng = np.random.default_rng(1)
    t = 5 + np.cumsum(rng.uniform(2, 4, 30))
    labels = [str(10 + k % 7) for k in range(30)]
    log = read_stimlog(write_log(tmp_path / "a.stimlog.csv", [(1, lab, ti + 40.0, 0) for lab, ti in zip(labels, t)]))
    recorded = np.delete(t, 12)  # the recording missed pulse 13
    (stim,), (report,) = align_stimlog([_stim(recorded)], log)
    assert report.method == "intervals" and report.unmatched_logged == (13,) and report.n_matched == 29
    assert list(stim.pulse_sites) == labels[:12] + labels[13:]


def test_align_rejects_a_log_for_an_output_without_events(tmp_path):
    log = read_stimlog(write_log(tmp_path / "a.stimlog.csv", [(2, "47", 1.0, 0)]))
    with pytest.raises(ValueError, match="no events"):
        align_stimlog([_stim([1.0])], log)


# -- sessions ---------------------------------------------------------------------------------- #
@pytest.fixture
def recording_with_log(make_mcs_file):
    path, _ = make_mcs_file(labels=("47", "12", "33", "15"), n_samples=60_000,
                            stim_onsets_us=[1_000_000, 3_300_000, 5_200_000], stim2_onsets_us=[1_200_000, 3_500_000])  # fmt: skip
    tag = "Random 1"
    path = path.rename(path.with_name(f"2026-10-08T12-00-00{tag}_E-00303.h5"))
    write_log(path.with_name(tag + ".stimlog.csv"),
              [(1, "47", 20.0, 0), (2, "12", 20.0, 200), (1, "33", 22.3, 0), (2, "47", 22.3, 200), (1, "12", 24.2, 0)])  # fmt: skip
    return path


def test_session_finds_and_applies_the_log(recording_with_log):
    s = load_session(recording_with_log)
    assert s.stimlog is not None and all(a.complete for a in s.alignment)
    sites = {e.source: e.pulse_sites for e in s.stim if e.kind == "Single Pulse"}
    assert sites == {"STG 1": ("47", "33", "12"), "STG 2": ("12", "47")}
    assert all(e.pulse_sites is None for e in load_session(recording_with_log, stimlog=None).stim)
    with pytest.raises(ValueError, match="both stim_site and a stimulation log"):
        load_session(recording_with_log, stim_site={"STG 1": "47", "STG 2": "12"})


def test_session_rejects_logged_sites_that_are_not_channels(recording_with_log):
    log = recording_with_log.with_name("Random 1.stimlog.csv")
    log.write_text(log.read_text().replace(",33,", ",99,"))
    with pytest.raises(ValueError, match="not channels"):
        load_session(recording_with_log)


def test_cli_reports_the_log(recording_with_log, capsys):
    assert main(["info", str(recording_with_log)]) == 0
    assert "stimulation log: Random 1.stimlog.csv" in capsys.readouterr().out
    assert main(["audit", str(recording_with_log)]) == 0
    out = capsys.readouterr().out
    assert "STG 1: 3 of 3 recorded pulses matched to the log" in out and "STG 2: 2 of 2" in out


# -- detection and audit ----------------------------------------------------------------------- #
def test_logged_sites_are_kept_and_blanked_after_their_own_pulses(tmp_path):
    onsets = np.arange(0.5, 19.5, 0.5)
    rec, _, stim = make_recording(duration_s=20.0, stim_onsets_s=onsets, spike_channel=2, spike_rate_hz=20.0)
    sites = ["2" if k % 2 == 0 else "" for k in range(onsets.size)]
    res = detect_spikes(rec, [stim.with_pulse_sites(sites)])
    assert res.excluded == {}  # per-pulse sites stay in detection
    t = res.trains.as_dict()["2"]
    after = t[:, None] - stim.ends_s[None, :]
    in_50ms = (after >= 0) & (after < 0.049)
    own = np.array([s == "2" for s in sites])
    assert not in_50ms[:, own].any()  # blanked for the maximum after its own pulses
    assert in_50ms[:, ~own].any()  # but not after other sites' pulses
    back = load_detection(save_detection(res, tmp_path / "det"))
    assert back.stim[0].pulse_sites == tuple(sites)


def test_audit_checks_logged_sites_against_the_artifact():
    rng = np.random.default_rng(0)
    n, ids = int(10 * FS), ["0", "1", "2", "3"]
    x = rng.normal(0, 5.0, (n, 4)).astype(np.float32)
    onsets = np.arange(0.5, 9.5, 0.5)
    sites = [ids[k % 4] for k in range(onsets.size)]
    for t, site in zip(onsets, sites):
        i = int(t * FS)
        x[i : i + 10] += 100.0
        x[i : i + 10, ids.index(site)] += 2000.0  # the stimulated electrode has the largest artifact
    rec = NumpyRecording([x], sampling_frequency=FS, channel_ids=ids)
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)
    stim = StimEvents("STG 1", "Single Pulse", onsets, onsets + 0.001)
    good = check_logged_sites(rec, stim.with_pulse_sites(sites))
    assert [c[1] for c in good] == [1, 1, 1, 1]
    wrong = check_logged_sites(rec, stim.with_pulse_sites([ids[(k + 1) % 4] for k in range(onsets.size)]))
    assert all(c[1] != 1 for c in wrong)  # a shifted table is caught


def test_find_stimlog_by_start_time(make_mcs_file):
    path, _ = make_mcs_file(n_samples=600_000)  # 60.5 s on the recording clock (Duration)
    rec = path.rename(path.with_name("2026-10-08T19-17-36Flex_Electronics_Andre_Random1_2.h5"))
    assert find_stimlog(rec) is None
    write_log(rec.with_name("2026-10-08T19-15-00.stimlog.csv"), [(1, "47", 1.0, 0)])  # started before the recording
    write_log(rec.with_name("2026-10-08T19-19-00.stimlog.csv"), [(1, "47", 1.0, 0)])  # started after it ended
    assert find_stimlog(rec) is None
    inside = write_log(rec.with_name("2026-10-08T19-17-47.stimlog.csv"), [(1, "47", 1.0, 0)])
    assert find_stimlog(rec) == inside  # the script's default name, started during the recording
    write_log(rec.with_name("2026-10-08T19-18-10.stimlog.csv"), [(1, "47", 1.0, 0)])
    with pytest.raises(ValueError, match="several stimulation logs started during"):
        find_stimlog(rec)


def test_marker_events_are_not_counted_as_pulses():
    from meagraph.cli import _pulses

    on = np.array([1.0, 2.0])
    stim = [StimEvents("STG 1", "Single Pulse", on, on + 0.001), StimEvents("STG 1", "Marker", on, on + 0.001),
            StimEvents("STG 2", "Marker", on, on + 0.001)]  # fmt: skip
    kept = _pulses(stim)
    assert [(s.source, s.kind) for s in kept] == [("STG 1", "Single Pulse"), ("STG 2", "Marker")]


def test_graph_explains_why_there_is_no_graph(tmp_path, capsys):
    from meagraph.detect import DetectionConfig, DetectionResult
    from meagraph.detect.threshold import ChannelQC
    from meagraph.spiketrains import SpikeTrains

    trains = SpikeTrains.from_dict({"47": [1.0, 2.0], "12": [1.5]}, 0.0, 10.0)
    det = DetectionResult(
        trains=trains, amplitudes_uv={u: np.zeros(t.size, np.float32) for u, t in trains.as_dict().items()},
        waveforms_uv={u: np.zeros((t.size, 30), np.float32) for u, t in trains.as_dict().items()},
        noise_uv={"47": 9.0, "12": 9.5}, excluded={}, recovery_ms={}, config=DetectionConfig(),
        qc=ChannelQC(("47", "12"), np.array([2, 1]), np.array([80, 90]), np.ones(2), np.zeros(2, bool)),
    )  # fmt: skip
    folder = save_detection(det, tmp_path / "results" / "rec" / "detect")
    assert main(["graph", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "0 of 2 electrodes passed the signal-quality check" in out and "3 negative vs 170 positive" in out
