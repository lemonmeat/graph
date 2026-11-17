"""Stimulus conditions (amplitude, phase duration, waveform) from stimulation logs (DECISIONS.md D31).

Recordings with one condition must behave exactly as before; recordings that mix conditions are
blanked, analysed and decoded per condition.
"""

from dataclasses import replace

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording

from meagraph import SpikeTrains
from meagraph.connectivity.pipeline import condition_folder_name, save_stim_graph, stim_conditions, stim_graph_from_detection
from meagraph.detect import DetectionConfig, detect_spikes, load_detection, save_detection
from meagraph.detect.threshold import ChannelQC, DetectionResult
from meagraph.evoked import EvokedConfig, blanked_until_by_site_ms, blanked_until_ms, evoked_responses
from meagraph.io import align_stimlog, condition_label, mixed_conditions, read_stimlog
from meagraph.io.mcs_events import StimEvents
from synthetic import FS

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
FAST = EvokedConfig(n_permutations=20, n_pair_permutations=20, n_repeats=2)
HI, LO = "1000mV/500us/bi500", "250mV/500us/bi500"


def _stim(onsets, sites=None, amps_uv=None, phase_us=500.0, wave="bi500", source="STG 1"):
    onsets = np.asarray(onsets, float)
    s = StimEvents(source, "Single Pulse", onsets, onsets + 0.001)
    if sites is not None:
        s = s.with_pulse_sites(sites)
    if amps_uv is not None:
        s = s.with_pulse_params(amps_uv, "uV", [phase_us] * onsets.size, [wave] * onsets.size)
    return s


# -- labels -------------------------------------------------------------------------------------- #
def test_condition_labels_and_mixing():
    assert condition_label(1_000_000, "uV", 500, "bi500") == "1000mV/500us/bi500"
    assert condition_label(5_000, "nA", 200, "") == "5uA/200us"
    assert condition_label(float("nan"), "", float("nan"), "") == ""
    one = _stim([1.0, 2.0], amps_uv=[1e6, 1e6])
    assert not mixed_conditions(one) and list(one.conditions) == [HI, HI]
    two = _stim([1.0, 2.0, 3.0], amps_uv=[1e6, 2.5e5, 1e6])
    assert mixed_conditions(two) and mixed_conditions([one, two])
    assert not mixed_conditions(_stim([1.0, 2.0]))  # no log: no conditions
    picked = two.select(np.array([False, True, False]))
    assert picked.pulse_amplitudes == (2.5e5,) and list(picked.conditions) == [LO]


def test_log_parameters_are_attached_to_the_matched_pulses(tmp_path):
    rows = ["trial,seq,stg,api_index,label,amplitude,unit,phase_us,waveform,offset_ms,interval_s,pc_time,pc_elapsed_s"]
    t = 10.0 + np.cumsum([0, 2.7, 3.1, 2.2])
    amps = [1000000, 250000, 1000000, 250000]
    for k, (ti, a) in enumerate(zip(t, amps), 1):
        rows.append(f"{k},{k},1,0,47,{a},uV,500,bi500,0.000,3.0,2026-11-20T12:00:00.000,{ti}")
    (tmp_path / "a.stimlog.csv").write_text("\n".join(rows) + "\n")
    (stim,), _ = align_stimlog([_stim(t + 30.0)], read_stimlog(tmp_path / "a.stimlog.csv"))
    assert stim.amplitude_unit == "uV" and stim.pulse_phase_us == (500.0,) * 4
    assert list(stim.conditions) == [HI, LO, HI, LO]


# -- detection ----------------------------------------------------------------------------------- #
def _artifact_recording(onsets, tails_uv, duration_s=24.0, seed=0):
    """Noise on 3 channels; each pulse leaves a decaying tail whose size depends on its condition."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    x = rng.normal(0.0, 5.0, (n, 3))
    t = np.arange(int(0.02 * FS))
    for on, tail in zip(onsets, tails_uv):
        i = int(round(on * FS))
        x[i : i + 10] += 20_000.0
        x[i + 10 : i + 10 + t.size] += (tail * np.exp(-t / (0.002 * FS)))[:, None]
    rec = NumpyRecording([x.astype(np.float32)], sampling_frequency=FS, channel_ids=["0", "1", "2"])
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)
    return rec


def test_detection_measures_recovery_per_condition(tmp_path):
    onsets = np.arange(0.5, 23.0, 0.5)
    big = np.arange(onsets.size) % 2 == 0
    rec = _artifact_recording(onsets, np.where(big, 4000.0, 50.0))
    stim = _stim(onsets, sites=[""] * onsets.size, amps_uv=np.where(big, 1e6, 2.5e5))
    res = detect_spikes(rec, [stim])
    assert set(res.recovery_ms) == {f"STG 1 @ {HI}", f"STG 1 @ {LO}"}
    hi, lo = (np.nanmedian(list(res.recovery_for("STG 1", c).values())) for c in (HI, LO))
    assert hi > lo + 1.0, (hi, lo)
    # pooled over both conditions when switched off, as before
    pooled = detect_spikes(rec, [stim], DetectionConfig(recovery_per_condition=False))
    assert set(pooled.recovery_ms) == {"STG 1"}
    assert pooled.recovery_for("STG 1", HI) == pooled.recovery_ms["STG 1"]
    # saved and loaded with the pulses' parameters
    back = load_detection(save_detection(res, tmp_path / "det"))
    assert list(back.stim[0].conditions) == list(stim.conditions)
    assert back.recovery_for("STG 1", LO) == pytest.approx(res.recovery_for("STG 1", LO), nan_ok=True)


def test_one_condition_is_detected_as_before():
    onsets = np.arange(0.5, 23.0, 0.5)
    rec = _artifact_recording(onsets, np.full(onsets.size, 1000.0))
    plain = detect_spikes(rec, [_stim(onsets)])
    logged = detect_spikes(rec, [_stim(onsets, amps_uv=np.full(onsets.size, 1e6))])
    assert set(logged.recovery_ms) == {"STG 1"}
    assert logged.recovery_ms == plain.recovery_ms
    for a, b in zip(plain.trains.times_s, logged.trains.times_s):
        np.testing.assert_array_equal(a, b)


# -- analyses ------------------------------------------------------------------------------------ #
def _two_condition_session(seed=0, n=16):
    """Sites X and Y at two amplitudes. At HI, X drives A (15 ms) and Y drives B (25 ms); at LO
    neither drives anything. Both responses fall inside stim_evoked's 30 ms window."""
    rng = np.random.default_rng(seed)
    sites = rng.permutation(np.repeat(["X", "Y", "X", "Y"], n))
    amps = rng.permutation(np.repeat([1e6, 2.5e5], 2 * n))
    onsets = 10.0 + np.cumsum(rng.uniform(3, 5, sites.size))
    t_stop = onsets[-1] + 10
    spikes = {c: list(rng.uniform(0, t_stop, int(0.3 * t_stop))) for c in "ABC"}
    for t, s, a in zip(onsets, sites, amps):
        if a == 1e6:
            spikes["A" if s == "X" else "B"] += list(t + (0.015 if s == "X" else 0.025) + rng.normal(0, 0.002, 3))
    trains = SpikeTrains.from_dict(spikes, 0.0, t_stop)
    return trains, _stim(onsets, sites=sites, amps_uv=amps)


def _detection(trains, stim, recovery):
    k = trains.n_units
    return DetectionResult(
        trains=trains, amplitudes_uv={c: np.full(t.size, -10.0) for c, t in trains.as_dict().items()},
        waveforms_uv={c: np.zeros((t.size, 4)) for c, t in trains.as_dict().items()},
        noise_uv=dict.fromkeys(trains.unit_ids, 1.0),
        qc=ChannelQC(trains.unit_ids, np.full(k, 100), np.full(k, 10), np.full(k, 1e-9), np.ones(k, bool)),
        excluded={}, recovery_ms=recovery, config=DetectionConfig(), stim=(stim,), analog_stream=None,
    )  # fmt: skip


def test_evoked_classes_are_site_and_condition():
    trains, stim = _two_condition_session()
    res = evoked_responses(trains, [stim], np.zeros((0, 2)), config=FAST)
    assert set(res.sites) == {f"X @ {HI}", f"Y @ {HI}", f"X @ {LO}", f"Y @ {LO}"}
    sig = {(r["site"], r["channel"]) for r in res.response_table if r["q_value"] < 0.05}
    assert sig == {(f"X @ {HI}", "A"), (f"Y @ {HI}", "B")}
    pairs = {(r["a"], r["b"]): r["balanced_accuracy"] for r in res.decoding["site_pairs"]}
    assert len(pairs) == 2  # sites are paired within a condition only
    hi = next(v for (a, b), v in pairs.items() if HI in a)
    lo = next(v for (a, b), v in pairs.items() if LO in a)
    assert hi > 0.9 and lo < 0.8
    assert res.decoding["conditions"]["balanced_accuracy"] > 0.7
    assert set(res.pulses.condition) == {HI, LO}


def test_evoked_with_one_logged_condition_uses_plain_sites():
    trains, stim = _two_condition_session(seed=1)
    one = stim.with_pulse_params(np.full(stim.n, 1e6), "uV", [500.0] * stim.n, ["bi500"] * stim.n)
    res = evoked_responses(trains, [one], np.zeros((0, 2)), config=FAST)
    assert set(res.sites) == {"X", "Y"} and "conditions" not in res.decoding


def test_blanking_floors_use_each_conditions_recovery():
    trains, stim = _two_condition_session(seed=2)
    det = _detection(trains, stim, {f"STG 1 @ {HI}": {"A": 9.0, "B": 9.0, "C": 9.0},
                                    f"STG 1 @ {LO}": {"A": 2.0, "B": 2.0, "C": 2.0}})  # fmt: skip
    # pulse 1 ms + recovery + guard 1 ms; the stimulated sites X and Y are not channels here
    np.testing.assert_allclose(blanked_until_ms(det, ["A", "B", "C"]), [11.0, 11.0, 11.0])
    sites, floor = blanked_until_by_site_ms(det, ["A", "B", "C"], condition=LO)
    assert set(sites) == {"X", "Y"}
    np.testing.assert_allclose(floor, 4.0)


def test_stim_graph_per_condition(tmp_path):
    trains, stim = _two_condition_session(seed=3)
    nodes = SpikeTrains.from_dict({**trains.as_dict(), "X": np.zeros(0), "Y": np.zeros(0)}, trains.t_start_s, trains.t_stop_s)
    rec = {f"STG 1 @ {c}": dict.fromkeys(nodes.unit_ids, 2.0) for c in (HI, LO)}
    det = _detection(nodes, stim, rec)
    det = replace(det, qc=ChannelQC(nodes.unit_ids, np.full(5, 100), np.full(5, 10), np.full(5, 1e-9),
                                    np.array([True, True, True, False, False])))  # sites X and Y are not readouts
    assert stim_conditions(det) == list(dict.fromkeys(stim.conditions.tolist()))
    graphs = {c: stim_graph_from_detection(det, condition=c) for c in (HI, LO)}
    edges = {c: {(e["source"], e["target"]) for e in g.edges()} for c, g in graphs.items()}
    assert edges[HI] == {("X", "A"), ("Y", "B")} and edges[LO] == set()
    assert graphs[HI].params["condition"] == HI
    out = save_stim_graph(graphs[HI], tmp_path / "graph_stim_evoked")
    assert out.name == condition_folder_name(HI) == "1000mV_500us_bi500"
    # one condition: a single graph in all/, as before
    one = _detection(nodes, stim.with_pulse_params(np.full(stim.n, 1e6), "uV", [500.0] * stim.n, ["bi500"] * stim.n),
                     {"STG 1": dict.fromkeys(nodes.unit_ids, 2.0)})  # fmt: skip
    assert stim_conditions(one) == []
    assert save_stim_graph(stim_graph_from_detection(one), tmp_path / "single").name == "all"
