"""Evoked-response analysis on synthetic pulses with known responses."""

import numpy as np
import pytest

from meagraph import SpikeTrains
from meagraph.detect.threshold import ChannelQC, DetectionConfig, DetectionResult
from meagraph.evoked import EvokedConfig, blanked_until_ms, decode, evoked_from_detection, evoked_responses, sham_times, wilson
from meagraph.io.mcs_events import StimEvents

FAST = EvokedConfig(n_permutations=20, n_pair_permutations=20, n_repeats=2)


def _session(seed=0, n_per_site=20):
    """Sites X and Y drive channels A and B (3 spikes at ~20 ms and ~40 ms); Z drives nothing."""
    rng = np.random.default_rng(seed)
    sites = rng.permutation(np.repeat(["X", "Y", "Z"], n_per_site))
    onsets = 10.0 + np.cumsum(rng.uniform(3, 5, sites.size))
    t_stop = onsets[-1] + 10
    spikes = {c: list(rng.uniform(0, t_stop, int(0.3 * t_stop))) for c in "ABCD"}
    for t, s in zip(onsets, sites):
        if s == "X":
            spikes["A"] += list(t + 0.020 + rng.normal(0, 0.002, 3))
        elif s == "Y":
            spikes["B"] += list(t + 0.040 + rng.normal(0, 0.002, 3))
    trains = SpikeTrains.from_dict(spikes, 0.0, t_stop)
    stim = StimEvents("STG 1", "Single Pulse", onsets, onsets + 0.001).with_pulse_sites(sites)
    return trains, stim, onsets, sites


def test_responses_and_decoding_find_the_driven_channels():
    trains, stim, _, _ = _session()
    res = evoked_responses(trains, [stim], np.zeros((0, 2)), config=FAST)
    sig = {(r["site"], r["channel"]) for r in res.response_table if r["q_value"] < 0.05}
    assert sig == {("X", "A"), ("Y", "B")}
    lat = {(r["site"], r["channel"]): r["median_first_ms"] for r in res.response_table}
    assert 15 < lat["X", "A"] < 25 and 35 < lat["Y", "B"] < 45
    d = res.decoding["sites"]
    assert d["recall"]["X"] > 0.9 and d["recall"]["Y"] > 0.9 and d["p_value"] < 0.1
    pairs = {(r["a"], r["b"]): r["balanced_accuracy"] for r in res.decoding["site_pairs"]}
    assert pairs["X", "Y"] > 0.9 and pairs["X", "Z"] > 0.9
    # Z evokes nothing, so Z and sham trials are confused: together they score at chance
    r = res.decoding["sites_and_sham"]["recall"]
    assert r["Z"] + r["sham"] < 1.3, r
    assert {r["channel"] for r in res.channel_table if r["q_value"] < 0.05} == {"A", "B"}


def test_shuffled_labels_decode_at_chance():
    trains, stim, onsets, sites = _session(seed=1)
    shuffled = stim.with_pulse_sites(np.random.default_rng(5).permutation(sites))
    res = evoked_responses(trains, [shuffled], np.zeros((0, 2)), config=FAST)
    assert abs(res.decoding["sites"]["balanced_accuracy"] - 1 / 3) < 0.15


def test_evoked_bursts_and_states():
    trains, stim, onsets, sites = _session(seed=2)
    x = onsets[sites == "X"]
    bursts = np.column_stack([x + 0.010, x + 0.100])  # every X pulse evokes a burst at 10 ms
    z = onsets[sites == "Z"][:3]
    bursts = np.vstack([bursts, np.column_stack([z - 0.05, z + 0.05])])  # three Z pulses land in a burst
    bursts = bursts[np.argsort(bursts[:, 0])]
    res = evoked_responses(trains, [stim], bursts, config=FAST)
    rows = {(r["site"], r["state"]): r for r in res.site_table}
    assert rows["X", "all"]["n_evoked"] == rows["X", "all"]["n_pulses"] and rows["X", "all"]["p_value"] < 1e-6
    assert rows["Y", "all"]["n_evoked"] == 0
    assert rows["Z", "in_burst"]["n_pulses"] == 3
    assert np.allclose(res.pulses.burst_latency_ms[res.pulses.label == "X"], 10.0)
    # A burst starting before the pulse is not evoked by it; the window starts at burst_from_ms.
    late = evoked_responses(trains, [stim], bursts, config=FAST.model_copy(update={"burst_from_ms": 12.0}))
    assert {r["site"]: r["n_evoked"] for r in late.site_table if r["state"] == "all"}["X"] == 0


def test_sham_times_avoid_pulses():
    cfg = EvokedConfig()
    on = np.array([10.0, 20.0])
    s = sham_times(on, 0.0, 30.0, cfg)
    assert s.size and not np.any((s[:, None] - on[None, :] > -0.15) & (s[:, None] - on[None, :] < 1.0))


def test_blanking_window_is_the_same_for_every_condition():
    """The stimulated electrode is blanked longest; its window starts after that for every pulse,
    so the decoder cannot identify a site from blanking."""
    trains, stim, onsets, sites = _session(seed=3)
    stim = stim.with_pulse_sites(np.where(sites == "X", "A", sites))  # site X is electrode A
    cfg = DetectionConfig()
    det = DetectionResult(
        trains=trains, amplitudes_uv={c: np.full(t.size, -10.0) for c, t in trains.as_dict().items()},
        waveforms_uv={c: np.zeros((t.size, 4)) for c, t in trains.as_dict().items()},
        noise_uv=dict.fromkeys(trains.unit_ids, 1.0),
        qc=ChannelQC(trains.unit_ids, np.full(4, 100), np.full(4, 10), np.full(4, 1e-9), np.ones(4, bool)),
        excluded={}, recovery_ms={"STG 1": {"A": 3.0, "B": 8.0, "C": float("nan"), "D": 0.5}},
        config=cfg, stim=(stim,), analog_stream=None,
    )  # fmt: skip
    np.testing.assert_allclose(blanked_until_ms(det, ["A", "B", "C", "D"]), [52.0, 10.0, 52.0, 3.0])
    res = evoked_from_detection(det, FAST)
    np.testing.assert_allclose(res.start_ms, [52.0, 12.0, 52.0, 12.0])
    # X's response on A (at 20 ms) is now inside the blanked part, for pulses and shams alike
    assert not any(r["q_value"] < 0.05 for r in res.response_table if r["channel"] == "A")


def test_spikes_just_before_pulses_are_dropped():
    trains, stim, onsets, _ = _session(seed=4)
    d = trains.as_dict()
    d["C"] = np.sort(np.concatenate([d["C"], onsets - 0.0012]))
    trains = SpikeTrains.from_dict(d, trains.t_start_s, trains.t_stop_s)
    det = DetectionResult(
        trains=trains, amplitudes_uv={c: np.full(t.size, -10.0) for c, t in trains.as_dict().items()},
        waveforms_uv={c: np.zeros((t.size, 4)) for c, t in trains.as_dict().items()},
        noise_uv=dict.fromkeys(trains.unit_ids, 1.0),
        qc=ChannelQC(trains.unit_ids, np.full(4, 100), np.full(4, 10), np.full(4, 1e-9), np.ones(4, bool)),
        excluded={}, recovery_ms={}, config=DetectionConfig(), stim=(stim,), analog_stream=None,
    )  # fmt: skip
    res = evoked_from_detection(det, FAST)
    assert res.n_dropped_pre >= onsets.size


def test_decode_skips_too_few_trials():
    out = decode(np.zeros((3, 2, 1)), np.array(["a", "a", "b"], dtype=object), FAST, np.random.default_rng(0))
    assert "skipped" in out


def test_wilson():
    lo, hi = wilson(0, 23)
    assert lo == 0 and hi == pytest.approx(0.143, abs=0.002)
