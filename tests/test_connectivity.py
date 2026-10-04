import json

import numpy as np
import pytest
import quantities as pq
import neo
from elephant.spike_train_correlation import spike_time_tiling_coefficient as elephant_sttc

from meagraph import SpikeTrains
from meagraph.benchmark import roc_auc, run_benchmark, scenarios, score
from meagraph.connectivity import ESTIMATORS, cross_correlograms, estimate, interval_jitter, load_result, save_result, to_networkx, window_counts
from meagraph.connectivity.pipeline import burst_controlled, positions_for
from meagraph.connectivity.stats import fdr_mask, sttc
from meagraph.detect.bursts import BurstConfig, network_bursts, remove_periods
from meagraph.synth import NetworkConfig, simulate_network

FAST = {"n_surrogates": 200}


# -- building blocks ---------------------------------------------------------------------- #
def test_correlogram_lag_convention_matches_window_counts():
    a = np.arange(0.1, 50, 0.2)
    b = a + 0.0022  # target 2.2 ms after source
    trains = SpikeTrains.from_dict({"A": a, "B": b}, 0.0, 51.0)
    ccg, edges = cross_correlograms(trains, 0.5, 30.0)
    k = int(np.argmax(ccg[0, 1]))
    assert (edges[k], edges[k + 1]) == (2.0, 2.5)
    assert edges[int(np.argmax(ccg[1, 0]))] == -2.5
    inside = (edges[:-1] >= 1.0) & (edges[1:] <= 4.0)
    assert ccg[0, 1, inside].sum() == window_counts(a, b, 0.001, 0.004) == a.size


def test_interval_jitter_keeps_each_spike_in_its_window():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.5, 20.0, 400))
    s = interval_jitter(t, 0.01, 0.5, 20.0, rng, 50)
    assert s.shape == (50, 400)
    window = lambda x: np.floor((x - 0.5) / 0.01)  # noqa: E731
    assert np.array_equal(np.sort(window(s), axis=1), np.tile(np.sort(window(t)), (50, 1)))
    assert np.all(np.diff(s, axis=1) >= 0) and s.min() >= 0.5 and s.max() <= 20.0
    offsets = (s - 0.5) / 0.01 - window(s)  # position within the window: uniform
    assert abs(offsets.mean() - 0.5) < 0.02


def _neo(x, t0, t1):
    return neo.SpikeTrain(x * pq.s, t_start=t0 * pq.s, t_stop=t1 * pq.s)


def test_sttc_matches_elephant_where_its_tolerance_is_negligible():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = (np.sort(rng.uniform(0, 0.5, rng.integers(2, 30))) for _ in range(2))
        assert sttc(a, b, 0.005, 0.0, 0.5) == pytest.approx(elephant_sttc(_neo(a, 0, 0.5), _neo(b, 0, 0.5), dt=0.005 * pq.s), abs=1e-12)


def test_sttc_uses_the_exact_coincidence_window():
    # Elephant's np.isclose tolerance would call these coincident at t = 1000 s (D13); we do not.
    assert sttc(np.array([1000.0]), np.array([1000.0051]), 0.005, 0.0, 2000.0) <= 0
    assert sttc(np.array([1000.0]), np.array([1000.0049]), 0.005, 0.0, 2000.0) > 0.99


def test_fdr_ignores_untested_entries():
    p = np.array([[np.nan, 0.001], [0.5, np.nan]])
    np.testing.assert_array_equal(fdr_mask(p, 0.05), [[False, True], [False, False]])


def test_roc_auc():
    assert roc_auc(np.array([1, 2, 3, 4.0]), np.array([False, False, True, True])) == 1.0
    assert roc_auc(np.array([1, 1, 1, 1.0]), np.array([False, True, False, True])) == 0.5


# -- simulator ------------------------------------------------------------------------------ #
def test_simulator_ground_truth_and_reproducibility():
    cfg = NetworkConfig(duration_s=200, n_units=10, connection_prob=0.2, weight=(0.2, 0.2), seed=3)
    a, b = simulate_network(cfg), simulate_network(cfg)
    assert all(np.array_equal(x, y) for x, y in zip(a.trains.times_s, b.trains.times_s))
    assert a.connected.sum() > 0 and np.all(np.isnan(a.delays_ms[~a.connected]))
    assert a.trains.positions_um.shape == (10, 3)
    # A connection shows up as excess target spikes at its delay.
    i, j = map(int, np.argwhere(a.connected)[0])
    d = a.delays_ms[i, j] / 1e3
    src, tgt = a.trains.times_s[i], a.trains.times_s[j]
    excess = window_counts(src, tgt, d - 0.001, d + 0.001) - window_counts(src, tgt, -d - 0.001, -d + 0.001)
    assert excess > 0.1 * src.size


def test_simulator_rejects_explosive_networks():
    with pytest.raises(ValueError, match="excitable"):
        simulate_network(NetworkConfig(n_units=10, connection_prob=1.0, weight=(0.5, 0.5)))


def test_simulated_bursts_are_detected():
    net = simulate_network(NetworkConfig(duration_s=120, n_units=16, connection_prob=0, burst_rate_hz=0.2, seed=2))
    found = network_bursts(net.trains, BurstConfig())
    centres = net.bursts.mean(axis=1)
    hit = [np.any((found[:, 0] - 0.05 <= c) & (c <= found[:, 1] + 0.05)) for c in centres]
    assert np.mean(hit) >= 0.8
    assert len(found) <= 2 * len(net.bursts) + 2


def test_remove_periods():
    trains = SpikeTrains.from_dict({"a": [1.0, 2.0, 3.0]}, 0.0, 4.0)
    out = remove_periods(trains, np.array([[1.5, 2.5]]))
    np.testing.assert_array_equal(out.times_s[0], [1.0, 3.0])
    assert out.duration_s == trains.duration_s


# -- estimators on ground truth -------------------------------------------------------------- #
@pytest.fixture(scope="module")
def strong_network():
    return simulate_network(NetworkConfig(duration_s=600, n_units=12, connection_prob=0.06, weight=(0.2, 0.2), seed=5))


@pytest.mark.parametrize("method", ["cch_hollow", "cch_jitter", "sttc"])
def test_estimators_recover_strong_connections(strong_network, method):
    overrides = FAST if "n_surrogates" in ESTIMATORS[method].Config.model_fields else {}
    s = score(estimate(strong_network.trains, method, overrides), strong_network)
    assert s["recall"] >= 0.8, s
    assert s["fp"] - s["fp_indirect"] <= 1, s  # false alarms beyond indirect correlations are rare
    if method != "sttc":
        assert s["delay_error_ms"] < 0.5


def test_false_positives_in_dense_networks_are_indirect():
    """Pairwise methods report chains and common input as edges: the functional-vs-direct limit."""
    net = simulate_network(NetworkConfig(duration_s=600, n_units=10, connection_prob=0.15, weight=(0.15, 0.15), seed=5))
    s = score(estimate(net.trains, "cch_jitter", FAST), net)
    assert s["fp"] > 0 and s["fp_indirect"] == s["fp"]


@pytest.mark.parametrize("method", ["cch_hollow", "cch_jitter", "sttc"])
def test_no_false_alarms_on_a_null_network(method):
    net = simulate_network(NetworkConfig(duration_s=300, n_units=10, connection_prob=0.0, seed=7))
    overrides = FAST if "n_surrogates" in ESTIMATORS[method].Config.model_fields else {}
    assert score(estimate(net.trains, method, overrides), net)["fp"] <= 1


def test_null_p_values_are_calibrated_and_not_floored():
    """Under no connections, about 5 % of pairs reach p < 0.05; and p-values are not limited to
    the 1/(N+1) Monte-Carlo floor, which would make BH across many pairs powerless (D17)."""
    net = simulate_network(NetworkConfig(duration_s=600, n_units=16, connection_prob=0.0, burst_rate_hz=0.24, seed=11))
    for method, limit in (("cch_jitter", 0.08), ("sttc", 0.10)):
        r = estimate(net.trains, method, FAST)
        p = r.p_values[r.tested]
        assert np.mean(p < 0.05) <= limit, method
    strong = simulate_network(NetworkConfig(duration_s=600, n_units=8, connection_prob=0.1, weight=(0.3, 0.3), seed=1))
    r = estimate(strong.trains, "cch_jitter", FAST)
    assert np.nanmin(r.p_values) < 1 / 201


def test_burst_control_and_result_storage(tmp_path, strong_network):
    bc = burst_controlled(strong_network.trains, "cch_jitter", FAST)
    assert np.array_equal(bc.robust.significant, bc.all.significant & bc.no_bursts.significant)
    out = save_result(bc.all, tmp_path / "g")
    back = load_result(out)
    np.testing.assert_array_equal(back.significant, bc.all.significant)
    np.testing.assert_allclose(back.weights, bc.all.weights)
    g = to_networkx(back)
    assert g.number_of_edges() == int(bc.all.significant.sum())
    node = next(iter(g.nodes(data=True)))[1]
    assert {"x_um", "y_um", "z_um", "n_spikes"} <= set(node)
    assert json.loads((out / "graph.json").read_text())["directed"] is True


def test_positions_come_from_the_probe():
    pos = positions_for(["47", "12"], "cube4x4x4_E-00303")
    assert pos.shape == (2, 3)
    with pytest.raises(KeyError):
        positions_for(["99"], "cube4x4x4_E-00303")


def test_benchmark_runs_and_scores_every_spike_set():
    rows = run_benchmark(scenarios(durations_s=(120,), weights=(0.15,), bursts=(False,), base=NetworkConfig(n_units=6)),
                         methods=("cch_hollow",), seeds=(0,), progress=None)  # fmt: skip
    assert {r["spikes"] for r in rows} == {"all", "no_bursts", "robust"}
    assert {r["scenario"] for r in rows} == {"no-bursts T=120s w=0.15", "no-bursts null T=600s", "stim null T=600s"}


def test_stimulation_periods_pad_and_merge():
    from meagraph.connectivity.pipeline import stimulation_periods
    from meagraph.io.mcs_events import StimEvents

    stim = StimEvents("STG 1", "Single Pulse", np.array([1.0, 1.1, 5.0]), np.array([1.002, 1.102, 5.002]))
    p = stimulation_periods([stim], pre_ms=1.0, post_ms=200.0)
    np.testing.assert_allclose(p, [[0.999, 1.302], [4.999, 5.202]])
    assert stimulation_periods([]).shape == (0, 2)


def test_cli_graph_on_a_detection_folder(tmp_path, capsys):
    from meagraph.cli import main
    from meagraph.detect import DetectionConfig, DetectionResult, save_detection
    from meagraph.detect.threshold import ChannelQC

    net = simulate_network(NetworkConfig(duration_s=300, n_units=6, connection_prob=0.2, weight=(0.3, 0.3), seed=4))
    ids = net.trains.unit_ids
    det = DetectionResult(
        trains=net.trains, amplitudes_uv={u: np.zeros(t.size, np.float32) for u, t in zip(ids, net.trains.times_s)},
        waveforms_uv={u: np.zeros((t.size, 30), np.float32) for u, t in zip(ids, net.trains.times_s)},
        noise_uv={u: 1.0 for u in ids}, excluded={}, recovery_ms={}, config=DetectionConfig(),
        qc=ChannelQC(ids, np.full(len(ids), 100), np.zeros(len(ids), int), np.zeros(len(ids)), np.ones(len(ids), bool)),
    )  # fmt: skip
    folder = save_detection(det, tmp_path / "results" / "rec" / "detect_default")
    assert main(["graph", str(folder), "--method", "cch_hollow"]) == 0
    out = capsys.readouterr().out
    assert "6 active channels" in out and "stimulation periods (if any) were not excluded" in out
    for sub in ("all", "no_bursts", "robust"):
        assert (tmp_path / "results" / "rec" / "graph_cch_hollow" / sub / "graph.graphml").exists()
