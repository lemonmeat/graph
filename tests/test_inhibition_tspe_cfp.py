"""Phase 4.5: inhibition in the simulator, TSPE and CFP, and signed scoring (DECISIONS.md D22)."""

import networkx as nx
import numpy as np
import pytest

from meagraph.benchmark import VARIANTS, score
from meagraph.connectivity import cross_correlograms, estimate, load_result, save_result
from meagraph.connectivity.pipeline import burst_controlled
from meagraph.synth import NetworkConfig, simulate_network

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def inhibitory_network():
    """High rates and long duration, so inhibitory dips are measurable."""
    return simulate_network(NetworkConfig(duration_s=1800, n_units=12, inhibitory_fraction=0.3, rate_hz=(3.0, 6.0),
                                          weight=(0.12, 0.12), inhibition=(0.8, 0.8), seed=11))  # fmt: skip


def test_inhibitory_connections_suppress_their_targets(inhibitory_network):
    net = inhibitory_network
    assert net.inhibitory.any() and not (net.inhibitory & net.excitatory).any()
    # Dale's principle: a unit's connections are all of one kind
    assert not (net.inhibitory.any(axis=1) & net.excitatory.any(axis=1)).any()
    ccg, edges = cross_correlograms(net.trains, 1.0, 30.0)
    c = (edges[:-1] + edges[1:]) / 2
    for i, j in zip(*np.nonzero(net.inhibitory)):
        d = net.delays_ms[i, j]
        ratio = ccg[i, j, (c > d + 0.5) & (c < d + 9.5)].mean() / ccg[i, j, c < -5].mean()
        assert ratio < 0.5, (i, j, ratio)  # 80 % of target spikes deleted in the window


def test_excitatory_only_networks_have_no_suppression():
    net = simulate_network(NetworkConfig(duration_s=60, seed=1))
    assert not net.inhibitory.any() and (net.connected == net.excitatory).all()


def test_tspe_orientation_delays_and_inhibition(inhibitory_network):
    net = inhibitory_network
    r = estimate(net.trains, "tspe", {"n_surrogates": 100})
    assert r.signed
    exc = r.significant & (r.weights > 0)
    assert (exc & net.excitatory).sum() >= 0.9 * net.excitatory.sum()  # (source, target), not Elephant's transpose
    hit = exc & net.excitatory
    assert np.median(np.abs(r.delays_ms[hit] - net.delays_ms[hit])) < 0.6
    assert (r.significant & (r.weights < 0) & net.inhibitory).sum() >= 1
    s = score(r, net)
    assert s["inh_tp"] >= 1 and s["inh_fp_reverse"] <= s["inh_fp"]
    # The reverse-inhibition rule only ever removes inhibitory calls
    kept = VARIANTS["tspe"]["tspe_noreverse"](r)
    assert not (kept & ~r.significant).any() and (kept | ~(r.significant & (r.weights > 0))).all()


def test_cfp_finds_peaks_but_its_width_rule_rejects_sharp_ones():
    net = simulate_network(NetworkConfig(duration_s=600, n_units=8, connection_prob=0.2, weight=(0.2, 0.2), seed=3))
    r = estimate(net.trains, "cfp", {"n_surrogates": 100})
    test = r.extra["test_significant"]
    assert (test & net.excitatory).sum() >= 0.9 * net.excitatory.sum()
    fit = r.extra["fit_m_t_w_offset"]
    hit = test & net.excitatory
    assert np.median(np.abs(fit[hit][:, 1] - net.delays_ms[hit])) < 1.0
    # Monosynaptic peaks are ~1 ms wide, below the published 5 ms minimum (D22)
    assert not (r.significant & net.excitatory).any()
    narrow = VARIANTS["cfp"]["cfp_narrow"](r)
    assert (narrow & net.excitatory).sum() == (test & net.excitatory).sum()


def test_robust_requires_the_same_sign(inhibitory_network):
    bc = burst_controlled(inhibitory_network.trains, "tspe", {"n_surrogates": 50}, bursts=np.zeros((0, 2)))
    with np.errstate(invalid="ignore"):
        assert not (bc.robust.significant & (np.sign(bc.all.weights) != np.sign(bc.no_bursts.weights))).any()


def test_signed_graphs_record_edge_type(tmp_path, inhibitory_network):
    r = estimate(inhibitory_network.trains, "tspe", {"n_surrogates": 50})
    back = load_result(save_result(r, tmp_path / "g"))
    assert back.signed
    types = {d["type"] for _, _, d in nx.read_graphml(tmp_path / "g" / "graph.graphml").edges(data=True)}
    assert types <= {"excitatory", "inhibitory"} and "excitatory" in types


# -- cch_gauss2 (Kumar et al. 2026, D23) ----------------------------------------------------- #
def test_gauss2_fit_recovers_parameters_and_the_paper_weight():
    from meagraph.connectivity.gauss2 import fit_gauss2, gauss2

    x = np.arange(-99, 100, 2.0)
    truth = [30, 4.0, 3.0, 10, -5.0, 40.0]
    p = fit_gauss2(gauss2(x, *truth) + np.random.default_rng(0).normal(0, 0.3, x.size), x, 2.0)
    np.testing.assert_allclose(p, truth, rtol=0.05, atol=0.3)
    assert (p[0] + p[3]) / (p[2] + p[5]) == pytest.approx(40 / 43, rel=0.02)


def test_gauss2_net_direction_and_storage(tmp_path):
    net = simulate_network(NetworkConfig(duration_s=600, n_units=8, connection_prob=0.2, weight=(0.2, 0.2), seed=3))
    r = estimate(net.trains, "cch_gauss2")
    assert np.isnan(r.p_values).all() and r.tested.sum() == 8 * 7  # no test, but every pair was fitted
    hit = r.significant & net.excitatory
    assert hit.sum() >= 0.9 * net.excitatory.sum()  # the weight goes to the source -> target direction
    assert not (r.significant & r.significant.T).any()  # one net direction per pair
    assert score(r, net)["auc"] > 0.9  # ranked by weight, since there are no p-values
    back = load_result(save_result(r, tmp_path / "g"))
    np.testing.assert_array_equal(back.tested, r.tested)
