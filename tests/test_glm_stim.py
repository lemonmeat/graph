"""Poisson GLM coupling (NeMoS) and stimulus-evoked site -> electrode connectivity."""

import numpy as np
import pytest
from scipy.stats import binom

from meagraph.connectivity.stim import StimConnectivityConfig, deconvolve, poisson_binomial_sf, stim_connectivity
from meagraph.spiketrains import SpikeTrains


def _driven_pair(duration_s=300.0, seed=0):
    """Unit 0 drives unit 1 at 3 ms (40 % transmission); unit 2 is independent."""
    rng = np.random.default_rng(seed)
    src = np.sort(rng.uniform(0, duration_s, int(3 * duration_s)))
    keep = rng.random(src.size) < 0.4
    driven = src[keep] + 3e-3 + rng.normal(0, 2e-4, keep.sum())
    tgt = np.sort(np.concatenate([driven, rng.uniform(0, duration_s, int(2 * duration_s))]))
    other = np.sort(rng.uniform(0, duration_s, int(2 * duration_s)))
    return SpikeTrains(("a", "b", "c"), (src, tgt[tgt < duration_s], other), 0.0, duration_s)


def test_glm_finds_the_edge_and_its_delay():
    pytest.importorskip("nemos")
    from meagraph.connectivity import estimate

    res = estimate(_driven_pair(), "glm", {"min_spikes": 50})
    assert res.significant[0, 1] and res.weights[0, 1] > 0
    assert 2.0 <= res.delays_ms[0, 1] <= 4.0
    assert not res.significant[1, 0]  # the reverse direction is not inferred
    assert not res.significant[0, 2] and not res.significant[2, 1] and not res.significant[2, 0]
    assert np.nanmax(res.extra["max_gradient"]) < 1e-3  # the fit converged
    assert res.extra["filters"].shape == (3, 3, 25)


def test_network_covariate_is_causal():
    """A centred kernel let a source's later spikes into the covariate and made independent units
    look inhibitory; the covariate at bin t must use only bins before t."""
    from meagraph.connectivity.glm import causal_smooth

    x = np.zeros(200)
    x[100] = 1.0
    y = causal_smooth(x, 5.0)
    assert np.all(y[:101] < 1e-12) and y[101] > 0 and y.sum() == pytest.approx(1.0)


def test_poisson_binomial_matches_binomial():
    p = np.full(12, 0.1)
    for k in (0, 1, 4, 12):
        assert poisson_binomial_sf(k, p) == pytest.approx(binom.sf(k - 1, 12, 0.1))


def test_deconvolution_weakens_the_indirect_path():
    r = np.zeros((3, 3))
    r[0, 1], r[1, 2], r[0, 2] = 0.5, 0.5, 0.25  # 0 -> 2 is the product of the chain
    a = deconvolve(r)
    assert a[0, 2] < 0.5 * a[0, 1]


def _stim_trains(seed=0):
    """Site "s" pulsed every 2 s; "r" fires 15 ms after 80 % of pulses; "q" is background only."""
    rng = np.random.default_rng(seed)
    pulses = np.arange(1.0, 61.0, 2.0)
    hit = rng.random(pulses.size) < 0.8
    resp = pulses[hit] + 0.015 + rng.normal(0, 3e-4, hit.sum())
    bg = lambda: rng.uniform(0, 62, 60)  # noqa: E731  (1 Hz)
    trains = SpikeTrains(("s", "r", "q"), (np.sort(bg()), np.sort(np.concatenate([resp, bg()])), np.sort(bg())), 0.0, 62.0)
    return trains, pulses


def test_stim_connectivity_finds_the_evoked_response():
    trains, pulses = _stim_trains()
    floor = np.full((1, 3), 10.0)
    res = stim_connectivity(trains, pulses, ["s"] * pulses.size, floor, readouts=["r", "q"])
    assert res.significant[0, 1] and not res.significant[0, 2]
    assert 14.0 <= res.delays_ms[0, 1] <= 16.5
    assert res.extra["jitter_ms"][0, 1] < 1.0
    assert not res.extra["direct"][0, 1]  # 15 ms is not a direct (<= 5 ms) latency
    assert not res.tested[1:].any()  # only the stimulated site has outgoing edges
    assert np.isnan(res.p_values[0, 0])


def test_stim_connectivity_respects_the_blanking_floor():
    trains, pulses = _stim_trains()
    floor = np.full((1, 3), 20.0)  # the response at 15 ms is inside blanking
    res = stim_connectivity(trains, pulses, ["s"] * pulses.size, floor, readouts=["r"], config=StimConnectivityConfig())
    assert not res.significant[0, 1]
