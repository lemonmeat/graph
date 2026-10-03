import numpy as np
import pytest

from meagraph import SpikeTrains


def _trains():
    return SpikeTrains.from_dict({"47": [0.6, 0.7, 0.9], "12": [], "33": [1.4]}, t_start_s=0.5, t_stop_s=1.5)


def test_counts_rates_and_order():
    st = _trains()
    assert st.unit_ids == ("47", "12", "33")
    assert st.channel_ids == st.unit_ids
    np.testing.assert_array_equal(st.n_spikes(), [3, 0, 1])
    np.testing.assert_allclose(st.rates_hz(), [3.0, 0.0, 1.0])


def test_select_reorders_and_rejects_unknown():
    st = _trains().with_positions(np.arange(9.0).reshape(3, 3))
    sub = st.select(["33", "47"])
    assert sub.unit_ids == ("33", "47")
    np.testing.assert_array_equal(sub.positions_um[0], [6, 7, 8])
    with pytest.raises(KeyError):
        st.select(["99"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(unit_ids=("a", "a"), times_s=([], [])), "unique"),
        (dict(unit_ids=("a",), times_s=([0.9, 0.8],)), "not sorted"),
        (dict(unit_ids=("a",), times_s=([0.1],)), "outside"),
        (dict(unit_ids=("a",), times_s=([], []),), "same length"),
    ],
)
def test_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SpikeTrains(t_start_s=0.5, t_stop_s=1.5, **kwargs)


def test_sorting_roundtrip_is_exact_to_a_sample():
    st = _trains()
    back = SpikeTrains.from_sorting(st.to_sorting(10_000.0), st.t_start_s, st.t_stop_s)
    assert back.unit_ids == st.unit_ids
    for a, b in zip(back.times_s, st.times_s):
        np.testing.assert_allclose(a, b, atol=0.5e-4)
