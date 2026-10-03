import h5py
import numpy as np
import pytest
import spikeinterface as si

from meagraph.io import McsH5Recording
from meagraph.io._mcs_layout import first_timestamp_s


def test_values_follow_row_index_adzero_and_gains(mcs_file):
    path, truth = mcs_file
    rec = McsH5Recording(path)
    uv = rec.get_traces(return_in_uV=True)
    np.testing.assert_allclose(uv, truth["uv"], rtol=1e-6, atol=1e-6)


def test_channel_ids_are_electrode_labels(mcs_file):
    path, truth = mcs_file
    rec = McsH5Recording(path)
    assert list(rec.channel_ids) == truth["labels"]
    assert rec.get_property("mcs_label")[0] == "E-00303 47"


def test_time_base_starts_at_first_timestamp(mcs_file):
    path, truth = mcs_file
    rec = McsH5Recording(path)
    assert rec.get_sampling_frequency() == truth["fs"]
    assert rec.get_start_time() == pytest.approx(truth["t_start_s"])
    assert rec.get_times()[0] == pytest.approx(0.5)
    assert rec.get_end_time() == pytest.approx(truth["t_stop_s"] - 1 / truth["fs"])


def test_raw_is_found_by_lineage_not_index(mcs_file):
    # The fixture stores the filter stream as Stream_0 and raw as Stream_1.
    path, truth = mcs_file
    np.testing.assert_allclose(McsH5Recording(path).get_traces(return_in_uV=True), truth["uv"], rtol=1e-6)
    for label in ("Filter (1)", "filter", "Filter (1);Filter; Filter Data1"):
        filt = McsH5Recording(path, stream=label).get_traces(return_in_uV=True)
        np.testing.assert_allclose(filt, truth["filtered_uv"], rtol=1e-6)


def test_stream_index_is_rejected(mcs_file):
    with pytest.raises(TypeError):
        McsH5Recording(mcs_file[0], stream=0)


def test_unknown_stream_lists_available(mcs_file):
    with pytest.raises(KeyError, match="Filter"):
        McsH5Recording(mcs_file[0], stream="Filter (9)")


def test_channel_subset_keeps_requested_order(mcs_file):
    path, truth = mcs_file
    rec = McsH5Recording(path)
    sub = rec.get_traces(channel_ids=["15", "47"], start_frame=10, end_frame=20, return_in_uV=True)
    np.testing.assert_allclose(sub, truth["uv"][10:20, [3, 0]], rtol=1e-6)


def test_single_channel_path_matches_full_read(make_mcs_file):
    labels = [str(k) for k in range(12, 32)]
    path, truth = make_mcs_file(labels=labels)
    rec = McsH5Recording(path)
    one = rec.get_traces(channel_ids=["20"], return_in_uV=True)
    np.testing.assert_allclose(one[:, 0], truth["uv"][:, labels.index("20")], rtol=1e-6)


def test_spikeinterface_roundtrip_keeps_time_and_values(mcs_file):
    rec = McsH5Recording(mcs_file[0])
    rec2 = si.load(rec.to_dict(include_annotations=True, include_properties=True))
    assert rec2.get_start_time() == rec.get_start_time()
    np.testing.assert_array_equal(rec2.get_traces(), rec.get_traces())


def test_non_volt_units_are_rejected(make_mcs_file):
    path, _ = make_mcs_file(unit="mV")
    with pytest.raises(ValueError, match="units"):
        McsH5Recording(path)


def test_contiguous_timestamp_blocks_are_accepted():
    blocks = np.array([[500_000, 0, 999], [600_000, 1000, 1999]])
    assert first_timestamp_s(blocks, tick_us=100, n_samples=2000) == pytest.approx(0.5)


def test_timestamp_gaps_are_rejected():
    blocks = np.array([[500_000, 0, 999], [700_000, 1000, 1999]])
    with pytest.raises(NotImplementedError):
        first_timestamp_s(blocks, tick_us=100, n_samples=2000)


def test_duplicate_electrode_labels_are_rejected(make_mcs_file):
    path, _ = make_mcs_file(labels=("47", "47"))
    with pytest.raises(ValueError, match="unique"):
        McsH5Recording(path)


def test_file_is_opened_read_only(mcs_file):
    path, _ = mcs_file
    rec = McsH5Recording(path)
    rec.get_traces(start_frame=0, end_frame=5)
    with h5py.File(path, "r") as f:  # still readable by others, nothing written
        assert "Data" in f
