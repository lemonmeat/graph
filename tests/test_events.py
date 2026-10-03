import numpy as np
import pytest

from meagraph.io import load_session, read_event_entities, read_mcs_spikes, read_stim_events
from meagraph.probe import UnverifiedGeometryWarning

pytestmark = pytest.mark.filterwarnings("ignore::meagraph.probe.UnverifiedGeometryWarning")


def test_entities_are_decoded_by_event_id_not_row(make_mcs_file):
    # InfoEvent rows reversed: row 0 now describes EventID 8. Decoding by row would mislabel.
    path, _ = make_mcs_file(stim_onsets_us=[510_000, 560_000], info_event_order=list(range(8))[::-1])
    ents = {e.event_id: e for e in read_event_entities(path)}
    assert set(ents) == {1, 2}
    assert ents[1].label.endswith("STG 1 Single Pulse Start")
    assert ents[2].label.endswith("STG 1 Single Pulse Stop")
    np.testing.assert_allclose(ents[1].timestamps_s, [0.51, 0.56])


def test_stim_events_pair_start_and_stop(mcs_file):
    (stim,) = read_stim_events(mcs_file[0])
    assert (stim.source, stim.kind, stim.n, stim.site) == ("STG 1", "Single Pulse", 3, None)
    np.testing.assert_allclose(stim.onsets_s, [0.51, 0.56, 0.61])
    np.testing.assert_allclose(stim.durations_s, 0.003)


def test_no_stimulation_gives_empty_list(make_mcs_file):
    path, _ = make_mcs_file()
    assert read_stim_events(path) == []


def test_unpairable_stops_are_dropped_with_warning(make_mcs_file):
    # Offsets after the next onset cannot belong to their own pulse.
    path, _ = make_mcs_file(stim_onsets_us=[510_000, 511_000], stim_offset_delay_us=3000)
    with pytest.warns(UserWarning, match="could not be paired"):
        (stim,) = read_stim_events(path)
    assert stim.offsets_s is None


def test_detector_spikes_by_electrode(mcs_file):
    trains = read_mcs_spikes(mcs_file[0], "Spike Detector")
    assert trains.as_dict().keys() == {"47", "12"}
    np.testing.assert_allclose(trains.as_dict()["47"], [0.52, 0.53, 0.54])
    assert trains.t_start_s == pytest.approx(0.5)


def test_sorter_units_decoded_by_segment_id(mcs_file):
    # Sorter SegmentIDs run backwards relative to table rows in the fixture.
    units = read_mcs_spikes(mcs_file[0], "Spike Sorter", merge_units=False).as_dict()
    np.testing.assert_allclose(units["47#1"], [0.52, 0.54])
    np.testing.assert_allclose(units["47#2"], [0.53])
    merged = read_mcs_spikes(mcs_file[0], "Spike Sorter").as_dict()
    np.testing.assert_allclose(merged["47"], [0.52, 0.53, 0.54])


def test_spikes_before_analog_data_are_dropped(make_mcs_file):
    path, _ = make_mcs_file(detector_spikes_us={"47": [400_000, 520_000]})
    with pytest.warns(UserWarning, match="outside the analog data"):
        trains = read_mcs_spikes(path)
    np.testing.assert_allclose(trains.as_dict()["47"], [0.52])


def test_session_assigns_stim_site(mcs_file):
    session = load_session(mcs_file[0], stim_site="47")
    assert session.stim[0].site == "47"
    assert session.excluded_channel_ids == ("15",)
    assert session.recording.get_start_time() == pytest.approx(0.5)


def test_session_rejects_unknown_site_or_source(mcs_file):
    with pytest.raises(ValueError, match="not a channel"):
        load_session(mcs_file[0], stim_site="99")
    with pytest.raises(KeyError, match="STG 2"):
        load_session(mcs_file[0], stim_site={"STG 2": "47"})


def test_session_warns_about_placeholder_geometry(mcs_file):
    with pytest.warns(UnverifiedGeometryWarning):
        load_session(mcs_file[0])
