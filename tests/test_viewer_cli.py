import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from meagraph.cli import main  # noqa: E402
from meagraph.detect import detect_spikes, save_detection  # noqa: E402
from meagraph.detect.store import default_detection_folder, find_detection  # noqa: E402
from meagraph.io import load_session  # noqa: E402
from meagraph.viewer import Viewer, load_viewer_data, nearest_on_screen  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def recording_with_spikes(make_mcs_file):
    path, _ = make_mcs_file(labels=("47", "12", "33", "15"), n_samples=20_000, stim_onsets_us=[600_000, 1_100_000, 1_600_000])
    session = load_session(path)
    result = detect_spikes(session.recording, session.stim)
    save_detection(result, default_detection_folder(path))
    return path


def test_viewer_finds_detection_results(recording_with_spikes):
    data = load_viewer_data(recording_with_spikes)
    assert data.electrodes == ("47", "12", "33")  # reference 15 dropped
    assert data.spike_source == "meagraph detect (5 sigma)"
    assert data.streams == ("Data Acquisition (1)", "Filter (1)")
    np.testing.assert_allclose(data.stim_onsets["STG 1"], [0.6, 1.1, 1.6])


def test_viewer_renders_and_navigates(recording_with_spikes, tmp_path):
    data = load_viewer_data(recording_with_spikes)
    v = Viewer(data, electrode="12", win=0.5)
    assert v.t0 == pytest.approx(data.t_start)
    v._set(elec="33", stream="Filter (1)", bandpass=True)
    assert v.map_label.get_text().startswith("electrode 33")

    class Key:
        key = "right"

    v._on_key(Key())
    assert v.t0 == pytest.approx(data.t_start + 0.25)
    v.fig.savefig(tmp_path / "view.png")
    assert (tmp_path / "view.png").stat().st_size > 10_000


def test_click_on_map_selects_nearest_electrode(recording_with_spikes):
    data = load_viewer_data(recording_with_spikes)
    v = Viewer(data)
    v.fig.canvas.draw()
    from mpl_toolkits.mplot3d import proj3d

    k = data.electrodes.index("33")
    xs, ys, _ = proj3d.proj_transform(*v.cube_xyz[[k]].T, v.ax_map.get_proj())
    x_px, y_px = v.ax_map.transData.transform([[xs[0], ys[0]]])[0]
    assert nearest_on_screen(v.ax_map, v.cube_xyz, x_px + 3, y_px - 3) == k
    assert nearest_on_screen(v.ax_map, v.cube_xyz, -500, -500) is None


def test_viewer_runs_from_results_without_the_recording(recording_with_spikes, tmp_path):
    with_h5 = load_viewer_data(recording_with_spikes)
    results = default_detection_folder(recording_with_spikes).parent
    # the .h5 still exists: a results folder finds it, and traces are shown
    assert load_viewer_data(results).has_traces
    recording_with_spikes.unlink()
    for path in (results, results / "detect", recording_with_spikes):
        data = load_viewer_data(path)
        assert not data.has_traces and data.recording_path is None
        assert data.electrodes == with_h5.electrodes
        np.testing.assert_array_equal(data.grid, with_h5.grid)
        assert data.active == with_h5.active and data.spike_source == with_h5.spike_source
        assert (data.t_start, data.t_stop) == pytest.approx((with_h5.t_start, with_h5.t_stop))
        np.testing.assert_allclose(data.stim_onsets["STG 1"], [0.6, 1.1, 1.6])
        assert data.spikes.keys() == with_h5.spikes.keys()
    v = Viewer(data, electrode="12", win=0.5)
    v._set(elec="33")
    assert "no trace" in v.ax_trace.get_title(loc="left")
    assert main(["view", str(results), "--save", str(tmp_path / "v.png")]) == 0
    assert (tmp_path / "v.png").stat().st_size > 10_000
    with pytest.raises(FileNotFoundError):
        load_viewer_data(tmp_path / "nothing_here")


def test_viewer_falls_back_to_no_spikes(make_mcs_file):
    path, _ = make_mcs_file()
    data = load_viewer_data(path)
    assert data.spikes == {} and "meagraph detect" in data.spike_source
    Viewer(data)  # still renders


def test_cli_detect_audit_view(make_mcs_file, capsys, tmp_path):
    path, _ = make_mcs_file(n_samples=20_000, stim_onsets_us=[600_000, 1_100_000, 1_600_000])
    assert main(["detect", str(path), "--stim-site", "47"]) == 0
    out = capsys.readouterr().out
    assert "excluded: {'47'" in out
    assert find_detection(path) == default_detection_folder(path)
    assert main(["audit", str(path)]) == 0
    assert "STG 1: 3 pulses" in capsys.readouterr().out
    assert main(["view", str(path), "--save", str(tmp_path / "v.png")]) == 0
    assert (tmp_path / "v.png").exists()


def test_cli_skips_sidecars(make_mcs_file, capsys, tmp_path):
    path, _ = make_mcs_file()
    sidecar = tmp_path / "x.spikes.h5"
    sidecar.write_bytes(b"")
    assert main(["info", str(path), str(sidecar)]) == 0
    assert "skipping x.spikes.h5" in capsys.readouterr().out
