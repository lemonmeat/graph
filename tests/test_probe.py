import warnings

import numpy as np
import pytest
import spikeinterface as si
from pydantic import ValidationError

from meagraph.io import McsH5Recording
from meagraph.probe import (
    GridContact,
    ProbeSpec,
    UnverifiedGeometryWarning,
    attach_probe,
    available_probe_specs,
    channel_distances_um,
    channel_positions_um,
    load_probe_spec,
)

pytestmark = pytest.mark.filterwarnings("ignore::meagraph.probe.UnverifiedGeometryWarning")
CUBE = "cube4x4x4_E-00303"


def _spec(**overrides):
    base = dict(
        name="t", ndim=3, pitch_row_um=10.0, pitch_col_um=20.0, layer_spacing_um=30.0, contact_radius_um=5.0,
        contacts=[GridContact(label="47", row=1, col=1, layer=1), GridContact(label="12", row=1, col=1, layer=2),
                  GridContact(label="33", row=2, col=3, layer=4)],
    )  # fmt: skip
    base.update(overrides)
    return ProbeSpec(**base)


def test_packaged_specs_are_listed():
    assert {CUBE, "mcs60_8x8_200um"} <= set(available_probe_specs())


def test_cube_has_59_contacts_and_5_empty_cells():
    spec = load_probe_spec(CUBE)
    cells = {(c.row, c.col, c.layer) for c in spec.contacts}
    empty = {(r, c, l) for r in range(1, 5) for c in range(1, 5) for l in range(1, 5)} - cells  # noqa: E741
    assert len(spec.contacts) == 59
    assert empty == {(1, 2, 4), (1, 3, 4), (1, 4, 4), (4, 3, 4), (4, 4, 4)}
    assert spec.reference_labels == ("15",)
    assert spec.geometry_verified is False


def test_planar_spec_is_2d_without_corners_or_reference():
    spec = load_probe_spec("mcs60_8x8_200um")
    assert spec.ndim == 2 and len(spec.contacts) == 59
    assert not {"11", "18", "81", "88", "15"} & set(spec.labels)
    pos = dict(zip(spec.labels, spec.positions_um()))
    np.testing.assert_allclose(pos["47"], [600.0, 1200.0])  # column 4, row 7 at 200 um


def test_positions_follow_pitches():
    pos = dict(zip(*[_spec().labels, _spec().positions_um()]))
    np.testing.assert_allclose(pos["47"], [0, 0, 0])
    np.testing.assert_allclose(pos["12"], [0, 0, 30])
    np.testing.assert_allclose(pos["33"], [40, 10, 90])


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(contacts=[GridContact(label="1", row=1, col=1), GridContact(label="2", row=1, col=1)]), "share a grid cell"),
        (dict(contacts=[GridContact(label="1", row=1, col=1), GridContact(label="1", row=1, col=2)]), "duplicate"),
        (dict(layer_spacing_um=None), "layer_spacing_um"),
        (dict(reference_labels=("47",)), "reference"),
        (dict(ndim=2, layer_spacing_um=None), "single layer"),
    ],
)
def test_invalid_specs_are_rejected(overrides, message):
    with pytest.raises(ValidationError, match=message):
        _spec(**overrides)


def test_spec_loads_from_any_yaml_path(tmp_path):
    (tmp_path / "map.csv").write_text("label,row,col,layer\n47,1,1,1\n12,2,1,1\n")
    (tmp_path / "mine.yaml").write_text(
        "name: mine\nndim: 2\nmap_file: map.csv\npitch_row_um: 50\npitch_col_um: 50\ncontact_radius_um: 5\n"
    )
    spec = load_probe_spec(tmp_path / "mine.yaml")
    assert spec.labels == ("47", "12")


def test_attach_probe_drops_reference_and_keeps_z(make_mcs_file):
    path, _ = make_mcs_file(labels=("47", "12", "33", "15"))
    with pytest.warns(UnverifiedGeometryWarning):
        rec, dropped = attach_probe(McsH5Recording(path), CUBE)
    assert dropped == ("15",)
    assert list(rec.channel_ids) == ["47", "12", "33"]
    spec = load_probe_spec(CUBE)
    expected = dict(zip(spec.labels, spec.positions_um()))
    np.testing.assert_allclose(rec.get_channel_locations(axes="xyz"), [expected[c] for c in rec.channel_ids])
    assert rec.get_channel_locations().shape[1] == 2  # SI's default silently drops z (D3)
    np.testing.assert_array_equal(rec.get_property("grid_layer"), [4, 3, 2])  # 47, 12, 33 in the cube map


def test_probe_survives_spikeinterface_roundtrip(make_mcs_file):
    path, _ = make_mcs_file(labels=("47", "12", "33"))
    rec, _ = attach_probe(McsH5Recording(path), CUBE)
    rec2 = si.load(rec.to_dict(include_annotations=True, include_properties=True))
    assert rec2.get_probe().ndim == 3
    np.testing.assert_array_equal(rec2.get_channel_locations(axes="xyz"), rec.get_channel_locations(axes="xyz"))


def test_unknown_channels_warn(make_mcs_file):
    path, _ = make_mcs_file(labels=("47", "99"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec, dropped = attach_probe(McsH5Recording(path), CUBE)
    assert dropped == ("99",)
    assert any("not on probe" in str(w.message) for w in caught)


def test_stacked_contacts_are_apart_in_3d(make_mcs_file):
    # 85 (row 2, col 1, layer 1) and 76 (row 2, col 1, layer 2) share x-y and differ only in z.
    path, _ = make_mcs_file(labels=("85", "76"))
    rec, _ = attach_probe(McsH5Recording(path), CUBE)
    spec = load_probe_spec(CUBE)
    d = channel_distances_um(rec)
    assert d[0, 1] == pytest.approx(spec.layer_spacing_um)
    np.testing.assert_allclose(rec.get_channel_locations()[0], rec.get_channel_locations()[1])  # 2D view collapses


def test_planar_positions_get_zero_z(make_mcs_file):
    path, _ = make_mcs_file(labels=("47", "12", "15"))
    rec, dropped = attach_probe(McsH5Recording(path), "mcs60_8x8_200um")
    assert dropped == ("15",)
    pos = channel_positions_um(rec)
    assert pos.shape == (2, 3) and np.all(pos[:, 2] == 0)
