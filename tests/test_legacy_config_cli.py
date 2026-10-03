import hashlib
import json

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from conftest import REPO
from meagraph.cli import main
from meagraph.config import SessionConfig, load_yaml, save_yaml, write_run_folder
from meagraph.io import inspect_file, read_spikes_sidecar, sidecar_path


def _write_sidecar(path, t_start=0.5):
    params = dict(k=5.0, cutout_ms=[1.0, 2.0], t_start=t_start, duration=10.0, fs=10000.0)
    with h5py.File(path, "w") as f:
        f.attrs["source"] = "rec.h5"
        f.attrs["params"] = json.dumps(params)
        g = f.create_group("SpikeStream").create_group("Electrode_47")
        g.create_dataset("ts", data=np.array([0.1, 2.0]))
        g.create_dataset("amp", data=np.array([-30.0, -40.0], np.float32))
        g.create_dataset("waveforms", data=np.zeros((30, 2), np.float32))
        g.attrs["sigma_uv"] = 4.5


def test_regression_baseline_is_intact():
    """The committed spikes.py outputs must never change (DECISIONS.md D7)."""
    base = REPO / "tests" / "data" / "legacy_baseline"
    lines = (base / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, name
        read_spikes_sidecar(base / name)  # and it still parses


def test_sidecar_path_matches_legacy_naming(tmp_path):
    assert sidecar_path(tmp_path / "a.b.h5").name == "a.b.spikes.h5"


def test_legacy_times_move_to_recording_clock(tmp_path):
    path = tmp_path / "rec.spikes.h5"
    _write_sidecar(path, t_start=0.5)
    legacy = read_spikes_sidecar(path)
    np.testing.assert_allclose(legacy.trains.as_dict()["47"], [0.6, 2.5])
    assert (legacy.trains.t_start_s, legacy.trains.t_stop_s) == (0.5, 10.5)
    assert legacy.waveforms_uv["47"].shape == (30, 2)
    assert legacy.cutout_ms == (1.0, 2.0) and legacy.sigma_uv["47"] == 4.5


def test_session_config_yaml_roundtrip(tmp_path):
    cfg = SessionConfig(path=tmp_path / "rec.h5", stim_site={"STG 1": "47"})
    assert load_yaml(save_yaml(cfg, tmp_path / "s.yaml"), SessionConfig) == cfg
    with pytest.raises(ValidationError):
        SessionConfig(path="x.h5", unknown_field=1)


def test_run_folder_stores_config_and_provenance(tmp_path, mcs_file):
    cfg = SessionConfig(path=mcs_file[0])
    out = write_run_folder(tmp_path / "run", cfg, inputs=[mcs_file[0]])
    assert load_yaml(out / "config.yaml", SessionConfig) == cfg
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["package"] == "meagraph" and prov["package_version"]
    assert "spikeinterface" in prov["dependencies"]
    assert prov["inputs"][0]["mcs_file_guid"] == "fixture-guid"
    with pytest.raises(FileExistsError):
        write_run_folder(tmp_path / "run", cfg)


def test_inventory_orders_streams_by_lineage(mcs_file):
    inv = inspect_file(mcs_file[0])
    assert [s.short_label for s in inv.analog_streams] == ["Data Acquisition (1)", "Filter (1)"]
    assert inv.raw_stream.name == "Stream_1"
    assert [e.event_id for e in inv.event_entities] == [1, 2]
    assert {s.short_label: s.n_events for s in inv.segment_streams} == {"Spike Sorter (1)": 4, "Spike Detector (1)": 4}


def test_cli_info_and_probes(mcs_file, capsys):
    assert main(["info", str(mcs_file[0])]) == 0
    out = capsys.readouterr().out
    assert "Data Acquisition (1)" in out and "STG 1 Single Pulse Start" in out
    assert main(["probes"]) == 0
    assert "cube4x4x4_E-00303" in capsys.readouterr().out
