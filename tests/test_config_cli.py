import json

import pytest
from pydantic import ValidationError

from meagraph.cli import main
from meagraph.config import SessionConfig, load_yaml, save_yaml, write_run_folder
from meagraph.io import inspect_file


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
