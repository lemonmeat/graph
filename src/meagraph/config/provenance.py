"""Run provenance: every output folder stores the exact config and the code that produced it."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
from pydantic import BaseModel

import meagraph
from meagraph.config.yaml_io import save_yaml

_TRACKED = ("spikeinterface", "probeinterface", "numpy", "scipy", "h5py", "neo", "elephant", "networkx", "pydantic")


class InputFile(BaseModel):
    path: str
    size_bytes: int
    modified_utc: str
    mcs_file_guid: str | None = None


class Provenance(BaseModel):
    package: str = "meagraph"
    package_version: str
    git_commit: str | None
    git_dirty: bool | None
    python: str
    platform: str
    dependencies: dict[str, str]
    created_utc: str
    inputs: list[InputFile]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=Path(meagraph.__file__).parent, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _input_file(path: Path) -> InputFile:
    stat = path.stat()
    guid = None
    if path.suffix == ".h5":
        try:
            with h5py.File(path, "r") as f:
                guid = f["Data"].attrs.get("FileGUID")
                guid = guid.decode() if isinstance(guid, bytes) else (str(guid) if guid is not None else None)
        except (OSError, KeyError):
            guid = None
    return InputFile(
        path=str(path.resolve()),
        size_bytes=stat.st_size,
        modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        mcs_file_guid=guid,
    )


def collect_provenance(inputs: Sequence[str | Path] = ()) -> Provenance:
    deps = {}
    for name in _TRACKED:
        try:
            deps[name] = version(name)
        except PackageNotFoundError:
            pass
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return Provenance(
        package_version=meagraph.__version__,
        git_commit=commit,
        git_dirty=None if status is None else bool(status),
        python=sys.version.split()[0],
        platform=platform.platform(),
        dependencies=deps,
        created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=[_input_file(Path(p)) for p in inputs],
    )


def write_run_folder(
    out_dir: str | Path,
    config: BaseModel,
    inputs: Sequence[str | Path] = (),
    overwrite: bool = False,
) -> Path:
    """Create ``out_dir`` with ``config.yaml`` and ``provenance.json``; returns the folder."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; pass overwrite=True to reuse it")
    out.mkdir(parents=True, exist_ok=True)
    save_yaml(config, out / "config.yaml")
    prov = collect_provenance(inputs)
    (out / "provenance.json").write_text(json.dumps(prov.model_dump(mode="json"), indent=2) + "\n")
    return out
