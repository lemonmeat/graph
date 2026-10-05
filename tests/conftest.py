import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from mcs_fixture import write_mcs_h5  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MEAGRAPH_DATA_DIR", REPO / "data"))


@pytest.fixture
def make_mcs_file(tmp_path):
    """Factory: ``make_mcs_file(**kwargs) -> (path, truth)``; see ``mcs_fixture.write_mcs_h5``."""
    counter = iter(range(1000))

    def make(**kwargs):
        path = tmp_path / f"fixture_{next(counter)}.h5"
        return path, write_mcs_h5(path, **kwargs)

    return make


@pytest.fixture
def mcs_file(make_mcs_file):
    """A fixture recording with 3 stimulation trains and MCS detector spikes."""
    return make_mcs_file(
        stim_onsets_us=[510_000, 560_000, 610_000],
        detector_spikes_us={"47": [520_000, 530_000, 540_000], "12": [600_000]},
    )


def real_recordings() -> list[Path]:
    return sorted(p for p in DATA_DIR.glob("*.h5") if not p.name.endswith(".spikes.h5"))


@pytest.fixture(scope="session")
def real_files() -> list[Path]:
    files = real_recordings()
    if not files:
        pytest.skip(f"no MCS recordings in {DATA_DIR} (set MEAGRAPH_DATA_DIR)")
    return files
