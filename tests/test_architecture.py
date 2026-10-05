"""Guards for the package's structural rules (docs/ARCHITECTURE.md)."""

import subprocess
import sys

import numpy as np

from meagraph.intervals import inside, merge_intervals

CORE = ["meagraph", "meagraph.io", "meagraph.probe", "meagraph.config", "meagraph.preprocess", "meagraph.stimulation",
        "meagraph.detect", "meagraph.connectivity", "meagraph.connectivity.pipeline", "meagraph.synth", "meagraph.benchmark",
        "meagraph.cli"]  # fmt: skip


def test_core_library_never_imports_a_gui():
    """Only the viewer (and plotting calls in meagraph.viz) may load matplotlib."""
    code = f"import sys; import {', '.join(CORE)}; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_merge_intervals():
    np.testing.assert_array_equal(merge_intervals([[5, 6], [1, 3], [3, 4], [2, 2.5], [7, 7]]), [[1, 4], [5, 6]])
    assert merge_intervals(np.zeros((0, 2))).shape == (0, 2)
    assert merge_intervals(np.array([[3, 9]], dtype=np.int64)).dtype == np.int64


def test_inside_is_closed():
    iv = np.array([[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_array_equal(inside([0.5, 1.0, 2.0, 3.0, 4.5, 5.5], iv), [False, True, True, False, True, False])
    assert not inside([1.0], np.zeros((0, 2))).any()


def test_estimators_need_only_spike_trains():
    """Connectivity estimators must not depend on file reading or detection, so any spike
    source (simulation, a saved result, a live stream in Phase 7) can feed them."""
    code = "import sys, meagraph.connectivity; print(sorted(m for m in ('meagraph.io', 'meagraph.detect', 'meagraph.stimulation') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
