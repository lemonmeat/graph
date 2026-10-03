"""meagraph: connectivity, stimulation and reservoir analysis for MEA recordings.

The library has no GUI imports, no global state and no hardcoded paths. Every step takes
in-memory objects and returns in-memory objects. See docs/PLAN.md for the module map.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("meagraph")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

from meagraph.spiketrains import SpikeTrains

__all__ = ["SpikeTrains", "__version__"]
