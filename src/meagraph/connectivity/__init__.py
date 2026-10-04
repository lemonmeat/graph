"""Connectivity estimators sharing one interface, plus significance tools and graph output.

    from meagraph.connectivity import estimate
    result = estimate(trains, "cch_jitter")          # or "cch_hollow", "sttc"
"""

from meagraph.connectivity import cch, sttc  # noqa: F401  (registers the estimators)
from meagraph.connectivity.base import ESTIMATORS, ConnectivityResult, estimate, register
from meagraph.connectivity.graph import load_result, save_result, to_networkx
from meagraph.connectivity.stats import cross_correlograms, fdr_mask, interval_jitter, window_counts

__all__ = [
    "ESTIMATORS",
    "ConnectivityResult",
    "cross_correlograms",
    "estimate",
    "fdr_mask",
    "interval_jitter",
    "load_result",
    "register",
    "save_result",
    "to_networkx",
    "window_counts",
]
