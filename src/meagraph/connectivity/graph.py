"""Graphs from connectivity results, and saving/loading them.

A result folder holds:
- ``graph.graphml`` and ``graph.json``: significant edges, nodes with 3D positions
- ``edges.csv``: one row per tested pair (weight, delay, p, significant)
- ``matrices.npz``: the full matrices, for exact reloading
- ``config.yaml`` and ``provenance.json``
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
from pydantic import BaseModel

from meagraph.config import write_run_folder
from meagraph.connectivity.base import ConnectivityResult


class _Params(BaseModel):
    """What config.yaml records: the estimator's params and, from the pipeline, how its input was chosen."""

    method: str
    params: dict
    pipeline: dict | None = None


def to_networkx(result: ConnectivityResult, significant_only: bool = True) -> nx.Graph:
    g = nx.DiGraph() if result.directed else nx.Graph()
    g.graph.update(method=result.method, duration_s=float(result.duration_s))
    for k, node in enumerate(result.node_ids):
        attrs = dict(n_spikes=int(result.n_spikes[k]))
        if result.positions_um is not None:
            for axis, v in zip("xyz", result.positions_um[k]):
                attrs[f"{axis}_um"] = float(v)
        g.add_node(node, **attrs)
    keep = result.significant if significant_only else result.tested
    for i, j in zip(*np.nonzero(keep)):
        if not result.directed and j < i:
            continue
        attrs = dict(weight=float(result.weights[i, j]), p_value=float(result.p_values[i, j]))
        if np.isfinite(result.delays_ms[i, j]):
            attrs["delay_ms"] = float(result.delays_ms[i, j])
        g.add_edge(result.node_ids[i], result.node_ids[j], **attrs)
    return g


def save_result(result: ConnectivityResult, folder: str | Path, inputs=(), overwrite: bool = False,
                pipeline: BaseModel | None = None) -> Path:  # fmt: skip
    """Write the files listed in the module docstring. ``pipeline`` (e.g. a ``GraphConfig``) is
    recorded in config.yaml next to the estimator's params."""
    params = _Params(method=result.method, params=result.params, pipeline=pipeline.model_dump(mode="json") if pipeline else None)
    out = write_run_folder(folder, params, inputs=inputs, overwrite=overwrite)
    g = to_networkx(result)
    nx.write_graphml(g, out / "graph.graphml")
    (out / "graph.json").write_text(json.dumps(nx.node_link_data(g, edges="edges"), indent=1))
    with open(out / "edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "weight", "delay_ms", "p_value", "significant"])
        for i, j in zip(*np.nonzero(result.tested)):
            if not result.directed and j < i:
                continue
            w.writerow([result.node_ids[i], result.node_ids[j], f"{result.weights[i, j]:.6g}", f"{result.delays_ms[i, j]:.3g}",
                        f"{result.p_values[i, j]:.4g}", int(result.significant[i, j])])  # fmt: skip
    np.savez_compressed(
        out / "matrices.npz", node_ids=np.array(result.node_ids), weights=result.weights, delays_ms=result.delays_ms,
        p_values=result.p_values, significant=result.significant, n_spikes=result.n_spikes,
        positions_um=result.positions_um if result.positions_um is not None else np.zeros((0, 3)),
    )  # fmt: skip
    (out / "result.json").write_text(json.dumps(dict(method=result.method, directed=result.directed, duration_s=result.duration_s, params=result.params)))
    return out


def load_result(folder: str | Path) -> ConnectivityResult:
    folder = Path(folder)
    meta = json.loads((folder / "result.json").read_text())
    z = np.load(folder / "matrices.npz")
    positions = z["positions_um"] if z["positions_um"].size else None
    return ConnectivityResult(
        method=meta["method"], node_ids=tuple(str(x) for x in z["node_ids"]), weights=z["weights"], delays_ms=z["delays_ms"],
        p_values=z["p_values"], significant=z["significant"], directed=meta["directed"], params=meta["params"],
        n_spikes=z["n_spikes"], duration_s=meta["duration_s"], positions_um=positions,
    )  # fmt: skip
