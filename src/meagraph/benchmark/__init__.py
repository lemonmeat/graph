"""Score connectivity estimators against synthetic ground truth.

    rows = run_benchmark(scenarios(), methods=("cch_hollow", "cch_jitter", "sttc"))

Each row is one (scenario, seed, method, spike set) with precision, recall, false-positive
rate, ROC AUC and delay error. ``spike set`` is ``all``, ``no_bursts`` (network-burst periods
removed) or ``robust`` (edges significant in both), the burst control of DECISIONS.md D14.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from meagraph.connectivity import ConnectivityResult
from meagraph.connectivity.pipeline import burst_controlled
from meagraph.detect.bursts import BurstConfig, network_bursts
from meagraph.synth import NetworkConfig, SyntheticNetwork, simulate_network


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney form, ties averaged); NaN if one class is empty."""
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def score(result: ConnectivityResult, truth: SyntheticNetwork, significant: np.ndarray | None = None) -> dict:
    """Detection metrics over the pairs the estimator tested. Undirected methods are scored
    against "connected in either direction", once per unordered pair."""
    sig = result.significant if significant is None else significant
    tested = result.tested.copy()
    true = truth.connected.copy()
    if not result.directed:
        true = true | true.T
        tested &= np.triu(np.ones_like(tested), 1)
    est, true_t = sig[tested], true[tested]
    tp, fp = int(np.sum(est & true_t)), int(np.sum(est & ~true_t))
    fn, tn = int(np.sum(~est & true_t)), int(np.sum(~est & ~true_t))
    # Pairs linked only indirectly (i -> k -> j, or common input k -> i and k -> j) really do
    # correlate; pairwise methods cannot tell them from direct connections.
    a = truth.connected.astype(np.int64)
    indirect = ((a @ a) > 0) | ((a.T @ a) > 0)
    if not result.directed:
        indirect = indirect | indirect.T
    fp_indirect = int(np.sum(sig[tested] & ~true_t & indirect[tested]))
    scores = -np.log10(np.clip(result.p_values[tested], 1e-300, 1))
    out = dict(
        n_tested=int(tested.sum()), n_true=int(true_t.sum()), tp=tp, fp=fp, fn=fn, fp_indirect=fp_indirect,
        precision=tp / (tp + fp) if tp + fp else float("nan"),
        precision_functional=(tp + fp_indirect) / (tp + fp) if tp + fp else float("nan"),
        recall=tp / (tp + fn) if tp + fn else float("nan"),
        false_positive_rate=fp / (fp + tn) if fp + tn else float("nan"),
        auc=roc_auc(scores, true_t),
    )  # fmt: skip
    hit = sig & truth.connected & tested
    out["delay_error_ms"] = float(np.nanmedian(np.abs(result.delays_ms[hit] - truth.delays_ms[hit]))) if result.directed and hit.any() else float("nan")
    return out


def scenarios(durations_s: Sequence[float] = (120, 300, 600, 1800), weights: Sequence[float] = (0.03, 0.06, 0.12),
              bursts: Sequence[bool] = (False, True), base: NetworkConfig | None = None) -> list[tuple[str, NetworkConfig]]:
    """The standard sweep: recovery vs recording length and connection strength, with and
    without network bursts, plus null networks (no connections) for the false-positive rate.
    Burst settings follow the DIV142 spontaneous recording (about 0.24 bursts/s of ~100 ms)."""
    base = base or NetworkConfig()
    burst_kw = dict(burst_rate_hz=0.24, burst_duration_ms=100.0, burst_gain=20.0)
    out = []
    for b in bursts:
        extra = burst_kw if b else {}
        tag = "bursts" if b else "no-bursts"
        for d in durations_s:
            for w in weights:
                out.append((f"{tag} T={d:g}s w={w:g}", base.model_copy(update=dict(duration_s=d, weight=(w, w), **extra))))
        out.append((f"{tag} null T=600s", base.model_copy(update=dict(duration_s=600.0, connection_prob=0.0, **extra))))
    out.append(("stim null T=600s", base.model_copy(update=dict(duration_s=600.0, connection_prob=0.0, stim_interval_s=5.0))))
    return out


def _run_one(job) -> list[dict]:
    """One simulated network, every method; module-level so worker processes can run it."""
    name, cfg, seed, methods, method_overrides, burst_config = job
    net = simulate_network(cfg.model_copy(update=dict(seed=seed)))
    periods = network_bursts(net.trains, burst_config)
    rows = []
    for method in methods:
        start = time.time()
        bc = burst_controlled(net.trains, method, (method_overrides or {}).get(method), bursts=periods)
        base = dict(scenario=name, seed=seed, method=method, duration_s=cfg.duration_s, weight=cfg.weight[1],
                    bursts=cfg.burst_rate_hz > 0, connection_prob=cfg.connection_prob, seconds=round(time.time() - start, 2))  # fmt: skip
        for spikes in ("all", "no_bursts", "robust"):
            rows.append({**base, "spikes": spikes, **score(getattr(bc, spikes), net)})
    return rows


def run_benchmark(
    scenario_list: Iterable[tuple[str, NetworkConfig]],
    methods: Sequence[str] = ("cch_hollow", "cch_jitter", "sttc"),
    seeds: Sequence[int] = (0, 1, 2),
    method_overrides: dict[str, dict] | None = None,
    burst_config: BurstConfig | None = None,
    n_jobs: int = 1,
    progress=print,
) -> list[dict]:
    """Every (scenario, seed) pair is one job; ``n_jobs`` > 1 runs them in worker processes."""
    jobs = [(name, cfg, seed, tuple(methods), method_overrides, burst_config) for name, cfg in scenario_list for seed in seeds]
    rows: list[dict] = []
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(n_jobs) as pool:
            results = pool.map(_run_one, jobs)
            for k, (job, job_rows) in enumerate(zip(jobs, results), 1):
                rows.extend(job_rows)
                if progress:
                    progress(f"[{k}/{len(jobs)}] {job[0]} seed {job[2]}")
    else:
        for k, job in enumerate(jobs, 1):
            rows.extend(_run_one(job))
            if progress:
                progress(f"[{k}/{len(jobs)}] {job[0]} seed {job[2]}")
    return rows


def write_rows(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def summarize(rows: list[dict], spikes: str = "all") -> list[dict]:
    """Mean over seeds per (scenario, method)."""
    keys = sorted({(r["scenario"], r["method"]) for r in rows if r["spikes"] == spikes})
    out = []
    for scen, method in keys:
        group = [r for r in rows if r["scenario"] == scen and r["method"] == method and r["spikes"] == spikes]
        agg = {"scenario": scen, "method": method, "spikes": spikes}
        for m in ("precision", "recall", "false_positive_rate", "auc", "delay_error_ms", "fp"):
            agg[m] = float(np.nanmean([r[m] for r in group])) if any(np.isfinite(r[m]) for r in group) else float("nan")
        out.append(agg)
    return out
