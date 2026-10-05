"""Score connectivity estimators against synthetic ground truth.

    rows = run_benchmark(scenarios(), methods=("cch_hollow", "cch_jitter", "sttc"))

Each row is one (scenario, seed, method, spike set) with precision, recall, false-positive
rate, ROC AUC and delay error. ``spike set`` is ``all``, ``no_bursts`` (network-burst periods
removed) or ``robust`` (edges significant in both), the burst control of DECISIONS.md D14.

Excitatory and inhibitory connections are scored separately. Methods that report inhibition
(``signed`` results) get ``inh_*`` columns; ``inh_auc`` is also given for any method that stores
a one-sided inhibition p-value (``extra["p_inhibitory"]``). Rule variants (``VARIANTS``) re-score
one estimate with a different edge rule, e.g. CFP without its minimum peak width, so rules can be
compared without re-running the estimator.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from meagraph.connectivity import ConnectivityResult
from meagraph.connectivity.pipeline import burst_controlled, robust_mask
from meagraph.detect.bursts import BurstConfig, network_bursts
from meagraph.synth import NetworkConfig, SyntheticNetwork, simulate_network


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney form, ties averaged); NaN if one class is empty."""
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _auc(p: np.ndarray | None, truth: np.ndarray, tested: np.ndarray, ranking: np.ndarray | None = None) -> float:
    """ROC area from p-values, or from ``ranking`` (higher = stronger) for methods without a test."""
    if ranking is not None:
        return roc_auc(ranking[tested], truth[tested])
    if p is None:
        return float("nan")
    return roc_auc(-np.log10(np.clip(p[tested], 1e-300, 1)), truth[tested])


def score(result: ConnectivityResult, truth: SyntheticNetwork, significant: np.ndarray | None = None) -> dict:
    """Detection metrics over the pairs the estimator tested. Undirected methods are scored
    against "connected in either direction", once per unordered pair. Excitatory metrics count
    only excitatory calls (for signed methods, edges with positive weight)."""
    sig = result.significant if significant is None else significant
    tested = result.tested.copy()
    true_exc, true_inh = truth.excitatory.copy(), truth.inhibitory.copy()
    if not result.directed:
        true_exc, true_inh = true_exc | true_exc.T, true_inh | true_inh.T
        tested &= np.triu(np.ones_like(tested), 1)
    with np.errstate(invalid="ignore"):
        exc_call = sig & (result.weights > 0) if result.signed else sig
        inh_call = sig & (result.weights < 0) if result.signed else np.zeros_like(sig)
    est, true_t = exc_call[tested], true_exc[tested]
    tp, fp = int(np.sum(est & true_t)), int(np.sum(est & ~true_t))
    fn, tn = int(np.sum(~est & true_t)), int(np.sum(~est & ~true_t))
    # Pairs linked only indirectly (i -> k -> j, or common input k -> i and k -> j) really do
    # correlate; pairwise methods cannot tell them from direct connections.
    a = truth.excitatory.astype(np.int64)
    indirect = ((a @ a) > 0) | ((a.T @ a) > 0)
    if not result.directed:
        indirect = indirect | indirect.T
    fp_indirect = int(np.sum(exc_call[tested] & ~true_t & indirect[tested]))
    p_exc = result.extra.get("p_excitatory", result.p_values)
    out = dict(
        n_tested=int(tested.sum()), n_true=int(true_t.sum()), tp=tp, fp=fp, fn=fn, fp_indirect=fp_indirect,
        precision=tp / (tp + fp) if tp + fp else float("nan"),
        precision_functional=(tp + fp_indirect) / (tp + fp) if tp + fp else float("nan"),
        recall=tp / (tp + fn) if tp + fn else float("nan"),
        false_positive_rate=fp / (fp + tn) if fp + tn else float("nan"),
        auc=_auc(p_exc, true_exc, tested, result.extra.get("ranking")),
    )  # fmt: skip
    hit = exc_call & truth.excitatory & tested
    out["delay_error_ms"] = float(np.nanmedian(np.abs(result.delays_ms[hit] - truth.delays_ms[hit]))) if result.directed and hit.any() else float("nan")
    # Inhibition
    n_inh = int(true_inh[tested].sum())
    out.update(n_true_inh=n_inh, inh_tp=float("nan"), inh_fp=float("nan"), inh_fn=float("nan"), inh_fp_reverse=float("nan"),
               inh_precision=float("nan"), inh_recall=float("nan"),
               inh_auc=_auc(result.extra.get("p_inhibitory"), true_inh, tested) if n_inh else float("nan"))  # fmt: skip
    if result.signed:
        e, t = inh_call[tested], true_inh[tested]
        itp, ifp, ifn = int(np.sum(e & t)), int(np.sum(e & ~t)), int(np.sum(~e & t))
        reverse = (true_exc.T & ~true_inh)[tested]  # the opposite direction of an excitatory connection
        out.update(inh_tp=itp, inh_fp=ifp, inh_fn=ifn, inh_fp_reverse=int(np.sum(e & reverse)),
                   inh_precision=itp / (itp + ifp) if itp + ifp else float("nan"),
                   inh_recall=itp / (itp + ifn) if itp + ifn else float("nan"))  # fmt: skip
    return out


def _cfp_any_width(r: ConnectivityResult) -> np.ndarray:
    """CFP's test without the minimum peak width (the delay limit still applies)."""
    fit = r.extra["fit_m_t_w_offset"]
    with np.errstate(invalid="ignore"):
        return r.extra["test_significant"] & np.isfinite(fit[..., 0]) & (fit[..., 1] <= r.params["max_delay_ms"])


def _tspe_no_reverse(r: ConnectivityResult) -> np.ndarray:
    """TSPE without inhibitory edges that oppose a significant excitatory edge (D22)."""
    with np.errstate(invalid="ignore"):
        exc = r.significant & (r.weights > 0)
        return r.significant & ~((r.weights < 0) & exc.T)


VARIANTS = {"cfp": {"cfp_narrow": _cfp_any_width}, "tspe": {"tspe_noreverse": _tspe_no_reverse}}


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


def inhibition_scenarios(durations_s: Sequence[float] = (600, 1800), bursts: Sequence[bool] = (False, True),
                         base: NetworkConfig | None = None) -> list[tuple[str, NetworkConfig]]:  # fmt: skip
    """Networks with inhibitory units (D22): 20 % of units inhibitory, each inhibitory connection
    deleting 60 % of target spikes for 10 ms; excitatory weight 0.06. Plus one high-rate network
    (2-10 Hz), where inhibition is easier to see because the target fires often enough to show a dip."""
    base = (base or NetworkConfig()).model_copy(update=dict(weight=(0.06, 0.06), inhibitory_fraction=0.2,
                                                            inhibition=(0.6, 0.6), inhibition_ms=10.0))  # fmt: skip
    burst_kw = dict(burst_rate_hz=0.24, burst_duration_ms=100.0, burst_gain=20.0)
    out = []
    for b in bursts:
        for d in durations_s:
            out.append((f"inh {'bursts' if b else 'no-bursts'} T={d:g}s", base.model_copy(update=dict(duration_s=d, **(burst_kw if b else {})))))
    out.append(("inh high-rate T=600s", base.model_copy(update=dict(duration_s=600.0, rate_hz=(2.0, 10.0)))))
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
                    bursts=cfg.burst_rate_hz > 0, connection_prob=cfg.connection_prob,
                    inhibitory_fraction=cfg.inhibitory_fraction, seconds=round(time.time() - start, 2))  # fmt: skip
        for spikes in ("all", "no_bursts", "robust"):
            rows.append({**base, "spikes": spikes, **score(getattr(bc, spikes), net)})
        for variant, rule in VARIANTS.get(method, {}).items():
            masks = dict(all=rule(bc.all), no_bursts=rule(bc.no_bursts))
            masks["robust"] = robust_mask(bc.all, bc.no_bursts, masks["all"], masks["no_bursts"])
            for spikes, result in (("all", bc.all), ("no_bursts", bc.no_bursts), ("robust", bc.all)):
                rows.append({**base, "method": variant, "spikes": spikes, **score(result, net, masks[spikes])})
    return rows


def run_benchmark(
    scenario_list: Iterable[tuple[str, NetworkConfig]],
    methods: Sequence[str] = ("cch_hollow", "cch_jitter", "sttc", "tspe", "cfp"),
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
        for m in ("precision", "recall", "false_positive_rate", "auc", "delay_error_ms", "fp", "inh_recall", "inh_precision", "inh_fp_reverse", "inh_auc"):
            agg[m] = float(np.nanmean([r[m] for r in group])) if any(np.isfinite(r[m]) for r in group) else float("nan")
        out.append(agg)
    return out
