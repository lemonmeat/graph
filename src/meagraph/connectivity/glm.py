"""Poisson GLM coupling (Pillow et al. 2008; Stevenson et al. 2008) fitted with NeMoS.

For each target unit the spike count in every ``bin_ms`` bin is modelled as Poisson with

    log rate(t) = b + sum_k h_k * (spike history of unit k)(t) + g * network(t)

* ``h_k`` is a coupling filter (``h_target`` is the self-history filter) on NeMoS's raised-cosine
  log-stretched basis (``RaisedCosineLogConv``): narrow basis functions at short lags, wide ones at
  long lags. NeMoS's convolution is causal and shifted by one bin, so a spike never predicts itself.
* ``network(t)`` (``network_sigma_ms`` > 0) is log(1 + the summed spikes of every *other* unit,
  smoothed with a half-Gaussian over past bins), standardized. It must be causal: a centred kernel
  lets the source's spikes *after* the target's spike into the covariate, which made independent
  units look inhibitory (p ~ 1e-11 in tests/test_glm_stim.py, 2026-11-16). It stands in for the
  shared drive of network bursts, so a
  coupling filter has to explain the target's spikes beyond what the whole network's rate
  explains. It is not a linear function of the coupling features, so it is not collinear with
  them, but it does contain the source's own spikes: with few units it absorbs part of a real
  edge (lower recall), which is the price of the burst control.

Fit: NeMoS ``GLM`` (LBFGS) with a weak ridge penalty ``l2 * |w|^2 / 2`` on the log-likelihood
summed over bins (intercept unpenalized), i.e. a Gaussian prior with SD ``1/sqrt(l2)`` in log
gain. Without it the fit does not converge: a unit never fires within its refractory period, so
the maximum-likelihood self-history coefficient at lag 1 is minus infinity (checked against an
exact Newton fit, 2026-11-16: the penalized NeMoS fit matches it to 4e-5). LBFGS runs in float32
and can crawl for minutes near the optimum, so it stops at tolerance 1e-6 and up to three float64
Newton steps finish the fit (``extra["max_gradient"]`` per target records how well).

Significance: a Wald test of ``h_source = 0`` (chi-squared with ``n_basis`` degrees of freedom)
from the penalized Fisher information ``X' diag(rate) X + l2 I``, then
Benjamini-Hochberg over all ordered pairs, as for the other methods. The weight is the coupling
filter at its largest-magnitude lag, in log gain (``exp(weight)`` multiplies the target's rate);
its sign says excitatory (+) or inhibitory (-), and that lag is the delay. Filters are in
``extra["filters"]`` (source, target, lag) with lags in ``extra["filter_lags_ms"]``.

Parameters are defaults flagged for the owner (DECISIONS.md D29): 1 ms bins, 6 basis functions
over 25 ms, network covariate with a causal 20 ms half-Gaussian, l2 = 1.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt, model_validator
from scipy.signal import fftconvolve
from scipy.stats import chi2

from meagraph.connectivity.base import ConnectivityResult, Progress, empty_matrices, register
from meagraph.connectivity.stats import fdr_mask
from meagraph.spiketrains import SpikeTrains


def binned_counts(trains: SpikeTrains, bin_s: float) -> np.ndarray:
    """(n_bins, n_units) spike counts on bins from ``t_start``; a partial last bin is dropped."""
    n_bins = int(np.floor((trains.t_stop_s - trains.t_start_s) / bin_s + 1e-9))
    counts = np.zeros((n_bins, trains.n_units), dtype=np.float32)
    for u, t in enumerate(trains.times_s):
        idx = np.floor((np.asarray(t) - trains.t_start_s) / bin_s + 1e-9).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        np.add.at(counts[:, u], idx, 1.0)
    return counts


def causal_smooth(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    """``x`` smoothed with a half-Gaussian over past bins only (lags 1 .. 4 sigma), so the value at
    bin t uses nothing from bin t or later."""
    lags = np.arange(1, int(np.ceil(4 * sigma_bins)) + 1)
    kernel = np.exp(-0.5 * (lags / sigma_bins) ** 2)
    out = fftconvolve(x, np.concatenate([[0.0], kernel / kernel.sum()]))[: x.size]
    return np.clip(out, 0, None)  # FFT round-off


def _basis(n_basis: int, window_bins: int):
    import nemos as nmo

    return nmo.basis.RaisedCosineLogConv(n_basis_funcs=n_basis, window_size=window_bins)


def history_features(counts: np.ndarray, n_basis: int, window_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal raised-cosine history features of every unit, (n_bins, n_units * n_basis) with unit
    ``u`` in columns ``u*n_basis:(u+1)*n_basis``, and the kernels (window_bins, n_basis); kernel row
    ``k`` is lag ``k + 1`` bins. The first ``window_bins`` rows (incomplete history) are NaN."""
    basis = _basis(n_basis, window_bins)
    feats = np.empty((counts.shape[0], counts.shape[1] * n_basis), dtype=np.float32)
    for u in range(counts.shape[1]):
        feats[:, u * n_basis : (u + 1) * n_basis] = np.asarray(basis.compute_features(counts[:, u]), dtype=np.float32)
    _, kernels = basis.evaluate_on_grid(window_bins)
    return feats, np.asarray(kernels, dtype=np.float64)


def _fit_poisson(x: np.ndarray, y: np.ndarray, cfg) -> tuple[np.ndarray, float]:
    import nemos as nmo

    # NeMoS minimizes the *mean* negative log-likelihood plus strength * |w|^2 / 2, so the
    # strength is l2 per bin.
    model = nmo.glm.GLM(observation_model="Poisson", solver_name="LBFGS", regularizer="Ridge",
                        regularizer_strength=cfg.l2 / len(y), solver_kwargs={"maxiter": cfg.max_iter, "tol": 1e-6})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # NeMoS warns about float32 inputs and convergence on every fit
        model.fit(x, y)
    return np.asarray(model.coef_, dtype=np.float64).ravel(), float(np.asarray(model.intercept_).ravel()[0])


def _penalized_objective(x64: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float) -> float:
    eta = np.clip(x64 @ w[:-1] + w[-1], -50, 50)
    return float(np.sum(y * eta - np.exp(eta)) - 0.5 * l2 * float(w[:-1] @ w[:-1]))


def _polish(x: np.ndarray, y: np.ndarray, coef: np.ndarray, intercept: float, cfg) -> tuple[np.ndarray, np.ndarray, float]:
    """Newton steps (with step halving) on the penalized log-likelihood from the NeMoS solution.

    NeMoS's LBFGS stops at a float32 tolerance; from there Newton converges in one or two steps,
    and its Hessian is the Fisher information the Wald test needs anyway. Returns the weights
    (intercept last), the penalized information at them, and the final max |gradient|."""
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    w = np.append(coef, intercept)
    pen = np.full(w.size, cfg.l2)
    pen[-1] = 0.0
    for _ in range(cfg.newton_steps + 1):
        rate = np.exp(np.clip(x64 @ w[:-1] + w[-1], -50, 50))
        grad = np.append(x64.T @ (y64 - rate), np.sum(y64 - rate)) - pen * w
        info = _fisher(x, rate) + np.diag(pen)
        if _ == cfg.newton_steps or np.max(np.abs(grad)) < 1e-6:
            break
        step = np.linalg.lstsq(info, grad, rcond=None)[0]
        f0, t = _penalized_objective(x64, y64, w, cfg.l2), 1.0
        while _penalized_objective(x64, y64, w + t * step, cfg.l2) < f0 and t > 1e-4:
            t /= 2
        w = w + t * step
    return w, info, float(np.max(np.abs(grad)))


def _fisher(x: np.ndarray, rate: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """``X' diag(rate) X`` with an intercept column appended, in float64, chunked."""
    k = x.shape[1] + 1
    h = np.zeros((k, k))
    for i in range(0, x.shape[0], chunk):
        xa = np.column_stack([x[i : i + chunk].astype(np.float64), np.ones(min(chunk, x.shape[0] - i))])
        h += xa.T @ (xa * rate[i : i + chunk, None])
    return h


@register
class PoissonGLM:
    name = "glm"
    signed = True

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        bin_ms: PositiveFloat = 1.0
        window_ms: PositiveFloat = 25.0  # coupling filters span lags 1 .. window bins
        n_basis: PositiveInt = 6
        network_sigma_ms: NonNegativeFloat = 20.0  # 0 turns the network-rate covariate off
        q: PositiveFloat = 0.05
        l2: PositiveFloat = 1.0  # ridge penalty on the summed log-likelihood (prior SD 1 in log gain)
        min_spikes: PositiveInt = 100
        max_iter: PositiveInt = 200  # NeMoS LBFGS iterations (float32 tolerance 1e-6)...
        newton_steps: int = 3  # ...then at most this many Newton steps in float64

        @model_validator(mode="after")
        def _check(self):
            if round(self.window_ms / self.bin_ms) < self.n_basis + 1:
                raise ValueError("window_ms / bin_ms must exceed n_basis")
            return self

    def estimate(self, trains: SpikeTrains, config: Config | None = None, progress: Progress | None = None) -> ConnectivityResult:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        cfg = config or self.Config()
        n, nb = trains.n_units, cfg.n_basis
        bin_s = cfg.bin_ms / 1e3
        window = int(round(cfg.window_ms / cfg.bin_ms))
        weights, delays, p_values, _ = empty_matrices(n)
        filters = np.full((n, n, window), np.nan)
        lags_ms = np.arange(1, window + 1) * cfg.bin_ms
        units = np.flatnonzero(trains.n_spikes() >= cfg.min_spikes)  # modelled units: sources and targets
        extra: dict = {"filter_lags_ms": lags_ms}
        if units.size >= 2:
            counts = binned_counts(trains, bin_s)
            feats, kernels = history_features(counts[:, units], nb, window)
            valid = slice(window, counts.shape[0])
            x_hist = feats[valid]
            total = counts.sum(axis=1)
            network_coef, max_grad = np.full(n, np.nan), np.full(n, np.nan)
            if progress:
                progress(0, units.size)
            for a, j in enumerate(units):
                y = counts[valid, j]
                x = x_hist
                if cfg.network_sigma_ms > 0:
                    others = causal_smooth((total - counts[:, j]).astype(np.float64), cfg.network_sigma_ms / cfg.bin_ms)
                    z = np.log1p(others[valid])
                    z = (z - z.mean()) / (z.std() or 1.0)  # standardized: LBFGS converges ~2x faster; coupling tests unchanged
                    x = np.column_stack([x_hist, z.astype(np.float32)])
                coef, intercept = _fit_poisson(x, y, cfg)
                w, info, max_grad[j] = _polish(x, y, coef, intercept, cfg)
                coef = w[:-1]
                cov = np.linalg.pinv(info)
                if cfg.network_sigma_ms > 0:
                    network_coef[j] = coef[-1]
                for b, i in enumerate(units):
                    sl = slice(b * nb, (b + 1) * nb)
                    beta = coef[sl]
                    h = kernels @ beta
                    filters[i, j] = h
                    if i == j:
                        continue
                    stat = float(beta @ np.linalg.pinv(cov[sl, sl]) @ beta)
                    p_values[i, j] = float(chi2.sf(stat, nb))
                    k = int(np.argmax(np.abs(h)))
                    weights[i, j], delays[i, j] = h[k], lags_ms[k]
                if progress:
                    progress(a + 1, units.size)
            extra.update(filters=filters, network_coef=network_coef, max_gradient=max_grad)
        significant = fdr_mask(p_values, cfg.q)
        return ConnectivityResult(
            method=self.name, node_ids=trains.unit_ids, weights=weights, delays_ms=delays, p_values=p_values,
            significant=significant, directed=True, params=cfg.model_dump(), n_spikes=trains.n_spikes(),
            duration_s=trains.duration_s, positions_um=trains.positions_um, extra=extra, signed=True,
        )  # fmt: skip
