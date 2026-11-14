"""Command line entry point: ``meagraph <command>``. Commands are thin wrappers over the library."""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path


def _stim_site(values: list[str] | None) -> str | dict[str, str] | None:
    """``--stim-site 47`` (one STG output) or ``--stim-site "STG 1=47" --stim-site "STG 2=82"``."""
    if not values:
        return None
    if len(values) == 1 and "=" not in values[0]:
        return values[0]
    pairs = [v.split("=", 1) for v in values]
    if any(len(p) != 2 for p in pairs):
        raise SystemExit("--stim-site takes one electrode, or 'STG n=electrode' for each output")
    return {k.strip().upper(): v.strip() for k, v in pairs}


def _stimlog(value: str | None) -> str | None:
    """``--stimlog``: "auto" (default) finds the log next to the recording; "none" ignores logs."""
    return None if value is not None and value.lower() == "none" else (value or "auto")


def _session(path: str, args: argparse.Namespace):
    """Open a recording with the session options shared by several commands, and report the log."""
    from meagraph.io import load_session

    session = load_session(path, probe=args.probe, stim_site=_stim_site(args.stim_site), stimlog=_stimlog(args.stimlog))
    if session.inventory.raw_stream is None:
        print(f"  no raw stream recorded; using {session.analog_stream}")
    if session.stimlog is not None:
        print(f"  stimulation log: {session.stimlog.path.name}")
        for a in session.alignment:
            print(f"    {a.summary()}")
    return session


def _pulses(stim):
    """Stimulation events that are pulses. An output's Marker events (recorded when the script sets
    the sideband marker bit) repeat its pulse times, so they are left out where pulses exist."""
    with_pulses = {s.source for s in stim if s.kind == "Single Pulse"}
    return [s for s in stim if s.kind == "Single Pulse" or s.source not in with_pulses]


def _recordings(paths: list[str]) -> list[str]:
    """Drop legacy ``*.spikes.h5`` sidecars that shell globs like ``*.h5`` pick up."""
    keep = [p for p in paths if not p.endswith(".spikes.h5")]
    for p in sorted(set(paths) - set(keep)):
        print(f"skipping {Path(p).name} (spike sidecar, not a recording)")
    return keep


def _info(args: argparse.Namespace) -> int:
    from meagraph.io.inventory import format_inventory, inspect_file
    from meagraph.io.stimlog import find_stimlog

    for i, path in enumerate(_recordings(args.paths)):
        if i:
            print()
        print(format_inventory(inspect_file(path, args.recording_index)))
        try:
            log = find_stimlog(path)
        except ValueError as exc:
            print(f"  stimulation log: {exc}")
        else:
            print(f"  stimulation log: {log.name if log else 'none found'}")
    return 0


def _probes(args: argparse.Namespace) -> int:
    from meagraph.probe.spec import available_probe_specs, load_probe_spec

    for name in available_probe_specs():
        spec = load_probe_spec(name)
        flag = "" if spec.geometry_verified else "  [placeholder geometry]"
        print(f"{name}: {len(spec.contacts)} contacts, {spec.ndim}D{flag}\n    {spec.description}")
    return 0


def _audit(args: argparse.Namespace) -> int:
    from meagraph.stimulation import audit_stimulation, format_audit

    for path in _recordings(args.paths):
        print(Path(path).name)
        session = _session(path, args)
        if not session.stim:
            print("  no stimulation events")
        for stim in _pulses(session.stim):
            print(format_audit(audit_stimulation(session.recording, stim)))
    return 0


def _detect(args: argparse.Namespace) -> int:
    from meagraph.detect import DetectionConfig, detect_spikes, save_detection
    from meagraph.detect.store import default_detection_folder

    config = DetectionConfig(n_jobs=args.n_jobs)
    for path in _recordings(args.paths):
        start = time.time()
        print(Path(path).name)
        session = _session(path, args)
        result = detect_spikes(session.recording, session.stim, config)
        out = Path(args.out) if args.out else default_detection_folder(path)
        save_detection(result, out, inputs=[path], overwrite=args.overwrite)
        n = result.trains.n_spikes()
        print(
            f"  {int(n.sum())} spikes on {int((n > 0).sum())} channels in "
            f"{time.time() - start:.0f} s; active channels (QC): {list(result.active_channels) or 'none'}"
        )
        if result.excluded:
            print(f"  excluded: {result.excluded}")
        if session.stim and all(s.sites is None for s in session.stim):
            print("  note: no --stim-site or stimulation log, so no stimulating electrode was excluded or extra-blanked")
        print(f"  wrote {out}")
    return 0


def _graph(args: argparse.Namespace) -> int:
    import numpy as np

    from meagraph.connectivity.pipeline import GraphConfig, graphs_from_detection, save_burst_controlled
    from meagraph.detect.store import find_detection, load_detection

    path = Path(args.path)
    folder = path if path.is_dir() else (Path(args.spikes) if args.spikes else find_detection(path))
    if folder is None:
        raise SystemExit(f"no detection results for {path.name}; run `meagraph detect` first")
    detection = load_detection(folder)
    config = GraphConfig(channels=args.channels, exclude_stim_post_ms=args.exclude_stim_ms)
    chosen = detection.active_channels if args.channels == "active" else detection.trains.unit_ids
    if len(chosen) < 2:
        print(f"{folder.parent.name}: {len(chosen)} {args.channels} channel(s); at least 2 are needed")
        if args.channels == "active":
            qc = detection.qc
            print(f"  no graph: {int(qc.active.sum())} of {len(qc.channel_ids)} electrodes passed the signal-quality check "
                  f"(negative crossings must clearly outnumber positive ones; see channels.csv, columns qc_negative and qc_positive)")
            print(f"  median noise {np.median(list(detection.noise_uv.values())):.1f} uV; "
                  f"{int(qc.n_negative.sum())} negative vs {int(qc.n_positive.sum())} positive events over all electrodes")
            print("  `--channels all` builds a graph from every electrode anyway, but without the check its edges may be noise")
        return 0
    graphs = graphs_from_detection(detection, methods=args.method, config=config)
    print(f"{folder.parent.name}: {len(chosen)} {args.channels} channels {list(chosen)}")
    if detection.stim and config.exclude_stim_post_ms > 0:
        n_pulses = sum(s.n for s in _pulses(detection.stim))
        print(f"  excluded {n_pulses} stimulation pulses (pulse to +{config.exclude_stim_post_ms:g} ms)")
    for method, bc in graphs.items():
        out = save_burst_controlled(bc, folder.parent / f"graph_{method}", config, inputs=[folder / "spikes.npz"], overwrite=True)
        n_all, n_quiet, n_robust = (int(r.significant.sum()) // (1 if r.directed else 2) for r in (bc.all, bc.no_bursts, bc.robust))
        print(f"  {method}: {int(bc.all.tested.sum()) // (1 if bc.all.directed else 2)} pairs tested; significant: "
              f"{n_all} all spikes, {n_quiet} without {len(bc.bursts)} network bursts, {n_robust} robust -> {out}")  # fmt: skip
        for e in bc.robust.edges()[:5]:
            arrow = "->" if bc.robust.directed else "--"
            delay = f", {e['delay_ms']:.2f} ms" if math.isfinite(e["delay_ms"]) else ""
            print(f"      {e['source']} {arrow} {e['target']}: weight {e['weight']:.3f}{delay}, p {e['p_value']:.2g}")
    return 0


def _evoked(args: argparse.Namespace) -> int:
    from meagraph.detect.store import detection_folder_for, load_detection
    from meagraph.evoked import EvokedConfig, evoked_from_detection, save_evoked

    folder = Path(args.spikes) if args.spikes else detection_folder_for(args.path)
    if folder is None:
        raise SystemExit(f"no detection results for {Path(args.path).name}; run `meagraph detect` first")
    detection = load_detection(folder)
    if not any(s.n for s in detection.stim):
        raise SystemExit(f"{folder.parent.name}: no stimulation pulses")
    config = EvokedConfig(channels=args.channels, bins_ms=tuple(args.bins_ms), active_gap_s=args.active_gap_s,
                          n_permutations=args.permutations, min_trials=args.min_trials)  # fmt: skip
    res = evoked_from_detection(detection, config)
    out = save_evoked(res, folder.parent / "evoked", inputs=[folder / "spikes.npz"], overwrite=True)

    lo, hi = config.window_ms
    states = {s: int((res.shams.state == s).sum()) for s in ("silent", "active", "in_burst")}
    print(f"{folder.parent.name}: {len(res.pulses.onset_s)} pulses at {len(res.sites)} sites, "
          f"{len(res.channels)} {args.channels} channels, {len(res.bursts)} network bursts")
    print(f"  window {lo:g}-{hi:g} ms after onset (later on electrodes still blanked; see channels.csv); "
          f"{len(res.shams.onset_s)} sham times {states}")
    if res.n_dropped_pre:
        print(f"  dropped {res.n_dropped_pre} spikes in the {config.drop_pre_ms:g} ms before pulses (pulse-locked artifact, D26)")
    print(f"\n  evoked burst onsets ({config.burst_from_ms:g}-{hi:g} ms), pulses vs shams in the same state:")
    print(f"    {'site':>6}  {'all':>22}  {'silent':>14}  {'active':>14}  {'in burst':>14}")
    for site in res.sites:
        rows = {r["state"]: r for r in res.site_table if r["site"] == site}
        a = rows["all"]
        cells = [f"{rows[s]['n_evoked']}/{rows[s]['n_pulses']} ({rows[s]['p_sham']:.0%})" if rows[s]["n_pulses"] else "-"
                 for s in ("silent", "active", "in_burst")]  # fmt: skip
        print(f"    {site:>6}  {a['n_evoked']:>3}/{a['n_pulses']:<3} vs {a['p_sham']:5.1%} p {a['p_value']:<6.2g}"
              f"  {cells[0]:>14}  {cells[1]:>14}  {cells[2]:>14}")  # fmt: skip
    sig = [r for r in res.response_table if r["q_value"] < config.alpha]
    print(f"\n  spike responses outside bursts (q < {config.alpha:g}): {len(sig)} site-electrode pairs")
    for r in sorted(sig, key=lambda r: r["q_value"])[:10]:
        print(f"    {r['site']:>6} -> {r['channel']:<4} {r['mean_spikes']:.2f} vs {r['mean_spikes_sham']:.2f} spikes, "
              f"first spike {r['median_first_ms']:.0f} ms, q {r['q_value']:.2g}")  # fmt: skip
    print("\n  decoding the stimulated site from responses (cross-validated balanced accuracy):")
    for task, d in res.decoding.items():
        if task == "site_pairs":
            continue
        if "skipped" in d:
            print(f"    {task}: skipped ({d['skipped']}); trials {dict(zip(d['classes'], d['n_trials']))}")
            continue
        print(f"    {task}: {d['balanced_accuracy']:.2f} vs chance {d['chance']:.2f} (permutation mean "
              f"{d['null_mean']:.2f}, p {d['p_value']:.2g}); {len(d['classes'])} classes, {d['n_folds']} folds")
        print("      recall: " + ", ".join(f"{c} {v:.2f}" for c, v in d["recall"].items()))
    pairs = res.decoding.get("site_pairs", [])
    if pairs:
        sites = sorted({r["a"] for r in pairs} | {r["b"] for r in pairs})
        acc = {(r["a"], r["b"]): r for r in pairs} | {(r["b"], r["a"]): r for r in pairs}
        print(f"  pairwise site decoding (balanced accuracy; * q < {config.alpha:g}):")
        print("    " + " " * 6 + "".join(f"{s:>7}" for s in sites))
        for a in sites:
            cells = [("      -" if a == b else f"{acc[a, b]['balanced_accuracy']:6.2f}{'*' if acc[a, b]['q_value'] < config.alpha else ' '}")
                     for b in sites]  # fmt: skip
            print(f"    {a:>6}" + "".join(cells))
    top = sorted((r for r in res.channel_table if r["q_value"] == r["q_value"]), key=lambda r: r["p_value"])[:8]
    if top:
        print("  electrodes that best separate sites (Kruskal-Wallis): "
              + ", ".join(f"{r['channel']} (q {r['q_value']:.2g})" for r in top))
    small = [s for s in res.sites if (res.pulses.label == s).sum() < config.min_trials]
    if small:
        print(f"  not decoded (< {config.min_trials} pulses): {small}")
    print(f"  wrote {out}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from meagraph.benchmark import inhibition_scenarios, run_benchmark, scenarios, summarize, write_rows
    from meagraph.synth import NetworkConfig

    progress = lambda message: print(message, flush=True)  # noqa: E731
    if args.quick:
        small = NetworkConfig(n_units=8)
        todo = scenarios(durations_s=(300,), weights=(0.1,), base=small) + inhibition_scenarios(durations_s=(300,), bursts=(False,), base=small)[:1]
        overrides = {m: {"n_surrogates": 200} for m in ("cch_jitter", "sttc", "tspe", "cfp")}
        rows = run_benchmark(todo, seeds=(0,), method_overrides=overrides, n_jobs=args.n_jobs, progress=progress)
    else:
        # TSPE and CFP use 200 surrogates here: their p-values come from a fitted normal (D17), and
        # 1000 would make the run several hours. Real-data defaults stay at 1000.
        overrides = {m: {"n_surrogates": 200} for m in ("tspe", "cfp")}
        rows = run_benchmark(scenarios() + inhibition_scenarios(), method_overrides=overrides, n_jobs=args.n_jobs, progress=progress)
    out = write_rows(rows, Path(args.out) / "benchmark.csv")
    for spikes in ("all", "robust"):
        print(f"\n{spikes} spikes:")
        for r in summarize(rows, spikes):
            inh = f"  inh recall {r['inh_recall']:.2f}" if r["inh_recall"] == r["inh_recall"] else ""
            print(f"  {r['scenario']:<26} {r['method']:<14} precision {r['precision']:.2f}  recall {r['recall']:.2f}  "
                  f"FPR {r['false_positive_rate']:.4f}  AUC {r['auc']:.2f}  delay err {r['delay_error_ms']:.2f} ms{inh}")  # fmt: skip
    print(f"\nwrote {out}")
    return 0


def _plot(args: argparse.Namespace) -> int:
    try:
        import matplotlib

        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # matplotlib is an optional extra
        raise SystemExit(f"plotting needs matplotlib: uv pip install -e '.[viewer]' ({exc})") from exc
    from meagraph import viz
    from meagraph.connectivity import cross_correlograms
    from meagraph.connectivity.pipeline import load_graph

    path = Path(args.path)
    if path.suffix == ".h5":  # a recording (the file itself is not read): its graph folder under results/
        path = path.parent / "results" / path.stem / f"graph_{args.method}"
    elif (path / f"graph_{args.method}").is_dir():  # results/<recording>
        path = path / f"graph_{args.method}"
    if not (path / "result.json").exists():
        path = path / args.spike_set
    if not (path / "result.json").exists():
        raise SystemExit(f"no graph at {path}; run `meagraph graph` first")
    saved = load_graph(path)
    res = saved.result
    params = res.params
    ccg, lags = cross_correlograms(saved.trains, params.get("bin_ms", 0.5), params.get("max_lag_ms", 30.0))
    recording = path.parent.parent.name
    fig = viz.plot_connectivity(res, ccg, lags, background_um=saved.detection.trains.positions_um,
                                background_ids=saved.detection.trains.channel_ids or saved.detection.trains.unit_ids,
                                title=f"{recording[:60]}\n{path.parent.name}/{path.name}")  # fmt: skip
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"saved {args.save}")
    else:
        plt.show()
    return 0


def _view(args: argparse.Namespace) -> int:
    try:
        from meagraph.viewer import main as viewer_main
    except ImportError as exc:  # matplotlib is an optional extra
        raise SystemExit(f"the viewer needs matplotlib: uv pip install -e '.[viewer]' ({exc})") from exc
    return viewer_main(args)


def build_parser() -> argparse.ArgumentParser:
    from meagraph.probe import DEFAULT_PROBE

    parser = argparse.ArgumentParser(prog="meagraph", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def session_args(p):
        p.add_argument("--probe", default=DEFAULT_PROBE, help="probe spec name or YAML path (see `meagraph probes`)")
        p.add_argument("--stim-site", action="append", help="stimulated electrode: '47', or 'STG 1=47' per output")
        p.add_argument("--stimlog", default="auto",
                       help="stimulation log with per-pulse sites: 'auto' (default) finds <tag>.stimlog.csv next to the recording; a path; or 'none'")

    p = sub.add_parser("info", help="summarise the streams, events and spike streams in MCS .h5 files")
    p.add_argument("paths", nargs="+")
    p.add_argument("--recording-index", type=int, default=0)
    p.set_defaults(func=_info)

    p = sub.add_parser("probes", help="list the packaged probe specifications")
    p.set_defaults(func=_probes)

    p = sub.add_parser("audit", help="stimulation audit: pulse structure, site, artifact recovery")
    p.add_argument("paths", nargs="+")
    session_args(p)
    p.set_defaults(func=_audit)

    p = sub.add_parser("detect", help="detect spikes and save them with QC, config and provenance")
    p.add_argument("paths", nargs="+")
    session_args(p)
    p.add_argument("--out", help="output folder (default: results/<recording>/detect next to the file)")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_detect)

    p = sub.add_parser("graph", help="connectivity graphs from detected spikes, with the network-burst control")
    p.add_argument("path", help="recording (.h5; uses its newest detection) or a detection folder")
    p.add_argument("--spikes", help="detection folder, if not the newest one")
    p.add_argument("--method", nargs="+", default=["cch_jitter", "cch_hollow", "sttc", "tspe", "cfp"])
    p.add_argument("--channels", choices=["active", "all"], default="active", help="QC-active channels only (default)")
    p.add_argument("--exclude-stim-ms", type=float, default=200.0,
                   help="drop spikes from each pulse to this long after it (stimulation recordings; 0 keeps them)")
    p.set_defaults(func=_graph)

    p = sub.add_parser("evoked", help="evoked responses per stimulation site: bursts vs sham, electrode responses, site decoding")
    p.add_argument("path", help="recording (.h5; the file need not exist), results/<recording>, or a detection folder")
    p.add_argument("--spikes", help="detection folder, if not the default one")
    p.add_argument("--channels", choices=["active", "all"], default="active", help="readout electrodes (default: QC-active)")
    p.add_argument("--bins-ms", type=float, nargs="+", default=[12.0, 30.0, 60.0, 100.0, 150.0],
                   help="response window and decoder bins: edges in ms after pulse onset")
    p.add_argument("--active-gap-s", type=float, default=5.0,
                   help="a pulse is in an 'active' period if the last burst ended less than this long before it")
    p.add_argument("--permutations", type=int, default=200, help="label permutations for the decoding p-value")
    p.add_argument("--min-trials", type=int, default=5, help="sites with fewer pulses are not decoded")
    p.set_defaults(func=_evoked)

    p = sub.add_parser("plot", help="draw a saved graph in 3D with its top 4 correlograms; click an electrode for its own")
    p.add_argument("path", help="results/<recording>, its graph_<method> folder, or one of all/no_bursts/robust inside it "
                                  "(a recording's .h5 path also works, and the file need not exist)")
    p.add_argument("--method", default="cch_jitter", help="graph to show when PATH is a recording")
    p.add_argument("--spike-set", choices=["all", "no_bursts", "robust"], default="all",
                   help="all spikes, network bursts removed, or edges significant in both")
    p.add_argument("--save", help="write an image instead of opening a window")
    p.set_defaults(func=_plot)

    p = sub.add_parser("benchmark", help="score the estimators on synthetic networks with known connections")
    p.add_argument("--quick", action="store_true", help="one small scenario per condition (seconds)")
    p.add_argument("--out", default="results/benchmark")
    p.add_argument("--n-jobs", type=int, default=4, help="simulated networks processed in parallel")
    p.set_defaults(func=_benchmark)

    p = sub.add_parser("view", help="interactive viewer with the 4x4x4 electrode selector")
    p.add_argument("path", help="recording (.h5), or results/<recording> or its detect folder (works without the .h5; "
                                "voltage traces need it)")
    session_args(p)
    p.add_argument("--spikes", help="detection folder (default: newest under results/<recording>/)")
    p.add_argument("--electrode", default=None)
    p.add_argument("--t0", type=float, default=None, help="window start, s on the recording clock")
    p.add_argument("--win", type=float, default=2.0)
    p.add_argument("--stream", default="auto", help="analog stream shown in the trace panel (default: the one analysed)")
    p.add_argument("--bandpass", action="store_true")
    p.add_argument("--save", help="render to an image instead of opening a window")
    p.set_defaults(func=_view)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with warnings.catch_warnings():
        # Placeholder geometry is flagged once by `meagraph probes`; repeating it on every command is noise.
        from meagraph.probe import UnverifiedGeometryWarning

        warnings.simplefilter("ignore", UnverifiedGeometryWarning)
        return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
