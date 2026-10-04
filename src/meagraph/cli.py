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


def _recordings(paths: list[str]) -> list[str]:
    """Drop legacy ``*.spikes.h5`` sidecars that shell globs like ``*.h5`` pick up."""
    keep = [p for p in paths if not p.endswith(".spikes.h5")]
    for p in sorted(set(paths) - set(keep)):
        print(f"skipping {Path(p).name} (spike sidecar, not a recording)")
    return keep


def _info(args: argparse.Namespace) -> int:
    from meagraph.io.inventory import format_inventory, inspect_file

    for i, path in enumerate(_recordings(args.paths)):
        if i:
            print()
        print(format_inventory(inspect_file(path, args.recording_index)))
    return 0


def _probes(args: argparse.Namespace) -> int:
    from meagraph.probe.spec import available_probe_specs, load_probe_spec

    for name in available_probe_specs():
        spec = load_probe_spec(name)
        flag = "" if spec.geometry_verified else "  [placeholder geometry]"
        print(f"{name}: {len(spec.contacts)} contacts, {spec.ndim}D{flag}\n    {spec.description}")
    return 0


def _audit(args: argparse.Namespace) -> int:
    from meagraph.io import load_session
    from meagraph.stimulation import audit_stimulation, format_audit

    for path in _recordings(args.paths):
        session = load_session(path, probe=args.probe, stim_site=_stim_site(args.stim_site))
        print(Path(path).name)
        if not session.stim:
            print("  no stimulation events")
        for stim in session.stim:
            print(format_audit(audit_stimulation(session.recording, stim)))
    return 0


def _detect(args: argparse.Namespace) -> int:
    from meagraph.detect import PROFILES, detect_spikes, save_detection
    from meagraph.detect.store import default_detection_folder
    from meagraph.io import load_session

    config = PROFILES[args.profile].model_copy(update={"n_jobs": args.n_jobs})
    for path in _recordings(args.paths):
        start = time.time()
        session = load_session(path, probe=args.probe, stim_site=_stim_site(args.stim_site))
        result = detect_spikes(session.recording, session.stim, config)
        out = Path(args.out) if args.out else default_detection_folder(path, args.profile)
        save_detection(result, out, inputs=[path], overwrite=args.overwrite)
        n = result.trains.n_spikes()
        print(
            f"{Path(path).name}\n  {int(n.sum())} spikes on {int((n > 0).sum())} channels in "
            f"{time.time() - start:.0f} s; active channels (QC): {list(result.active_channels) or 'none'}"
        )
        if result.excluded:
            print(f"  excluded: {result.excluded}")
        if session.stim and all(s.site is None for s in session.stim) and config.exclude_stim_site:
            print("  note: no --stim-site given, so no stimulating electrode was excluded")
        print(f"  wrote {out}")
    return 0


def _graph(args: argparse.Namespace) -> int:
    from meagraph.connectivity.pipeline import graphs_from_detection, save_burst_controlled, stimulation_periods
    from meagraph.detect.store import find_detection, load_detection
    from meagraph.io import read_stim_events

    path = Path(args.path)
    folder = path if path.is_dir() else (Path(args.spikes) if args.spikes else find_detection(path))
    if folder is None:
        raise SystemExit(f"no detection results for {path.name}; run `meagraph detect` first")
    detection = load_detection(folder)
    chosen = detection.active_channels if args.channels == "active" else detection.trains.unit_ids
    if len(chosen) < 2:
        print(f"{folder}: {len(chosen)} {args.channels} channel(s); at least 2 are needed")
        return 0
    stim = read_stim_events(path) if path.is_file() else []
    excluded = stimulation_periods(stim, post_ms=args.exclude_stim_ms) if stim and args.exclude_stim_ms > 0 else None
    graphs = graphs_from_detection(detection, methods=args.method, probe=args.probe, channels=args.channels, exclude_periods=excluded)
    print(f"{folder.parent.name}: {len(chosen)} {args.channels} channels {list(chosen)}")
    if excluded is not None:
        print(f"  excluded {len(excluded)} stimulation periods (pulse to +{args.exclude_stim_ms:g} ms)")
    elif path.is_dir():
        print("  note: given a detection folder, so stimulation periods (if any) were not excluded")
    for method, bc in graphs.items():
        out = save_burst_controlled(bc, folder.parent / f"graph_{method}", overwrite=True)
        n_all, n_quiet, n_robust = (int(r.significant.sum()) // (1 if r.directed else 2) for r in (bc.all, bc.no_bursts, bc.robust))
        print(f"  {method}: {int(bc.all.tested.sum()) // (1 if bc.all.directed else 2)} pairs tested; significant: "
              f"{n_all} all spikes, {n_quiet} without {len(bc.bursts)} network bursts, {n_robust} robust -> {out}")  # fmt: skip
        for e in bc.robust.edges()[:5]:
            arrow = "->" if bc.robust.directed else "--"
            delay = f", {e['delay_ms']:.2f} ms" if math.isfinite(e["delay_ms"]) else ""
            print(f"      {e['source']} {arrow} {e['target']}: weight {e['weight']:.3f}{delay}, p {e['p_value']:.2g}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from meagraph.benchmark import run_benchmark, scenarios, summarize, write_rows
    from meagraph.synth import NetworkConfig

    if args.quick:
        rows = run_benchmark(scenarios(durations_s=(300,), weights=(0.1,), base=NetworkConfig(n_units=8)), seeds=(0,),
                             method_overrides={m: {"n_surrogates": 200} for m in ("cch_jitter", "sttc")})  # fmt: skip
    else:
        rows = run_benchmark(scenarios())
    out = write_rows(rows, Path(args.out) / "benchmark.csv")
    for spikes in ("all", "robust"):
        print(f"\n{spikes} spikes:")
        for r in summarize(rows, spikes):
            print(f"  {r['scenario']:<26} {r['method']:<11} precision {r['precision']:.2f}  recall {r['recall']:.2f}  "
                  f"FPR {r['false_positive_rate']:.4f}  AUC {r['auc']:.2f}  delay err {r['delay_error_ms']:.2f} ms")  # fmt: skip
    print(f"\nwrote {out}")
    return 0


def _view(args: argparse.Namespace) -> int:
    try:
        from meagraph.viewer import main as viewer_main
    except ImportError as exc:  # matplotlib is an optional extra
        raise SystemExit(f"the viewer needs matplotlib: uv pip install -e '.[viewer]' ({exc})") from exc
    return viewer_main(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meagraph", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def session_args(p):
        p.add_argument("--probe", default="cube4x4x4_E-00303", help="probe spec name or YAML path (see `meagraph probes`)")
        p.add_argument("--stim-site", action="append", help="stimulated electrode: '47', or 'STG 1=47' per output")

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
    p.add_argument("--profile", choices=["default", "legacy"], default="default")
    p.add_argument("--out", help="output folder (default: results/<recording>/detect_<profile> next to the file)")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_detect)

    p = sub.add_parser("graph", help="connectivity graphs from detected spikes, with the network-burst control")
    p.add_argument("path", help="recording (.h5; uses its newest detection) or a detection folder")
    p.add_argument("--spikes", help="detection folder, if not the newest one")
    p.add_argument("--method", nargs="+", default=["cch_jitter", "cch_hollow", "sttc"])
    p.add_argument("--channels", choices=["active", "all"], default="active", help="QC-active channels only (default)")
    p.add_argument("--probe", default="cube4x4x4_E-00303")
    p.add_argument("--exclude-stim-ms", type=float, default=200.0,
                   help="drop spikes from each pulse to this long after it (stimulation recordings; 0 keeps them)")
    p.set_defaults(func=_graph)

    p = sub.add_parser("benchmark", help="score the estimators on synthetic networks with known connections")
    p.add_argument("--quick", action="store_true", help="one small scenario per condition (about a minute)")
    p.add_argument("--out", default="results/benchmark")
    p.set_defaults(func=_benchmark)

    p = sub.add_parser("view", help="interactive viewer with the 4x4x4 electrode selector")
    p.add_argument("path")
    session_args(p)
    p.add_argument("--spikes", help="detection folder (default: newest under results/<recording>/)")
    p.add_argument("--electrode", default=None)
    p.add_argument("--t0", type=float, default=None, help="window start, s on the recording clock")
    p.add_argument("--win", type=float, default=2.0)
    p.add_argument("--stream", default="raw", help="analog stream shown in the trace panel")
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
