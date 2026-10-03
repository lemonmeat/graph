"""Command line entry point: ``meagraph <command>``. Commands are thin wrappers over the library."""

from __future__ import annotations

import argparse
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
