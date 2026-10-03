"""Command line entry point: ``meagraph <command>``. Commands are thin wrappers over the library."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _info(args: argparse.Namespace) -> int:
    from meagraph.io.inventory import format_inventory, inspect_file

    for i, path in enumerate(args.paths):
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meagraph", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="summarise the streams, events and spike streams in MCS .h5 files")
    p.add_argument("paths", nargs="+")
    p.add_argument("--recording-index", type=int, default=0)
    p.set_defaults(func=_info)

    p = sub.add_parser("probes", help="list the packaged probe specifications")
    p.set_defaults(func=_probes)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
