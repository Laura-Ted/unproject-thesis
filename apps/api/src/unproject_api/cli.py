"""Command line entry point for the API.

Everything the server needs is either a path or comes out of an experiment
config, so a running server can be described by the command that started it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from unproject_api.app import build_settings, create_app
from unproject_api.viewer import ensure_built

DEFAULT_CONFIG = Path("experiments/crowd.toml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unproject-api", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(), help="repository root")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "experiment config supplying the bodies settings; live runs are only "
            f"reproducible if this is a versioned file (default: {DEFAULT_CONFIG})"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="do not rebuild the viewer even if its sources are newer than the bundle",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    settings = build_settings(args.root, args.config)
    # Before the app is created, since that is what decides whether there is a
    # bundle to mount.
    if not args.no_build:
        ensure_built(settings.root / "apps" / "web")
    app = create_app(settings)
    logging.getLogger("unproject_api").info(
        "open http://%s:%d", "localhost" if args.host == "127.0.0.1" else args.host, args.port
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
