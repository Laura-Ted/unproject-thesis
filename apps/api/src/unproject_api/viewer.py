"""Keeping the served viewer bundle up to date.

The API can serve the built viewer, which is what makes running the whole thing
one command. That is only true if the build happens by itself: a bundle quietly
older than the source is worse than no bundle, because it looks like the code
did not work rather than like it was not rebuilt.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Everything that changes what the bundle should contain.
_SOURCES = ("src", "index.html", "package.json", "vite.config.ts", "tsconfig.app.json")

_BUILD_TIMEOUT_SECONDS = 300


def newest_change(web: Path) -> float:
    """When the viewer's sources last changed."""
    newest = 0.0
    for name in _SOURCES:
        path = web / name
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    newest = max(newest, child.stat().st_mtime)
    return newest


def is_stale(web: Path) -> bool:
    """Whether the bundle is missing or older than what it was built from."""
    built = web / "dist" / "index.html"
    if not built.is_file():
        return True
    return built.stat().st_mtime < newest_change(web)


def ensure_built(web: Path) -> bool:
    """Build the viewer if it needs it. Returns whether a bundle is there.

    A failure is reported and survived rather than raised: the API is still
    useful without the viewer, and `pnpm dev` is still a way to get one.
    """
    if not (web / "package.json").is_file():
        return False
    if not is_stale(web):
        return True

    pnpm = shutil.which("pnpm")
    if pnpm is None:
        logger.warning("the viewer needs rebuilding but pnpm is not installed")
        return (web / "dist" / "index.html").is_file()

    logger.info("building the viewer, one moment")
    try:
        result = subprocess.run(
            [pnpm, "build"],
            cwd=web,
            capture_output=True,
            text=True,
            timeout=_BUILD_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("could not build the viewer: %s", error)
        return (web / "dist" / "index.html").is_file()

    if result.returncode != 0:
        # The compiler's own words, not a summary of them.
        logger.warning("the viewer did not build:\n%s", (result.stderr or result.stdout).strip())
        return (web / "dist" / "index.html").is_file()

    logger.info("viewer built")
    return True
