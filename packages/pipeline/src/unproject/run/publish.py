"""Transactional publication of a stage artifact.

A stage that writes into the directory a reader may be reusing can lie in two
ways: it can leave partial output that looks finished, and it can destroy a good
artifact while failing to produce its replacement. Building the whole directory
somewhere else and moving it into place at the end removes both.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


class PublishError(Exception):
    """Raised when a finished artifact cannot be moved into place."""


@contextmanager
def staged_artifact(target: Path, *, descriptor: str) -> Iterator[Path]:
    """Yield a directory to build in, then publish it as `target`.

    Nothing lands at `target` until the block finishes, so a concurrent reader
    sees either the previous complete artifact or nothing claiming to be one.

    A failed block still publishes, because the manifest it leaves behind is the
    record of what went wrong and belongs where the user will look for it. The
    exception is a `target` that already holds a complete artifact: that one is
    kept, and the failed attempt is dropped.
    """
    staging = _staging_dir(target)
    try:
        yield staging
    except BaseException:
        _discard(staging, target, descriptor)
        raise
    _swap(staging, target)


def _staging_dir(target: Path) -> Path:
    """A fresh sibling of `target`, so the final move stays within one filesystem."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid4().hex[:8]}"
    staging.mkdir()
    return staging


def _discard(staging: Path, target: Path, descriptor: str) -> None:
    """Deal with the staging directory of a stage that raised.

    Never raises: the stage's own exception is the one worth propagating, and
    the disk is one of the things that may have gone wrong.
    """
    try:
        if (target / descriptor).is_file():
            logger.info("keeping the complete artifact at %s; this run failed", target)
            shutil.rmtree(staging, ignore_errors=True)
            return
        _swap(staging, target)
    except (OSError, PublishError) as error:
        logger.warning("could not clean up %s: %s", staging, error)


def _swap(staging: Path, target: Path) -> None:
    """Move `staging` onto `target` on the same filesystem.

    A rename cannot replace a directory that has anything in it, so whatever is
    at `target` steps aside first and is only deleted once the new artifact has
    landed. If that second rename fails, the first is undone: a failed overwrite
    leaves the artifact that was already there.
    """
    if not target.exists():
        staging.replace(target)
        return

    superseded = target.parent / f".{target.name}.superseded-{uuid4().hex[:8]}"
    target.replace(superseded)
    try:
        staging.replace(target)
    except OSError as error:
        superseded.replace(target)
        shutil.rmtree(staging, ignore_errors=True)
        msg = f"could not publish {staging.name} as {target}, kept the previous artifact: {error}"
        raise PublishError(msg) from error
    shutil.rmtree(superseded, ignore_errors=True)
