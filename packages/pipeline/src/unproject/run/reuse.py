"""Checking that an artifact is still all there before reusing it.

A descriptor is a claim about the files beside it, and a stage that trusts the
claim will happily build on a frame the disk never finished writing. Every
reuse path checks the claim first.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def unusable_reason(directory: Path, expected: Iterable[tuple[str, int]]) -> str | None:
    """Say why the artifact in `directory` cannot be reused, or None if it can.

    `expected` pairs each path the descriptor references, relative to the
    artifact directory, with the size it says that file has.
    """
    for relative, byte_length in expected:
        try:
            actual = (directory / relative).stat().st_size
        except OSError:
            return f"{relative} is referenced by the artifact but is not on disk"
        if actual != byte_length:
            return (
                f"{relative} is {actual} bytes on disk, but the artifact says "
                f"{byte_length}; the file is truncated or was replaced"
            )
    return None
