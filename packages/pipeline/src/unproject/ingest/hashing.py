"""Content hashing shared by the frame sources.

Two sources computing input identity differently would make their artifacts
incomparable, so the digest lives in one place.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Chunked so hashing a multi-gigabyte capture does not load it into memory.
_CHUNK_BYTES = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
