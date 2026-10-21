"""Sparse point cloud I/O.

Both backends produce points and stage [4] initialises from them, so the format
lives here rather than inside either backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Points = NDArray[np.float64]
Colors = NDArray[np.uint8]

_HEADER = """ply
format binary_little_endian 1.0
element vertex {count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

# float32 positions and uchar colours, which is what every splat viewer reads.
_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


class PointCloudError(Exception):
    """Raised when a point cloud cannot be written or read back."""


def write_ply(path: Path, points: Points, colors: Colors) -> int:
    """Write a binary little-endian PLY and return the point count."""
    positions = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(positions) != len(rgb):
        raise PointCloudError(f"{len(positions)} points against {len(rgb)} colours")

    vertices = np.empty(len(positions), dtype=_VERTEX_DTYPE)
    vertices["x"], vertices["y"], vertices["z"] = positions.T.astype(np.float32)
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(_HEADER.format(count=len(vertices)).encode("ascii"))
        handle.write(vertices.tobytes())
    return len(vertices)


def read_ply(path: Path) -> tuple[Points, Colors]:
    """Read back what `write_ply` wrote.

    Deliberately narrow: it parses this writer's own header rather than PLY in
    general, and says so instead of guessing when it sees something else.
    """
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise PointCloudError(f"{path} has no PLY header")

    header = raw[:end].decode("ascii", errors="replace")
    if "binary_little_endian" not in header:
        raise PointCloudError(f"{path} is not in the binary little-endian layout we write")

    count = _vertex_count(header, path)
    body = raw[end + len(marker) :]
    expected = count * _VERTEX_DTYPE.itemsize
    if len(body) != expected:
        raise PointCloudError(f"{path} declares {count} vertices but holds {len(body)} bytes")

    vertices = np.frombuffer(body, dtype=_VERTEX_DTYPE, count=count)
    positions = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1, dtype=np.float64)
    rgb = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=-1)
    return positions, rgb.astype(np.uint8)


def _vertex_count(header: str, path: Path) -> int:
    for line in header.splitlines():
        if line.startswith("element vertex "):
            return int(line.split()[-1])
    raise PointCloudError(f"{path} declares no vertex count")
