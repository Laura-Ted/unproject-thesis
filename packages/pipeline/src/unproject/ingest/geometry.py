"""The frame geometry of `IngestConfig.geometry`, in exact integer arithmetic.

The matrix suite of protocol v2 compares one sequence at four geometries:
16:9 and letterboxed to a square, each at full and at half resolution. Those
variants used to be transcodes carried from machine to machine, because
libx264 is not bit-stable across builds. Here they are derived from the
decoded source frames with two operations that have one answer on every
machine: a box average over integer blocks and a zero-filled pad. What a
machine writes is then a function of the source bytes and the config alone,
which is what the ingest hash already claims.
"""

from __future__ import annotations

import numpy as np

from unproject.config import FrameGeometry
from unproject.ingest.types import ImageArray


class GeometryError(ValueError):
    """The frame cannot take the requested geometry."""


def apply_geometry(image: ImageArray, geometry: FrameGeometry) -> ImageArray:
    """Downscale, then letterbox, as the config says; the input is untouched."""
    out = image
    if geometry.downscale > 1:
        out = downscale(out, geometry.downscale)
    if geometry.letterbox_square:
        out = letterbox_square(out)
    return out


def downscale(image: ImageArray, factor: int) -> ImageArray:
    """Box-average `factor` x `factor` blocks, rounded half up, in integers.

    Requires the frame to divide by the factor: a 1920x1080 frame halves to
    960x540 exactly, and anything that would need resampling is refused
    rather than approximated differently on different machines.
    """
    height, width = image.shape[:2]
    if factor < 1:
        raise GeometryError(f"downscale factor must be at least 1, got {factor}")
    if height % factor or width % factor:
        raise GeometryError(
            f"a {width}x{height} frame does not divide by {factor}; the geometry has to be exact"
        )
    if factor == 1:
        return image
    channels = image.shape[2] if image.ndim == 3 else 1
    blocks = image.reshape(height // factor, factor, width // factor, factor, channels)
    total = blocks.sum(axis=(1, 3), dtype=np.uint32)
    count = factor * factor
    averaged = (total + count // 2) // count
    result = averaged.astype(np.uint8)
    return result if image.ndim == 3 else result.reshape(height // factor, width // factor)


def letterbox_square(image: ImageArray) -> ImageArray:
    """Pad with black to a square of the long edge, the content centred.

    An odd difference puts the extra row (or column) at the bottom (or right),
    matching `ffmpeg pad=W:H:x:y` with the offset rounded down.
    """
    height, width = image.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    padding = [(top, side - height - top), (left, side - width - left)]
    if image.ndim == 3:
        padding.append((0, 0))
    return np.pad(image, padding, mode="constant", constant_values=0)


def output_size(width: int, height: int, geometry: FrameGeometry | None) -> tuple[int, int]:
    """The written frame size for a source size, without touching pixels."""
    if geometry is None:
        return width, height
    if geometry.downscale > 1:
        if width % geometry.downscale or height % geometry.downscale:
            raise GeometryError(
                f"a {width}x{height} frame does not divide by {geometry.downscale}; "
                "the geometry has to be exact"
            )
        width //= geometry.downscale
        height //= geometry.downscale
    if geometry.letterbox_square:
        side = max(width, height)
        width = height = side
    return width, height
