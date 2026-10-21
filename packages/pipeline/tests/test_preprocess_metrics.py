"""Tests for the sharpness and disparity metrics.

The inputs are built so the answer is known: a translation of N pixels should
measure N, and a blurred copy should measure less sharp than the original.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from unproject.preprocess.metrics import (
    GrayArray,
    disparity,
    sharpness,
    to_working_gray,
    working_scale,
)


def _textured(width: int = 320, height: int = 240) -> GrayArray:
    """A frame with enough corners for the tracker to work with."""
    rng = np.random.default_rng(seed=0)
    noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    # Smoothed slightly, or the corners are just per-pixel noise.
    return cast(GrayArray, cv2.GaussianBlur(noise, (3, 3), 0))


def test_blurring_lowers_sharpness() -> None:
    crisp = _textured()
    blurred = cast(GrayArray, cv2.GaussianBlur(crisp, (21, 21), 0))

    assert sharpness(blurred) < sharpness(crisp)


def test_sharpness_of_a_flat_image_is_near_zero() -> None:
    flat = np.full((240, 320), 128, dtype=np.uint8)

    assert sharpness(flat) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("shift", [4, 10, 25])
def test_disparity_measures_a_known_translation(shift: int) -> None:
    reference = _textured()
    candidate: GrayArray = np.roll(reference, shift=shift, axis=1)

    measured = disparity(reference, candidate)

    assert measured.median_px == pytest.approx(float(shift), abs=1.0)
    assert measured.tracked_ratio > 0.5


def test_disparity_of_an_identical_frame_is_zero() -> None:
    reference = _textured()

    measured = disparity(reference, reference)

    assert measured.median_px == pytest.approx(0.0, abs=0.5)
    assert measured.tracked_ratio > 0.9


def test_disparity_reports_zero_tracking_on_a_textureless_frame() -> None:
    flat: GrayArray = np.full((240, 320), 200, dtype=np.uint8)

    measured = disparity(flat, flat)

    # goodFeaturesToTrack returns None here, and the caller needs a number back.
    assert measured.tracked_ratio == 0.0
    assert measured.median_px == 0.0


def test_tracking_falls_as_content_leaves_the_frame() -> None:
    reference = _textured()

    ratios = [
        disparity(reference, _translated(reference, dx)).tracked_ratio for dx in (20, 160, 260)
    ]

    # Points that leave the frame stop being tracked, so the ratio stands in
    # for how far the pair has drifted. min_tracked_ratio keys off this.
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] < 0.3


def test_tracking_does_not_detect_a_substituted_scene() -> None:
    """A known limitation, pinned here so it does not get assumed away.

    Lucas-Kanade's status flag reports convergence, not correctness. On an
    unrelated frame most points still converge somewhere, so the ratio stays
    high while the displacement means nothing.
    """
    reference = _textured()
    rng = np.random.default_rng(seed=99)
    unrelated: GrayArray = rng.integers(0, 256, size=reference.shape, dtype=np.uint8)

    measured = disparity(reference, unrelated)

    assert measured.tracked_ratio > 0.5
    # The blur floor is what rejects such frames in practice; a blown-out or
    # empty one has near-zero sharpness long before tracking notices.
    assert sharpness(np.full_like(reference, 128)) < sharpness(reference)


def _translated(gray: GrayArray, dx: int) -> GrayArray:
    """Shift right with an empty border, so content leaves the frame.

    np.roll wraps it around instead, and nothing becomes untrackable.
    """
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, 0.0]], dtype=np.float32)
    height, width = gray.shape
    return cast(GrayArray, cv2.warpAffine(gray, matrix, (width, height)))


def test_working_scale_caps_the_long_edge() -> None:
    assert working_scale(3840, 2160, 480) == pytest.approx(480 / 3840)
    assert working_scale(2160, 3840, 480) == pytest.approx(480 / 3840)


def test_working_scale_never_upscales() -> None:
    assert working_scale(320, 240, 480) == 1.0


def test_to_working_gray_downscales_and_flattens_channels() -> None:
    colour: NDArray[np.uint8] = np.zeros((480, 640, 3), dtype=np.uint8)

    gray = to_working_gray(colour, long_edge=320)

    assert gray.ndim == 2
    assert max(gray.shape) == 320
