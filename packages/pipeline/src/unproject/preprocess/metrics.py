"""The two measurements keyframe selection is built on.

Arrays in, numbers out. No config and no I/O, so these can be tested against
inputs whose answer is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

# Single-channel uint8, shape (height, width).
GrayArray = NDArray[np.uint8]

# 200 keeps the median stable without making tracking the cost of the stage.
_MAX_CORNERS = 200
_QUALITY_LEVEL = 0.01
_MIN_CORNER_DISTANCE = 8


@dataclass(frozen=True, slots=True)
class Disparity:
    """How far the scene moved between two frames, and how well it tracked."""

    # Median displacement in pixels of the images it was measured on.
    median_px: float
    # Fraction of the seeded points the tracker kept. Falls as content leaves
    # the frame, which is how we detect a pair that has drifted too far apart.
    # Lucas-Kanade's status means "converged", not "converged correctly", so an
    # unrelated frame still scores high. Do not read this as confidence.
    tracked_ratio: float
    # False when there were no corners to seed, or none survived. The other two
    # fields are then zero because nothing was measured, not because nothing moved.
    measurable: bool = True


def sharpness(gray: GrayArray) -> float:
    """Variance of the Laplacian, which drops as edges smear.

    Scene-dependent, so the value only means something next to other frames of
    the same sequence measured at the same resolution.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def disparity(reference: GrayArray, candidate: GrayArray) -> Disparity:
    """Apparent motion between two frames, via sparse optical flow.

    Tracking features rather than differencing pixels, because a change of
    exposure moves every pixel value without the camera having moved at all.
    """
    features = cv2.goodFeaturesToTrack(
        reference,
        maxCorners=_MAX_CORNERS,
        qualityLevel=_QUALITY_LEVEL,
        minDistance=_MIN_CORNER_DISTANCE,
    )
    # Returns None, not an empty array, when the frame has no corners.
    if features is None or len(features) == 0:
        return Disparity(median_px=0.0, tracked_ratio=0.0, measurable=False)

    # nextPts is an output buffer here: without OPTFLOW_USE_INITIAL_FLOW its
    # contents are ignored, and passing one keeps the call typed.
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        reference, candidate, features, np.empty_like(features)
    )
    kept = cast(NDArray[np.bool_], status.ravel() == 1)
    tracked_ratio = float(kept.mean())
    if not kept.any():
        return Disparity(median_px=0.0, tracked_ratio=0.0, measurable=False)

    moved = tracked[kept] - features[kept]
    # A few badly tracked points are enough to wreck the mean.
    median_px = float(np.median(np.linalg.norm(moved, axis=-1)))
    return Disparity(median_px=median_px, tracked_ratio=tracked_ratio)


def to_working_gray(image: NDArray[np.uint8], long_edge: int) -> GrayArray:
    """Grayscale copy at the resolution both metrics are measured on."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = working_scale(int(gray.shape[1]), int(gray.shape[0]), long_edge)
    if scale == 1.0:
        return cast(GrayArray, gray)
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cast(GrayArray, resized)


def working_scale(width: int, height: int, long_edge: int) -> float:
    """Downscale factor capping the long edge; never enlarges."""
    return min(1.0, long_edge / max(width, height))
