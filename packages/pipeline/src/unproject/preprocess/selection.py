"""Keyframe selection.

Grayscale frames in, the indices worth keeping out. No I/O, so the algorithm can
be tested against sequences built to have a known answer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from unproject.config import PreprocessConfig
from unproject.preprocess.metrics import GrayArray, disparity, sharpness

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Keyframe:
    """One selected frame, with the measurements behind the decision."""

    index: int
    sharpness: float
    # Motion from the previous keyframe, in output-image pixels. Zero for the
    # first, which has nothing to measure against.
    displacement_px: float
    tracked_ratio: float
    # Kept by the safety net below rather than on merit.
    forced: bool


def select_keyframes(
    grays: Sequence[GrayArray],
    config: PreprocessConfig,
    *,
    disparity_scale: float = 1.0,
) -> list[Keyframe]:
    """Reduce a sequence to the frames that carry new information.

    `grays` are at the metric working resolution. `disparity_scale` converts a
    displacement measured there into output-image pixels, so `min_disparity_px`
    keeps its meaning when `metric_long_edge` is tuned for speed.
    """
    if not grays:
        return []

    sharpnesses = [sharpness(gray) for gray in grays]
    blur_floor = float(np.percentile(sharpnesses, config.blur_percentile))

    first = _sharpest(sharpnesses, stop=min(config.first_frame_window, len(grays)))
    kept = [
        Keyframe(
            index=first,
            sharpness=sharpnesses[first],
            displacement_px=0.0,
            tracked_ratio=1.0,
            forced=False,
        )
    ]

    last_kept = first
    # Only blur rejections count towards the safety net. Skipping a static
    # stretch is correct behaviour, since there is no parallax there to capture
    # and a forced keyframe would just add a degenerate pair.
    blurred_since_kept: list[int] = []

    for index in range(first + 1, len(grays)):
        if sharpnesses[index] < blur_floor:
            blurred_since_kept.append(index)
            if len(blurred_since_kept) > config.max_skipped_frames:
                rescued = max(blurred_since_kept, key=lambda i: sharpnesses[i])
                measured = disparity(grays[last_kept], grays[rescued])
                kept.append(
                    Keyframe(
                        index=rescued,
                        sharpness=sharpnesses[rescued],
                        displacement_px=measured.median_px * disparity_scale,
                        tracked_ratio=measured.tracked_ratio,
                        forced=True,
                    )
                )
                logger.debug(
                    "forced keyframe %d after %d blurred frames", rescued, len(blurred_since_kept)
                )
                last_kept = rescued
                blurred_since_kept = []
            continue

        measured = disparity(grays[last_kept], grays[index])
        displacement = measured.median_px * disparity_scale

        if not measured.measurable:
            # Nothing to track against, so redundancy cannot be shown either way.
            # Keep the frame; the recorded tracked_ratio of zero says why.
            logger.debug("frame %d could not be measured against %d", index, last_kept)
        else:
            # A collapsed track means the pair is already past what matching can
            # bridge, so take this frame instead of drifting further.
            too_far = measured.tracked_ratio < config.min_tracked_ratio
            if not too_far and displacement < config.min_disparity_px:
                continue

        kept.append(
            Keyframe(
                index=index,
                sharpness=sharpnesses[index],
                displacement_px=displacement,
                tracked_ratio=measured.tracked_ratio,
                forced=False,
            )
        )
        last_kept = index
        blurred_since_kept = []

    return kept


def _sharpest(sharpnesses: Sequence[float], *, stop: int) -> int:
    """Index of the sharpest frame in the opening window.

    A handheld take often starts while the shot is still being framed, so the
    anchor is chosen rather than taken from position zero.
    """
    return max(range(stop), key=lambda i: sharpnesses[i])
