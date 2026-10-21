"""Tests for keyframe selection.

Each frame is translated by an exact number of pixels, so which indices should
be kept follows arithmetically from the disparity threshold.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import pytest

from unproject.config import PreprocessConfig
from unproject.preprocess.metrics import GrayArray
from unproject.preprocess.selection import select_keyframes


def _panning_sequence(*, count: int, step_px: int) -> list[GrayArray]:
    """A textured frame panning sideways by `step_px` per frame."""
    rng = np.random.default_rng(seed=0)
    # Wider than the crop, so panning reveals new content rather than wrapping.
    canvas = cv2.GaussianBlur(rng.integers(0, 256, (240, 1200), dtype=np.uint8), (3, 3), 0)
    return [
        cast(GrayArray, canvas[:, i * step_px : i * step_px + 320].copy()) for i in range(count)
    ]


def _blur(frame: GrayArray) -> GrayArray:
    return cast(GrayArray, cv2.GaussianBlur(frame, (21, 21), 0))


def _config(**overrides: object) -> PreprocessConfig:
    base = {
        "min_disparity_px": 12.0,
        "blur_percentile": 0.0,
        "first_frame_window": 1,
    }
    return PreprocessConfig.model_validate(base | overrides)


# Thresholds sit between two multiples of the step. Optical flow measures the
# pan to about 1e-5 px, so one placed exactly on an achievable displacement
# decides on float noise instead of on the algorithm.
def test_keeps_every_third_frame_at_four_pixels_of_pan() -> None:
    # 4 px per frame against a 10 px threshold: two fall short, three clear it.
    frames = _panning_sequence(count=13, step_px=4)

    kept = select_keyframes(frames, _config(min_disparity_px=10.0))

    assert [keyframe.index for keyframe in kept] == [0, 3, 6, 9, 12]


def test_a_lower_threshold_keeps_more() -> None:
    frames = _panning_sequence(count=13, step_px=4)

    kept = select_keyframes(frames, _config(min_disparity_px=6.0))

    assert [keyframe.index for keyframe in kept] == [0, 2, 4, 6, 8, 10, 12]


def test_a_static_camera_yields_a_single_keyframe() -> None:
    frames = _panning_sequence(count=1, step_px=0) * 20

    kept = select_keyframes(frames, _config())

    # No parallax anywhere, so nothing beyond the anchor is worth keeping.
    assert [keyframe.index for keyframe in kept] == [0]


def test_the_first_keyframe_is_the_sharpest_of_the_opening_window() -> None:
    frames = _panning_sequence(count=13, step_px=4)
    frames[0] = _blur(frames[0])
    frames[1] = _blur(frames[1])

    kept = select_keyframes(frames, _config(first_frame_window=4, blur_percentile=0.0))

    # Frame 0 is not automatically the anchor.
    assert kept[0].index == 2


def test_blurred_frames_are_rejected() -> None:
    frames = _panning_sequence(count=13, step_px=4)
    frames[3] = _blur(frames[3])

    kept = select_keyframes(frames, _config(blur_percentile=20.0))

    assert 3 not in [keyframe.index for keyframe in kept]


def test_the_safety_net_keeps_the_sharpest_of_a_blurred_stretch() -> None:
    pristine = _panning_sequence(count=24, step_px=4)
    frames = list(pristine)
    for index in range(1, 20):
        frames[index] = _blur(frames[index])
    # Least blurred of the run, so the pick is unambiguous, but still under the
    # floor so it cannot be kept on merit.
    frames[10] = cast(GrayArray, cv2.GaussianBlur(pristine[10], (15, 15), 0))

    kept = select_keyframes(frames, _config(blur_percentile=80.0, max_skipped_frames=15))

    forced = [keyframe for keyframe in kept if keyframe.forced]
    # Without the net, 19 blurred frames leave a hole in the chain.
    assert forced
    assert forced[0].index == 10


def test_low_disparity_does_not_trigger_the_safety_net() -> None:
    frames = _panning_sequence(count=1, step_px=0) * 40

    kept = select_keyframes(frames, _config(max_skipped_frames=5))

    # Only blur rejections count towards the net. A static camera has no
    # parallax to capture, so forced keyframes there would be degenerate pairs.
    assert [keyframe.index for keyframe in kept] == [0]


def test_disparity_scale_converts_to_output_pixels() -> None:
    frames = _panning_sequence(count=13, step_px=4)

    # Measured at half the output resolution, so 4 px measured is 8 px of
    # output and a 10 px threshold clears every other frame, not every third.
    kept = select_keyframes(frames, _config(min_disparity_px=10.0), disparity_scale=2.0)

    assert [keyframe.index for keyframe in kept] == [0, 2, 4, 6, 8, 10, 12]


def test_records_the_measurements_that_justified_each_keyframe() -> None:
    frames = _panning_sequence(count=7, step_px=4)

    kept = select_keyframes(frames, _config(min_disparity_px=10.0))

    assert kept[0].displacement_px == 0.0
    assert kept[0].tracked_ratio == 1.0
    for keyframe in kept[1:]:
        assert keyframe.displacement_px == pytest.approx(12.0, abs=1.5)
        assert keyframe.tracked_ratio > 0.5
        assert keyframe.sharpness > 0.0


def test_an_empty_sequence_selects_nothing() -> None:
    assert select_keyframes([], _config()) == []


def test_a_single_frame_is_kept() -> None:
    frames = _panning_sequence(count=1, step_px=4)

    kept = select_keyframes(frames, _config())

    assert [keyframe.index for keyframe in kept] == [0]
