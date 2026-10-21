"""Tests for the learned backend's deterministic parts.

The model itself is not asserted on: its correctness is a question for the
evaluation metrics, not for a unit test. What is tested here is everything
around it that would silently produce plausible wrong numbers - the intrinsic
conversion out of the model's cropped resolution, and the back-projection that
turns depth into the cloud stage [4] starts from.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from unproject.config import LingbotConfig
from unproject.pose.backend import PoseBackend, PoseBackendError
from unproject.pose.geometry import camera_center, quaternion_to_matrix
from unproject.pose.lingbot import (
    PATCH_SIZE,
    LingbotBackend,
    backproject,
    intrinsics_to_keyframe,
    model_space_geometry,
    poses_from_extrinsics,
    select_confident,
)

_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def test_the_stage_loads_one_backend_library_at_a_time() -> None:
    """pycolmap and torch cannot share a process.

    Each bundles its own libomp (see `openmp_guard.py`). A run
    only ever uses one backend, so the stage has to import them lazily. This
    asserts it still does; a plain module-level import would put both in the
    same process the moment the stage is loaded.
    """
    probe = (
        "import sys; import unproject.pose.stage; "
        "print('pycolmap' in sys.modules, 'torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


def test_backend_satisfies_the_protocol() -> None:
    assert isinstance(LingbotBackend(LingbotConfig(), config_dir=Path()), PoseBackend)


def test_weights_are_pinned_by_revision_and_digest() -> None:
    config = LingbotConfig()
    # A tag or a branch would let an upstream reupload change the results of a
    # run that reproduces its config exactly.
    assert len(config.revision) == 40
    assert len(config.sha256) == 64


def test_empty_input_is_refused() -> None:
    with pytest.raises(PoseBackendError):
        LingbotBackend(LingbotConfig(), config_dir=Path()).estimate([], [])


def test_resized_height_is_a_whole_number_of_patches() -> None:
    # The aggregator tokenises in patches, so a height that is not a multiple of
    # the patch size cannot be represented at all.
    for height in (713, 720, 1080, 1440):
        _, resized_height, _ = model_space_geometry(1280, height, 518)
        assert resized_height % PATCH_SIZE == 0


def test_a_tall_frame_is_cropped_and_a_wide_one_is_not() -> None:
    _, tall_height, tall_crop = model_space_geometry(480, 640, 518)
    assert tall_height > 518
    assert tall_crop == (tall_height - 518) // 2

    _, wide_height, wide_crop = model_space_geometry(1280, 720, 518)
    assert wide_height < 518
    assert wide_crop == 0


def test_degenerate_sizes_are_refused() -> None:
    with pytest.raises(PoseBackendError):
        model_space_geometry(0, 480, 518)


def test_intrinsics_come_back_at_the_keyframe_resolution() -> None:
    # A model-space focal length is measured in the cropped 518-pixel image and
    # has to be scaled back, or it cannot be compared with COLMAP's.
    width, height = 1280, 720
    resized_width, resized_height, _ = model_space_geometry(width, height, 518)
    scale_x, scale_y = resized_width / width, resized_height / height

    fx, fy, cx, cy = intrinsics_to_keyframe(
        _intrinsics(600.0 * scale_x, 600.0 * scale_y, 259.0, 147.0),
        width=width,
        height=height,
        image_size=518,
    )
    assert fx == pytest.approx(600.0)
    assert fy == pytest.approx(600.0)
    assert cx == pytest.approx(259.0 / scale_x)
    assert cy == pytest.approx(147.0 / scale_y)


def test_the_crop_offset_moves_the_principal_point() -> None:
    # A portrait frame loses rows off the top, so the model's centre sits lower
    # in the keyframe than it does in the crop.
    width, height = 480, 640
    _, resized_height, crop_top = model_space_geometry(width, height, 518)
    assert crop_top > 0

    *_, cy = intrinsics_to_keyframe(
        _intrinsics(700.0, 700.0, 259.0, 259.0), width=width, height=height, image_size=518
    )
    assert cy == pytest.approx((259.0 + crop_top) / (resized_height / height))
    assert cy > 259.0


def test_a_pixel_at_the_principal_point_lands_on_the_optical_axis() -> None:
    depth = np.full((3, 3), 2.0)
    points = backproject(depth, _intrinsics(100.0, 100.0, 1.0, 1.0), _IDENTITY_Q, np.zeros(3))
    centre = points.reshape(3, 3, 3)[1, 1]
    assert np.allclose(centre, [0.0, 0.0, 2.0])


def test_back_projection_undoes_the_projection() -> None:
    # Project a known world point, then lift the depth back and expect the point.
    intrinsics = _intrinsics(320.0, 320.0, 160.0, 120.0)
    quaternion = np.array([np.cos(0.3), 0.0, np.sin(0.3), 0.0])
    quaternion /= np.linalg.norm(quaternion)
    translation = np.array([0.4, -0.2, 3.0])

    world = np.array([0.3, -0.5, 1.2])
    camera = quaternion_to_matrix(quaternion) @ world + translation
    column = intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]
    row = intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]

    depth = np.zeros((240, 320))
    depth[round(row), round(column)] = camera[2]
    lifted = backproject(depth, intrinsics, quaternion, translation)
    recovered = lifted.reshape(240, 320, 3)[round(row), round(column)]
    assert np.allclose(recovered, world, atol=1e-2)


def test_the_translation_is_not_the_camera_position() -> None:
    # Same trap as the COLMAP path: a zero-depth pixel lifts to -R^T t, not -t.
    quaternion = np.array([np.cos(0.5), 0.0, 0.0, np.sin(0.5)])
    quaternion /= np.linalg.norm(quaternion)
    translation = np.array([1.0, 2.0, 3.0])

    origin = backproject(np.zeros((1, 1)), _intrinsics(1.0, 1.0, 0.0, 0.0), quaternion, translation)
    expected = -quaternion_to_matrix(quaternion).T @ translation
    assert np.allclose(origin[0], expected)
    assert not np.allclose(origin[0], -translation)


def test_selection_keeps_the_confident_half() -> None:
    confidence = np.arange(100, dtype=np.float64).reshape(10, 10)
    keep = select_confident(confidence, percentile=50.0, budget=1000)
    assert keep.size == 50
    assert confidence.reshape(-1)[keep].min() >= 50.0


def test_selection_respects_the_budget_and_repeats() -> None:
    confidence = np.arange(400, dtype=np.float64).reshape(20, 20)
    first = select_confident(confidence, percentile=0.0, budget=37)
    second = select_confident(confidence, percentile=0.0, budget=37)
    assert first.size == 37
    # Strided rather than sampled, so two runs of the same config agree without
    # a seed having to be carried for it.
    assert np.array_equal(first, second)


def test_selection_spreads_across_the_frame() -> None:
    # A budget met by taking the first N indices would cover only the top rows.
    confidence = np.ones((40, 40), dtype=np.float64)
    keep = select_confident(confidence, percentile=0.0, budget=20)
    rows = keep // 40
    assert rows.min() < 5
    assert rows.max() > 34


def test_the_models_translation_is_where_the_camera_is() -> None:
    """The direction the whole backend turns on.

    The model emits camera-to-world, so its `t` is the camera position and the
    stored pose has to be the inverse of what it hands over. Reading it the
    other way costs nothing at write time and points every camera backwards:
    measured on two benchmark scenes, 173 degrees of mean rotation error
    against 1.1.
    """
    rotation = quaternion_to_matrix(np.array([np.cos(0.4), 0.0, np.sin(0.4), 0.0]))
    position = np.array([1.5, -0.5, 3.0])
    extrinsics = np.concatenate([rotation, position[:, None]], axis=1)[None]

    pose = poses_from_extrinsics(extrinsics, [7])[0]
    assert pose.keyframe_index == 7
    # The stored pose is world-to-camera, so the camera sits at -R^T t.
    assert np.allclose(
        camera_center(np.array(pose.quaternion_wxyz), np.array(pose.translation)), position
    )
    # And the stored translation is not the position, which is the giveaway.
    assert not np.allclose(np.array(pose.translation), position)


def test_the_stored_rotation_is_the_inverse_of_the_models() -> None:
    rotation = quaternion_to_matrix(np.array([np.cos(0.3), np.sin(0.3), 0.0, 0.0]))
    extrinsics = np.concatenate([rotation, np.zeros((3, 1))], axis=1)[None]

    pose = poses_from_extrinsics(extrinsics, [0])[0]
    stored = quaternion_to_matrix(np.array(pose.quaternion_wxyz))
    assert np.allclose(stored, rotation.T)
    # A camera-to-world R turns camera axes into world axes; the stored one has
    # to turn them back.
    assert np.allclose(stored @ rotation, np.eye(3))


def test_a_ring_of_cameras_survives_the_conversion() -> None:
    # An orbit facing inward is the shape a real capture makes, and the one a
    # direction error destroys: every camera ends up looking outward.
    centres, extrinsics = [], []
    for degrees in range(0, 360, 30):
        angle = np.deg2rad(degrees)
        centre = np.array([4.0 * np.sin(angle), 0.0, 4.0 * np.cos(angle)])
        forward = -centre / np.linalg.norm(centre)
        down = np.array([0.0, -1.0, 0.0])
        right = np.cross(down, forward)
        # Camera-to-world columns, which is what the model emits.
        rotation = np.stack([right, down, forward], axis=1)
        centres.append(centre)
        extrinsics.append(np.concatenate([rotation, centre[:, None]], axis=1))

    poses = poses_from_extrinsics(np.array(extrinsics), list(range(len(centres))))
    for pose, centre in zip(poses, centres, strict=True):
        recovered = camera_center(np.array(pose.quaternion_wxyz), np.array(pose.translation))
        assert np.allclose(recovered, centre, atol=1e-9)
        # World-to-camera: the third row is the view direction in world space.
        view = quaternion_to_matrix(np.array(pose.quaternion_wxyz))[2, :]
        assert view @ (-centre / np.linalg.norm(centre)) == pytest.approx(1.0)


def test_the_weights_live_in_the_repository() -> None:
    # A run that can be reproduced from its config should not also depend on a
    # user-wide cache that the config says nothing about.
    assert LingbotConfig().cache_dir == Path("../models")
