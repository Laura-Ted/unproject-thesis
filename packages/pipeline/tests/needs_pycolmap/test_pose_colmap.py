"""Tests for the COLMAP backend's conversion into the project convention.

The full COLMAP pipeline needs real footage and is not exercised here. What is
exercised is the translation between its types and ours, which is where a silent
error would live: pycolmap stores quaternions as (x, y, z, w) and this project
stores (w, x, y, z), and both orderings produce a valid rotation.

In this directory because it imports pycolmap; see `openmp_guard.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from unproject.config import ColmapConfig
from unproject.pose.backend import PoseBackendError
from unproject.pose.colmap import (
    ColmapBackend,
    ColmapCamera,
    ColmapImage,
    ColmapPoint3D,
    ColmapRigid3d,
    _collect_estimate,
    _pose_of,
)
from unproject.pose.geometry import camera_center, quaternion_to_matrix

_ANY_DIR = Path("/tmp/unproject-not-used")


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


class _FakeImage:
    """Only what the backend reads, wrapping a real pycolmap Rigid3d."""

    def __init__(
        self, name: str, camera_id: int, rigid: ColmapRigid3d, *, posed: bool = True
    ) -> None:
        self.name = name
        self.camera_id = camera_id
        self.has_pose = posed
        self._rigid = rigid

    def cam_from_world(self) -> ColmapRigid3d:
        return self._rigid


class _FakePoint:
    def __init__(self, xyz: np.ndarray, color: np.ndarray) -> None:
        self.xyz = xyz
        self.color = color


class _FakeReconstruction:
    def __init__(
        self,
        cameras: Mapping[int, ColmapCamera],
        images: Mapping[int, ColmapImage],
        points: Mapping[int, ColmapPoint3D],
        error: float = 0.42,
    ) -> None:
        self.cameras = cameras
        self.images = images
        self.points3D = points
        self._error = error

    def compute_mean_reprojection_error(self) -> float:
        return self._error


def _rigid(rotation: np.ndarray, translation: Sequence[float]) -> ColmapRigid3d:
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), np.array(translation, dtype=float))


def _camera(camera_id: int = 1) -> ColmapCamera:
    return pycolmap.Camera.create_from_model_name(camera_id, "SIMPLE_RADIAL", 500.0, 640, 480)


def test_the_real_pycolmap_types_still_satisfy_our_protocols() -> None:
    # Catches an upstream rename, which is how these protocols go stale.
    assert isinstance(_rigid(np.eye(3), [0.0, 0.0, 0.0]), ColmapRigid3d)
    assert isinstance(_camera(), ColmapCamera)


def test_pycolmap_exposes_cam_from_world_as_a_method() -> None:
    # A posed Image cannot be built without a full reconstruction, so the two
    # facts the backend depends on are pinned off the class instead. Reading
    # cam_from_world as an attribute returns the bound method and fails later.
    shape = {
        "Image.cam_from_world is a method": callable(pycolmap.Image.cam_from_world),
        "Image.has_pose is a property": isinstance(pycolmap.Image.has_pose, property),
        "Camera.extra_params_idxs is a method": callable(pycolmap.Camera.extra_params_idxs),
    }

    assert all(shape.values()), shape


def test_pycolmap_stores_the_quaternion_as_xyzw() -> None:
    # Pinned because the whole conversion below exists for this one fact.
    quat = pycolmap.Rotation3d(_rotation_z(90.0)).quat

    assert np.allclose(quat, [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-9)


def test_the_quaternion_is_reordered_not_copied() -> None:
    rotation = _rotation_z(90.0)
    image = _FakeImage("000000.png", 1, _rigid(rotation, [1.0, 2.0, 3.0]))

    quaternion, translation = _pose_of(image)

    # (w, x, y, z) for 90 degrees about Z, which is not what pycolmap handed over.
    assert np.allclose(quaternion, [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], atol=1e-9)
    assert np.allclose(quaternion_to_matrix(np.array(quaternion)), rotation, atol=1e-9)
    assert translation == (1.0, 2.0, 3.0)


def test_a_copied_quaternion_would_give_a_different_rotation() -> None:
    rotation = _rotation_z(90.0)
    raw = np.asarray(pycolmap.Rotation3d(rotation).quat, dtype=float)

    # Reading pycolmap's array as (w, x, y, z) yields a rotation that is valid
    # and wrong, which is why the reordering has its own test.
    assert not np.allclose(quaternion_to_matrix(raw), rotation, atol=1e-6)


def test_the_pose_stays_world_to_camera() -> None:
    # A camera at (5, 0, 0), turned 180 degrees about Z.
    centre = np.array([5.0, 0.0, 0.0])
    rotation = _rotation_z(180.0)
    image = _FakeImage("000000.png", 1, _rigid(rotation, -rotation @ centre))

    quaternion, translation = _pose_of(image)

    assert np.allclose(
        camera_center(np.array(quaternion), np.array(translation)), centre, atol=1e-9
    )


def test_collect_maps_names_back_to_keyframe_indices() -> None:
    reconstruction = _FakeReconstruction(
        cameras={1: _camera()},
        images={
            10: _FakeImage("000002.png", 1, _rigid(np.eye(3), [0.0, 0.0, 0.0])),
            11: _FakeImage("000000.png", 1, _rigid(_rotation_z(30.0), [1.0, 0.0, 0.0])),
        },
        points={},
    )

    estimate = _collect_estimate(reconstruction, {"000000.png": 0, "000002.png": 2})

    # Sorted by keyframe index, whatever order COLMAP's own ids came in.
    assert [pose.keyframe_index for pose in estimate.poses] == [0, 2]


def test_collect_skips_unposed_images() -> None:
    reconstruction = _FakeReconstruction(
        cameras={1: _camera()},
        images={
            10: _FakeImage("000000.png", 1, _rigid(np.eye(3), [0.0, 0.0, 0.0])),
            11: _FakeImage("000001.png", 1, _rigid(np.eye(3), [0.0, 0.0, 0.0]), posed=False),
        },
        points={},
    )

    estimate = _collect_estimate(reconstruction, {"000000.png": 0, "000001.png": 1})

    # An image COLMAP failed to place carries no pose and is simply absent.
    assert [pose.keyframe_index for pose in estimate.poses] == [0]


def test_collect_reads_intrinsics_without_branching_on_the_model() -> None:
    reconstruction = _FakeReconstruction(cameras={1: _camera()}, images={}, points={})

    estimate = _collect_estimate(reconstruction, {})

    camera = estimate.cameras[0]
    assert camera.model == "SIMPLE_RADIAL"
    assert (camera.fx, camera.fy) == (500.0, 500.0)
    assert (camera.cx, camera.cy) == (320.0, 240.0)
    # SIMPLE_RADIAL carries one radial term, which lands in distortion.
    assert len(camera.distortion) == 1


def test_collect_gathers_points_and_colours() -> None:
    reconstruction = _FakeReconstruction(
        cameras={1: _camera()},
        images={},
        points={
            1: _FakePoint(np.array([1.0, 2.0, 3.0]), np.array([10, 20, 30], dtype=np.uint8)),
            2: _FakePoint(np.array([4.0, 5.0, 6.0]), np.array([40, 50, 60], dtype=np.uint8)),
        },
    )

    estimate = _collect_estimate(reconstruction, {})

    assert estimate.points is not None
    assert estimate.points.shape == (2, 3)
    assert estimate.colors is not None
    assert estimate.colors.dtype == np.uint8


def test_collect_reports_no_points_rather_than_an_empty_array() -> None:
    estimate = _collect_estimate(
        _FakeReconstruction(cameras={1: _camera()}, images={}, points={}), {}
    )

    assert estimate.points is None


def test_the_matcher_follows_the_source_kind() -> None:
    video = ColmapBackend(ColmapConfig(), work_dir=_ANY_DIR, source_kind="video")
    images = ColmapBackend(ColmapConfig(), work_dir=_ANY_DIR, source_kind="image_directory")

    # Video frames arrive in order, so neighbours are the likely matches. An
    # image directory promises no such thing.
    assert video.resolve_matcher() == "sequential"
    assert images.resolve_matcher() == "exhaustive"


def test_an_explicit_matcher_overrides_the_source_kind() -> None:
    backend = ColmapBackend(
        ColmapConfig(matcher="exhaustive"), work_dir=_ANY_DIR, source_kind="video"
    )

    assert backend.resolve_matcher() == "exhaustive"


def test_describe_records_the_backend_version() -> None:
    backend = ColmapBackend(ColmapConfig(), work_dir=_ANY_DIR, source_kind="video")

    info = backend.describe()

    # A results table needs to say which build produced the numbers.
    assert info.name == "colmap"
    assert info.version == str(pycolmap.__version__)
    assert len(info.config_hash) == 64


def test_mismatched_inputs_are_rejected() -> None:
    backend = ColmapBackend(ColmapConfig(), work_dir=_ANY_DIR, source_kind="video")

    with pytest.raises(PoseBackendError, match="against"):
        backend.estimate([_ANY_DIR / "a.png"], [0, 1])
