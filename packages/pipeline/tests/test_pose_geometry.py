"""Tests for the rotation and pose conversions.

These are the conversions the learned-versus-COLMAP comparison rests on. A sign
error here produces poses that look plausible and are wrong, so the cases below
use rotations whose answer can be written down by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from unproject.pose.geometry import (
    CONVENTION,
    GeometryError,
    camera_center,
    invert_pose,
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_to_matrix,
)

_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def test_convention_is_named_for_the_artifact() -> None:
    # The artifact repeats this string so a reader never has to guess.
    assert CONVENTION == "colmap_world_to_camera_opencv"


def test_identity_quaternion_is_the_identity_matrix() -> None:
    assert np.allclose(quaternion_to_matrix(_IDENTITY_Q), np.eye(3))


def test_ninety_degrees_about_z_maps_x_onto_y() -> None:
    quaternion = matrix_to_quaternion(_rotation_z(90.0))

    rotated = quaternion_to_matrix(quaternion) @ np.array([1.0, 0.0, 0.0])

    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("degrees", [0.0, 30.0, 90.0, 179.0, 180.0, 270.0])
def test_matrix_and_quaternion_round_trip(degrees: float) -> None:
    matrix = _rotation_z(degrees)

    assert np.allclose(quaternion_to_matrix(matrix_to_quaternion(matrix)), matrix, atol=1e-9)


def test_the_180_degree_case_is_stable() -> None:
    # The naive trace formula divides by something that vanishes here, which is
    # the rotation a turntable capture produces halfway round.
    quaternion = matrix_to_quaternion(_rotation_z(180.0))

    assert np.all(np.isfinite(quaternion))
    assert np.allclose(quaternion_to_matrix(quaternion), _rotation_z(180.0), atol=1e-9)


def test_normalisation_pins_the_sign() -> None:
    quaternion = np.array([-0.5, 0.5, -0.5, 0.5])

    normalised = normalize_quaternion(quaternion)

    # q and -q are the same rotation, so the sign is fixed to keep artifacts
    # stable between runs and comparable between backends.
    assert normalised[0] >= 0.0
    assert np.allclose(quaternion_to_matrix(normalised), quaternion_to_matrix(quaternion))


def test_normalisation_scales_to_unit_length() -> None:
    assert np.isclose(np.linalg.norm(normalize_quaternion(np.array([2.0, 0.0, 0.0, 0.0]))), 1.0)


def test_a_zero_quaternion_is_rejected() -> None:
    with pytest.raises(GeometryError, match="no direction"):
        normalize_quaternion(np.zeros(4))


def test_camera_centre_is_not_the_translation() -> None:
    # Camera at (3, 0, 0) in world space, rotated 90 degrees about Z.
    rotation = _rotation_z(90.0).T
    centre = np.array([3.0, 0.0, 0.0])
    quaternion = matrix_to_quaternion(rotation)
    translation = -rotation @ centre

    assert np.allclose(camera_center(quaternion, translation), centre, atol=1e-9)
    # Reading `t` as the position is a silent error that misplaces every camera.
    assert not np.allclose(translation, centre)


def test_camera_centre_of_the_identity_pose_is_the_origin() -> None:
    assert np.allclose(camera_center(_IDENTITY_Q, np.zeros(3)), np.zeros(3))


def test_inverting_a_pose_twice_returns_it() -> None:
    quaternion = matrix_to_quaternion(_rotation_z(37.0))
    translation = np.array([1.0, -2.0, 0.5])

    once_q, once_t = invert_pose(quaternion, translation)
    twice_q, twice_t = invert_pose(once_q, once_t)

    assert np.allclose(twice_q, quaternion, atol=1e-9)
    assert np.allclose(twice_t, translation, atol=1e-9)


def test_inverting_swaps_centre_and_translation() -> None:
    quaternion = matrix_to_quaternion(_rotation_z(45.0))
    translation = np.array([0.0, 2.0, -1.0])

    _, inverted_t = invert_pose(quaternion, translation)

    # The camera-to-world translation is the camera centre in world space.
    assert np.allclose(inverted_t, camera_center(quaternion, translation), atol=1e-9)


def test_a_reflection_is_rejected() -> None:
    with pytest.raises(GeometryError, match="determinant"):
        matrix_to_quaternion(np.diag([1.0, 1.0, -1.0]))


def test_a_non_orthonormal_matrix_is_rejected() -> None:
    with pytest.raises(GeometryError, match="orthonormal"):
        matrix_to_quaternion(np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
