"""Rotation and pose conversions.

Every backend has to land on one convention or the comparison between them is
meaningless. The one this project uses is COLMAP's, described in `CONVENTION`
and repeated in every artifact so a reader cannot guess wrong.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

Vector3 = NDArray[np.float64]
Matrix3 = NDArray[np.float64]
QuaternionWXYZ = NDArray[np.float64]

# Named in `poses.json` so the file is interpretable without this source.
#
#   extrinsics map world into camera:  x_cam = R @ x_world + t
#   quaternion order is (w, x, y, z)
#   camera axes are OpenCV's: +X right, +Y down, +Z forward along the view
#
# Chosen because COLMAP is one of the two backends, so the reference path needs
# no conversion at all, and because every 3DGS trainer consumes this layout.
CONVENTION: Final = "colmap_world_to_camera_opencv"

_UNIT_TOLERANCE = 1e-6


class GeometryError(ValueError):
    """Raised on input that is not a valid rotation."""


def normalize_quaternion(quaternion: QuaternionWXYZ) -> QuaternionWXYZ:
    """Scale to unit length and fix the sign.

    A quaternion and its negation are the same rotation, so the sign is pinned
    to make artifacts byte-stable between runs and diffable between backends.
    """
    array = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(array))
    if norm < _UNIT_TOLERANCE:
        raise GeometryError(f"quaternion has no direction: {array.tolist()}")

    unit = array / norm
    if unit[0] < 0.0:
        unit = -unit
    return unit


def quaternion_to_matrix(quaternion: QuaternionWXYZ) -> Matrix3:
    """Rotation matrix for a (w, x, y, z) quaternion."""
    w, x, y, z = normalize_quaternion(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: Matrix3) -> QuaternionWXYZ:
    """Inverse of `quaternion_to_matrix`.

    Uses the largest diagonal term as the pivot. The naive formula divides by a
    quantity that goes to zero near a 180 degree rotation, which is exactly the
    case a turntable capture produces.
    """
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    _check_rotation(rotation)

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        pivot = int(np.argmax(np.diag(rotation)))
        other = [(pivot + 1) % 3, (pivot + 2) % 3]
        scale = np.sqrt(1.0 + rotation[pivot, pivot] - sum(rotation[i, i] for i in other)) * 2.0
        components = np.zeros(3)
        components[pivot] = 0.25 * scale
        components[other[0]] = (rotation[other[0], pivot] + rotation[pivot, other[0]]) / scale
        components[other[1]] = (rotation[other[1], pivot] + rotation[pivot, other[1]]) / scale
        w = (rotation[other[1], other[0]] - rotation[other[0], other[1]]) / scale
        quaternion = np.array([w, *components])

    return normalize_quaternion(quaternion)


def camera_center(quaternion: QuaternionWXYZ, translation: Vector3) -> Vector3:
    """Where the camera sits in world space.

    The extrinsics are world-to-camera, so the position is not the translation
    itself but `-R^T t`. Reading `t` as the position is a common and silent
    mistake, and it puts every camera in the wrong place.
    """
    rotation = quaternion_to_matrix(quaternion)
    return -rotation.T @ np.asarray(translation, dtype=np.float64).reshape(3)


def invert_pose(quaternion: QuaternionWXYZ, translation: Vector3) -> tuple[QuaternionWXYZ, Vector3]:
    """Swap the direction of a rigid transform.

    Learned models usually emit camera-to-world, while this project stores
    world-to-camera, so one of the two backends always needs this.
    """
    rotation = quaternion_to_matrix(quaternion)
    inverse = rotation.T
    shifted = -inverse @ np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix_to_quaternion(inverse), shifted


def _check_rotation(matrix: Matrix3) -> None:
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6):
        raise GeometryError("matrix is not orthonormal, so it is not a rotation")
    determinant = float(np.linalg.det(matrix))
    if not np.isclose(determinant, 1.0, atol=1e-6):
        # -1 means a reflection crept in, usually from a mishandled axis flip.
        raise GeometryError(f"rotation has determinant {determinant:.6f}, expected 1")
