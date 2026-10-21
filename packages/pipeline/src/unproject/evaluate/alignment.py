"""Lining two pose runs up, so the difference between them can be measured.

Neither backend knows the real scale of the scene. COLMAP fixes it arbitrarily
when it triangulates the first pair; the learned model has its own. So the two
trajectories are the same reconstruction only up to a similarity - a scale, a
rotation and a translation - and comparing them means solving for that first and
calling what is left the error.

The solution is Umeyama's, which is the least-squares optimum and closed form
rather than iterative, so a run cannot converge differently on two machines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Points = NDArray[np.float64]
Matrix3 = NDArray[np.float64]
Vector3 = NDArray[np.float64]

# A spread or a scale below this counts as zero: the source trajectory is a
# single point, or the two have no consistent scale, and no similarity can be
# recovered.
_DEGENERACY_TOLERANCE = 1e-9

# A similarity has seven degrees of freedom. Fewer than three points leave it
# free, and three leave only two of their nine coordinates to disagree, so the
# fit absorbs most of whatever error there is.
_MINIMUM_POINTS = 4


class AlignmentError(ValueError):
    """Raised when two trajectories cannot be lined up."""


@dataclass(frozen=True, slots=True)
class Similarity:
    """The transform taking one run's world into the other's."""

    scale: float
    rotation: Matrix3
    translation: Vector3

    def apply(self, points: Points) -> Points:
        moved: Points = self.scale * (np.asarray(points, dtype=np.float64) @ self.rotation.T)
        return moved + self.translation


def similarity_transform(source: Points, target: Points) -> Similarity:
    """Least-squares similarity taking `source` onto `target`.

    Umeyama 1991. The reflection guard matters: without it a noisy or nearly
    planar trajectory can be fitted by a mirrored rotation, which lowers the
    residual and describes a scene that is inside out.
    """
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise AlignmentError(f"expected two matching (n, 3) arrays, got {a.shape} and {b.shape}")
    if len(a) < _MINIMUM_POINTS:
        raise AlignmentError(
            f"{len(a)} point pairs cannot determine a similarity; {_MINIMUM_POINTS} are needed"
        )

    centroid_a, centroid_b = a.mean(axis=0), b.mean(axis=0)
    centred_a, centred_b = a - centroid_a, b - centroid_b

    variance_a = float((centred_a**2).sum() / len(a))
    if variance_a < _DEGENERACY_TOLERANCE:
        raise AlignmentError("the source trajectory has no extent, so no scale can be recovered")

    covariance = centred_b.T @ centred_a / len(a)
    left, singular, right = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0.0:
        correction[2, 2] = -1.0

    rotation: Matrix3 = left @ correction @ right
    scale = float((singular * np.diag(correction)).sum() / variance_a)
    if scale < _DEGENERACY_TOLERANCE:
        raise AlignmentError("the two trajectories have no consistent scale between them")

    return Similarity(
        scale=scale,
        rotation=rotation,
        translation=centroid_b - scale * (rotation @ centroid_a),
    )


def rotation_error_degrees(first: Matrix3, second: Matrix3) -> float:
    """Angle of the rotation that takes `first` to `second`.

    The geodesic distance on SO(3), which is the only rotation error that does
    not depend on how the rotation happens to be parameterised. Euler angles and
    per-component quaternion differences both do.
    """
    relative = np.asarray(first, dtype=np.float64) @ np.asarray(second, dtype=np.float64).T
    # Rounding can push the trace a hair outside the valid range for arccos.
    cosine = (float(np.trace(relative)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def trajectory_extent(points: Points) -> float:
    """Diagonal of the bounding box, as the scale errors are quoted against.

    An absolute distance means nothing when the scene's own scale is arbitrary.
    A fraction of how far the camera travelled is comparable between scenes.
    """
    array = np.asarray(points, dtype=np.float64)
    if len(array) == 0:
        return 0.0
    return float(np.linalg.norm(array.max(axis=0) - array.min(axis=0)))
