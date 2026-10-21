"""Tests for the trajectory alignment behind the pose comparison.

Every error the pose comparison reports passes through this arithmetic. A
similarity fitted wrongly reports an error that is really a scale mismatch, or
hides a real one, and either way the conclusion changes. The cases below use
transforms whose answer is known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from unproject.evaluate.alignment import (
    AlignmentError,
    Similarity,
    rotation_error_degrees,
    similarity_transform,
    trajectory_extent,
)


def _orbit(count: int = 12, radius: float = 2.0) -> np.ndarray:
    """Camera centres on a ring, which is the shape a real capture makes."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack(
        [radius * np.cos(angles), 0.3 * np.sin(3.0 * angles), radius * np.sin(angles)], axis=1
    )


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def test_recovers_a_transform_it_was_given() -> None:
    source = _orbit()
    truth = Similarity(
        scale=3.5, rotation=_rotation_z(40.0), translation=np.array([1.0, -2.0, 0.5])
    )
    solved = similarity_transform(source, truth.apply(source))

    assert solved.scale == pytest.approx(3.5)
    assert np.allclose(solved.rotation, truth.rotation)
    assert np.allclose(solved.translation, truth.translation)


def test_identical_trajectories_align_to_unit_scale() -> None:
    source = _orbit()
    solved = similarity_transform(source, source)
    assert solved.scale == pytest.approx(1.0)
    assert np.allclose(solved.rotation, np.eye(3))
    assert np.allclose(solved.translation, np.zeros(3))


def test_alignment_leaves_no_residual_when_the_runs_agree() -> None:
    source = _orbit()
    target = Similarity(
        scale=0.25, rotation=_rotation_z(-110.0), translation=np.array([4.0, 4.0, -1.0])
    ).apply(source)

    aligned = similarity_transform(source, target).apply(source)
    assert np.allclose(aligned, target, atol=1e-9)


def test_scale_is_solved_for_rather_than_counted_as_error() -> None:
    # The whole reason for a similarity rather than a rigid fit: neither backend
    # knows the true scale, so a trajectory ten times larger is the same answer.
    source = _orbit()
    solved = similarity_transform(source, source * 10.0)
    assert solved.scale == pytest.approx(10.0)
    assert np.allclose(solved.apply(source), source * 10.0)


def test_a_mirrored_trajectory_is_not_fitted_by_a_reflection() -> None:
    # Without the determinant guard the best fit to a mirrored scene is a
    # reflection, which reports near-zero error for a reconstruction that is
    # inside out.
    source = _orbit()
    mirrored = source * np.array([1.0, 1.0, -1.0])

    solved = similarity_transform(source, mirrored)
    assert np.linalg.det(solved.rotation) == pytest.approx(1.0)
    residual = np.linalg.norm(solved.apply(source) - mirrored, axis=1)
    assert residual.mean() > 0.1


def test_a_displaced_camera_shows_up_as_error() -> None:
    source = _orbit()
    target = source.copy()
    target[5] += np.array([0.4, 0.0, 0.0])

    residual = np.linalg.norm(similarity_transform(source, target).apply(source) - target, axis=1)
    assert residual.max() > 0.2
    assert residual[5] == pytest.approx(residual.max())


def test_too_few_points_is_refused() -> None:
    # A similarity has seven degrees of freedom, so three points leave only two
    # of their nine coordinates to disagree; fewer than four are refused.
    with pytest.raises(AlignmentError, match="cannot determine"):
        similarity_transform(np.zeros((3, 3)), np.eye(3))


def test_a_trajectory_with_no_extent_is_refused() -> None:
    still = np.zeros((8, 3))
    with pytest.raises(AlignmentError, match="no extent"):
        similarity_transform(still, _orbit(8))


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(AlignmentError, match="matching"):
        similarity_transform(_orbit(8), _orbit(9))


def test_rotation_error_is_zero_for_the_same_rotation() -> None:
    assert rotation_error_degrees(_rotation_z(31.0), _rotation_z(31.0)) == pytest.approx(0.0)


def test_rotation_error_is_the_angle_between_them() -> None:
    assert rotation_error_degrees(_rotation_z(10.0), _rotation_z(55.0)) == pytest.approx(45.0)


def test_rotation_error_is_symmetric_and_capped_at_180() -> None:
    first, second = _rotation_z(0.0), _rotation_z(190.0)
    assert rotation_error_degrees(first, second) == pytest.approx(170.0)
    assert rotation_error_degrees(second, first) == pytest.approx(170.0)


def test_rotation_error_survives_a_trace_at_the_limit() -> None:
    # arccos of anything past 1 is a NaN, and rounding puts the trace of an
    # identity comparison there routinely.
    assert rotation_error_degrees(np.eye(3), np.eye(3)) == 0.0


def test_extent_is_the_bounding_box_diagonal() -> None:
    points = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    assert trajectory_extent(points) == pytest.approx(5.0)


def test_extent_of_nothing_is_zero() -> None:
    assert trajectory_extent(np.zeros((0, 3))) == 0.0
