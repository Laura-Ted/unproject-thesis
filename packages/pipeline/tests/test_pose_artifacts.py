"""Tests for the poses.json contract.

This file is the same shape whichever backend wrote it, which is what makes the
learned-versus-COLMAP comparison mean anything. The validators below are what stop
a backend from quietly writing something the other could not produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from unproject.pose.artifacts import (
    POSE_SCHEMA_VERSION,
    BackendInfo,
    Camera,
    CameraPose,
    PoseArtifact,
    RegistrationSummary,
)
from unproject.pose.geometry import quaternion_to_matrix

_IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _camera(camera_id: int = 1) -> Camera:
    return Camera(
        camera_id=camera_id,
        model="SIMPLE_RADIAL",
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        distortion=[0.01],
    )


def _pose(keyframe_index: int, camera_id: int = 1) -> CameraPose:
    return CameraPose(
        keyframe_index=keyframe_index,
        camera_id=camera_id,
        quaternion_wxyz=_IDENTITY,
        translation=(0.0, 0.0, 0.0),
    )


def _artifact(poses: list[CameraPose], *, total: int = 4, **overrides: object) -> PoseArtifact:
    defaults: dict[str, object] = {
        "preprocess_hash": "abc123",
        "backend": BackendInfo(name="colmap", version="4.1.1", config_hash="deadbeef"),
        "cameras": [_camera()],
        "poses": poses,
        "registration": RegistrationSummary(
            keyframes_total=total,
            keyframes_registered=len(poses),
            points_count=100,
        ),
    }
    return PoseArtifact.model_validate(defaults | overrides)


def test_the_convention_travels_with_the_file() -> None:
    artifact = _artifact([_pose(0)])

    # A reader must not have to consult our source to interpret the numbers.
    assert artifact.convention == "colmap_world_to_camera_opencv"


def test_round_trips_through_json() -> None:
    artifact = _artifact([_pose(0), _pose(1)])

    assert PoseArtifact.model_validate_json(artifact.model_dump_json()) == artifact


def test_a_non_unit_quaternion_is_rejected() -> None:
    # Downstream code assumes unit length and produces a scaled scene otherwise.
    with pytest.raises(ValueError, match="not unit length"):
        CameraPose(
            keyframe_index=0,
            camera_id=1,
            quaternion_wxyz=(1.0, 1.0, 0.0, 0.0),
            translation=(0.0, 0.0, 0.0),
        )


def test_a_pose_referencing_an_unknown_camera_is_rejected() -> None:
    with pytest.raises(ValueError, match="no camera declares"):
        _artifact([_pose(0, camera_id=99)])


def test_two_poses_for_one_keyframe_are_rejected() -> None:
    with pytest.raises(ValueError, match="same keyframe_index"):
        _artifact([_pose(0), _pose(0)])


def test_the_registered_count_must_match_the_poses() -> None:
    with pytest.raises(ValueError, match="poses stored but registration reports"):
        _artifact(
            [_pose(0)],
            registration=RegistrationSummary(keyframes_total=4, keyframes_registered=3),
        )


def test_registering_more_than_were_given_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than were given"):
        RegistrationSummary(keyframes_total=2, keyframes_registered=3)


def test_partial_registration_is_a_result_not_an_error() -> None:
    artifact = _artifact([_pose(0), _pose(2)], total=4)

    # A backend that places half the views is a finding worth recording, and the
    # headline number the two backends are compared on.
    assert artifact.registration.keyframes_registered == 2
    assert artifact.registration.keyframes_total == 4


def test_a_backend_without_reprojection_error_is_allowed() -> None:
    # COLMAP minimises this; a learned backend has no equivalent, and faking a
    # placeholder would put a meaningless number in a results table.
    summary = RegistrationSummary(keyframes_total=4, keyframes_registered=4)

    assert summary.mean_reprojection_error_px is None


def test_an_unknown_backend_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        BackendInfo.model_validate({"name": "mystery", "version": "1", "config_hash": "x"})


def test_an_unreadable_schema_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _artifact([_pose(0)], schema_version=POSE_SCHEMA_VERSION + 1)


def test_an_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        _artifact([_pose(0)], convntion="typo")


def test_a_stored_quaternion_survives_a_round_trip_as_a_rotation() -> None:
    pose = CameraPose(
        keyframe_index=0,
        camera_id=1,
        quaternion_wxyz=(0.5, 0.5, 0.5, 0.5),
        translation=(1.0, 2.0, 3.0),
    )

    assert np.isclose(np.linalg.det(quaternion_to_matrix(np.array(pose.quaternion_wxyz))), 1.0)
