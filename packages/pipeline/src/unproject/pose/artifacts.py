"""On-disk contract produced by stage [3].

This is the experiment's variable. The file has the same shape whichever backend
wrote it, which is the whole basis of the learned-versus-COLMAP comparison, so
changing it invalidates every result measured before the change.
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.pose.geometry import CONVENTION

# 2 added points_byte_length, so a reader can tell a complete cloud from one a
# full disk cut short.
POSE_SCHEMA_VERSION = 2

ARTIFACT_FILENAME = "poses.json"
POINTS_FILENAME = "points.ply"

_UNIT_TOLERANCE = 1e-6


class Camera(BaseModel):
    """Intrinsics for one physical camera.

    Focal length is stored per axis even for models that only estimate one, so
    consumers never branch on the model name to read a focal length.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    camera_id: int = Field(ge=0)
    # The backend's own model name, kept for traceability rather than for
    # dispatch. COLMAP writes things like SIMPLE_RADIAL or OPENCV.
    model: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0.0)
    fy: float = Field(gt=0.0)
    cx: float
    cy: float
    # Model-dependent and often empty. Order follows whatever the named model
    # defines; nothing here reinterprets it.
    distortion: list[float] = Field(default_factory=list)


class CameraPose(BaseModel):
    """Where one keyframe was taken from.

    World-to-camera, per the convention on the artifact. The camera position in
    world space is `-R^T t`, not `t`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Links back to the stage [2] keyframe, so a pose can be traced to an image
    # and from there to the original frame.
    keyframe_index: int = Field(ge=0)
    camera_id: int = Field(ge=0)
    quaternion_wxyz: tuple[float, float, float, float]
    translation: tuple[float, float, float]

    @model_validator(mode="after")
    def _check_unit_quaternion(self) -> Self:
        norm = float(np.linalg.norm(np.asarray(self.quaternion_wxyz, dtype=np.float64)))
        if abs(norm - 1.0) > _UNIT_TOLERANCE:
            msg = f"quaternion is not unit length ({norm:.9f}); normalise before storing"
            raise ValueError(msg)
        return self


class BackendInfo(BaseModel):
    """Which estimator produced the poses, and at what version.

    The comparison is between backends, so a run whose backend version is
    unrecorded cannot be placed in a results table.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["colmap", "lingbot"]
    version: str
    config_hash: str


class RegistrationSummary(BaseModel):
    """How much of the input the backend managed to place.

    `keyframes_registered` against `keyframes_total` is a headline number of
    the pose comparison: a backend that solves half the views is not comparable
    on reconstruction quality alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    keyframes_total: int = Field(ge=0)
    keyframes_registered: int = Field(ge=0)
    points_count: int = Field(default=0, ge=0)
    # COLMAP minimises this quantity; a learned backend has no equivalent, so it
    # stays optional rather than being faked with a placeholder.
    mean_reprojection_error_px: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if self.keyframes_registered > self.keyframes_total:
            msg = (
                f"registered {self.keyframes_registered} of {self.keyframes_total} keyframes, "
                "which is more than were given"
            )
            raise ValueError(msg)
        return self


class PoseArtifact(BaseModel):
    """Stage [3] output descriptor, identical in shape across backends."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=POSE_SCHEMA_VERSION, ge=1)
    # Repeated in the file so it can be read without consulting the source.
    convention: Literal["colmap_world_to_camera_opencv"] = CONVENTION
    preprocess_hash: str
    backend: BackendInfo
    cameras: list[Camera]
    # Only the keyframes the backend placed. A short list is a result, not a
    # failure, and `registration` says how short.
    poses: list[CameraPose]
    registration: RegistrationSummary
    # Sparse cloud beside this file, which stage [4] initialises from. None when
    # the backend produced no points.
    points_path: str | None = None
    points_byte_length: int | None = Field(default=None, ge=0)

    def expected_files(self) -> list[tuple[str, int]]:
        """Every file this artifact promises, with the size it promises."""
        if self.points_path is None or self.points_byte_length is None:
            return []
        return [(self.points_path, self.points_byte_length)]

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != POSE_SCHEMA_VERSION:
            msg = (
                f"pose artifact schema_version {self.schema_version} is not readable by "
                f"this build (expected {POSE_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        known = {camera.camera_id for camera in self.cameras}
        unknown = sorted({pose.camera_id for pose in self.poses} - known)
        if unknown:
            msg = f"poses reference camera_id {unknown} that no camera declares"
            raise ValueError(msg)

        indices = [pose.keyframe_index for pose in self.poses]
        if len(set(indices)) != len(indices):
            msg = "two poses claim the same keyframe_index"
            raise ValueError(msg)

        if len(self.poses) != self.registration.keyframes_registered:
            msg = (
                f"{len(self.poses)} poses stored but registration reports "
                f"{self.registration.keyframes_registered}"
            )
            raise ValueError(msg)
        return self
