"""The interface both pose estimators go through."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from unproject.pose.artifacts import BackendInfo, Camera, CameraPose
from unproject.pose.pointcloud import Colors, Points


class PoseBackendError(Exception):
    """Raised when a backend cannot estimate poses for the given keyframes."""


class PoseEstimate:
    """What a backend returns before it becomes an artifact.

    Plain object rather than a model: it carries numpy arrays and is consumed
    immediately by the stage, which is what validates and writes.
    """

    def __init__(
        self,
        *,
        cameras: list[Camera],
        poses: list[CameraPose],
        points: Points | None = None,
        colors: Colors | None = None,
        mean_reprojection_error_px: float | None = None,
    ) -> None:
        self.cameras = cameras
        self.poses = poses
        self.points = points
        self.colors = colors
        self.mean_reprojection_error_px = mean_reprojection_error_px


@runtime_checkable
class PoseBackend(Protocol):
    """Estimates camera poses for an ordered set of keyframe images.

    Implementations return poses in the project convention, converting from
    whatever they use internally. Registering fewer keyframes than it was given
    is a legitimate result and is reported, not raised.

    Implemented: `ColmapBackend` and `LingbotBackend`.
    """

    def describe(self) -> BackendInfo: ...

    def estimate(self, image_paths: list[Path], keyframe_indices: list[int]) -> PoseEstimate: ...
