"""Stage [3] Pose: estimate camera poses for the keyframes.

The backend is the experiment's variable, so `poses.json` has the same shape
whichever one wrote it.
"""

from unproject.pose.artifacts import (
    BackendInfo,
    Camera,
    CameraPose,
    PoseArtifact,
    RegistrationSummary,
)
from unproject.pose.backend import PoseBackend, PoseBackendError, PoseEstimate
from unproject.pose.geometry import CONVENTION, GeometryError

__all__ = [
    "CONVENTION",
    "BackendInfo",
    "Camera",
    "CameraPose",
    "GeometryError",
    "PoseArtifact",
    "PoseBackend",
    "PoseBackendError",
    "PoseEstimate",
    "RegistrationSummary",
]
