"""Stage [B] Bodies: fit a parametric body to every person in every frame.

Only the artifact types are re-exported. The estimator stays one import away
because it reaches torch, which cannot share a process with pycolmap.
"""

from unproject.bodies.artifacts import (
    CONVENTION,
    BodiesArtifact,
    BodyFrame,
    BodyModelInfo,
    CameraIntrinsics,
    DetectionSummary,
    EstimatorInfo,
    ParameterArrays,
    PersonRecord,
)

__all__ = [
    "CONVENTION",
    "BodiesArtifact",
    "BodyFrame",
    "BodyModelInfo",
    "CameraIntrinsics",
    "DetectionSummary",
    "EstimatorInfo",
    "ParameterArrays",
    "PersonRecord",
]
