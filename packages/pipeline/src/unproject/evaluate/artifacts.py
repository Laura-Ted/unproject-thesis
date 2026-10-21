"""On-disk contract produced by stage [5].

Holds the pose comparison of the preliminary scene-reconstruction work
(ADR-0002 §1): how far a learned backend's cameras land from the classical
baseline's, on the same keyframes. Reconstruction quality against held-out views
joins it when stage [4] exists, which will bump the schema version.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

METRICS_SCHEMA_VERSION = 1

ARTIFACT_FILENAME = "metrics.json"


class RunReference(BaseModel):
    """Which pose run a side of the comparison came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    version: str
    artifact_dir: str
    keyframes_registered: int = Field(ge=0)


class PoseErrors(BaseModel):
    """How far apart the two runs put the same camera.

    Translation is quoted as a fraction of the reference trajectory's extent as
    well as raw: the scene's units are arbitrary, so the raw number is only
    comparable within one scene and the fraction is comparable across scenes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    translation_mean: float = Field(ge=0.0)
    translation_median: float = Field(ge=0.0)
    translation_max: float = Field(ge=0.0)
    translation_mean_fraction: float = Field(ge=0.0)
    rotation_mean_degrees: float = Field(ge=0.0)
    rotation_median_degrees: float = Field(ge=0.0)
    rotation_max_degrees: float = Field(ge=0.0)


class KeyframeError(BaseModel):
    """One camera's error, so drift can be located and not only totalled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keyframe_index: int = Field(ge=0)
    translation: float = Field(ge=0.0)
    rotation_degrees: float = Field(ge=0.0)


class PoseComparison(BaseModel):
    """The alignment between two runs and what it leaves behind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference: RunReference
    candidate: RunReference
    # Only keyframes both runs placed. COLMAP can decline to register one, so
    # the two sets are not always the same and the count belongs in the result.
    keyframes_compared: int = Field(ge=0)
    keyframes_reference_only: int = Field(ge=0)
    keyframes_candidate_only: int = Field(ge=0)
    # Scale of the similarity taking the candidate onto the reference. Neither
    # backend knows the true scale, so this is a free parameter, not an error.
    scale: float = Field(gt=0.0)
    reference_extent: float = Field(ge=0.0)
    errors: PoseErrors
    per_keyframe: list[KeyframeError]

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if len(self.per_keyframe) != self.keyframes_compared:
            msg = (
                f"{len(self.per_keyframe)} per-keyframe entries against "
                f"{self.keyframes_compared} compared"
            )
            raise ValueError(msg)
        return self


class MetricsArtifact(BaseModel):
    """Stage [5] output descriptor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=METRICS_SCHEMA_VERSION, ge=1)
    experiment: str
    preprocess_hash: str
    pose: PoseComparison

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != METRICS_SCHEMA_VERSION:
            msg = (
                f"metrics artifact schema_version {self.schema_version} is not readable by "
                f"this build (expected {METRICS_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self
