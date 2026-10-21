"""On-disk contract produced by stage [2]."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.ingest.types import SourceMetadata

# 2 added KeyframeRecord.held_out, which stage [4] has to honour.
# 3 added KeyframeRecord.byte_length, so a reader can tell a complete keyframe
# from one a full disk cut short.
PREPROCESS_SCHEMA_VERSION = 3

ARTIFACT_FILENAME = "preprocess.json"
KEYFRAMES_DIRNAME = "keyframes"


class KeyframeRecord(BaseModel):
    """One keyframe, with the measurements that selected it.

    The metrics travel with the record so a bad reconstruction can be explained
    rather than only scored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    keyframe_index: int = Field(ge=0)
    # Index in the original source, so a keyframe traces back to the footage.
    source_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    path: str
    byte_length: int = Field(ge=0)
    sharpness: float = Field(ge=0.0)
    displacement_px: float = Field(ge=0.0)
    tracked_ratio: float = Field(ge=0.0, le=1.0)
    forced: bool = False
    # Withheld from training so stage [5] scores a view the model never saw.
    # With millions of free parameters a model can fit its training views and
    # still be wrong from every other angle.
    held_out: bool = False


class ResizeSummary(BaseModel):
    """What was done to the frame size.

    Camera intrinsics are in pixels and scale with this factor, so a run that
    resized without recording it cannot be interpreted afterwards.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scale: float = Field(gt=0.0, le=1.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SelectionSummary(BaseModel):
    """How the frame count came down, and under which thresholds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frames_considered: int = Field(ge=0)
    keyframes_kept: int = Field(ge=0)
    forced_keyframes: int = Field(ge=0)
    held_out_every: int | None = None
    held_out_count: int = Field(default=0, ge=0)
    min_disparity_px: float = Field(gt=0.0)
    min_tracked_ratio: float = Field(gt=0.0, le=1.0)
    blur_percentile: float = Field(ge=0.0, lt=100.0)
    # The lowest sharpness among the kept keyframes. Not the threshold the
    # percentile resolved to, which `select_keyframes` does not return: a forced
    # keyframe sits below that threshold.
    blur_floor: float = Field(ge=0.0)
    metric_long_edge: int = Field(gt=0)


class PreprocessArtifact(BaseModel):
    """Stage [2] output descriptor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=PREPROCESS_SCHEMA_VERSION, ge=1)
    # Which stage [1] artifact these keyframes came from.
    ingest_hash: str
    # Copied rather than referenced, so the artifact can be read on its own.
    source: SourceMetadata
    resize: ResizeSummary
    # Undistortion needs intrinsics that only stage [3] produces. The field is
    # reserved so filling it in later is not a schema change.
    undistortion: None = None
    selection: SelectionSummary
    image_format: str = "png"
    keyframes: list[KeyframeRecord]

    def expected_files(self) -> list[tuple[str, int]]:
        """Every file this artifact promises, with the size it promises."""
        return [(keyframe.path, keyframe.byte_length) for keyframe in self.keyframes]

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != PREPROCESS_SCHEMA_VERSION:
            msg = (
                f"preprocess artifact schema_version {self.schema_version} is not "
                f"readable by this build (expected {PREPROCESS_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self
