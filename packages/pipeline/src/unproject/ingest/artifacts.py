"""On-disk contract produced by stage [1].

Stage [2] reads this and nothing else about the input, which is what keeps the
stages independently resumable.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.config import FrameGeometry
from unproject.ingest.types import SourceMetadata

# 2 added SourceMetadata.kind, which tells a reader whether the frames came from
# a video or from a directory of images.
# 3 added FrameRecord.byte_length, so a reader can tell a complete frame from
# one a full disk cut short.
INGEST_SCHEMA_VERSION = 3

ARTIFACT_FILENAME = "ingest.json"
FRAMES_DIRNAME = "frames"


class FrameRecord(BaseModel):
    """One extracted frame, as later stages refer to it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    # Relative to the artifact directory, so it can be moved as a whole.
    path: str
    byte_length: int = Field(ge=0)


class SelectionSummary(BaseModel):
    """Which slice of the source was kept, and why it stopped where it did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_stride: int = Field(ge=1)
    start_time_seconds: float | None = None
    end_time_seconds: float | None = None
    max_frames: int | None = None
    frames_decoded: int = Field(ge=0)
    frames_written: int = Field(ge=0)
    # Set when `max_frames` cut the run short, so "the whole scene" and "the
    # first N frames of it" stay distinguishable.
    truncated: bool = False


class IngestArtifact(BaseModel):
    """Stage [1] output descriptor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=INGEST_SCHEMA_VERSION, ge=1)
    source: SourceMetadata
    selection: SelectionSummary
    # Lossless. With JPEG, the held-out metrics of stage [5] would partly be
    # measuring compression artifacts.
    image_format: str = "png"
    # The geometry applied to every written frame, and the size that made
    # them; absent (older artifacts, or no geometry) the frames are the
    # source's size. Optional so artifacts written before the field existed
    # still read.
    geometry: FrameGeometry | None = None
    frame_width: int | None = Field(default=None, gt=0)
    frame_height: int | None = Field(default=None, gt=0)
    frames: list[FrameRecord]

    def expected_files(self) -> list[tuple[str, int]]:
        """Every file this artifact promises, with the size it promises."""
        return [(frame.path, frame.byte_length) for frame in self.frames]

    def frame_size(self) -> tuple[int, int]:
        """Width and height of the frames on disk: the source's unless a geometry was applied."""
        if self.frame_width is not None and self.frame_height is not None:
            return self.frame_width, self.frame_height
        return self.source.width, self.source.height

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != INGEST_SCHEMA_VERSION:
            msg = (
                f"ingest artifact schema_version {self.schema_version} is not readable "
                f"by this build (expected {INGEST_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self
