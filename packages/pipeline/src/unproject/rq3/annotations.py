"""What a person was actually doing, written down by hand.

RQ3 needs something to compare the model against, and for a controlled scene
that something is a tape measure and a stopwatch. This is the file those go
into.

**Coordinates and units.** Marker positions are in the same frame the artifact
uses - camera-relative OpenCV metres, +X right, +Y down, +Z forward - measured
from the camera rather than from a room corner. That is a deliberate limit: it
avoids needing the camera's pose in a world frame, which nothing here estimates,
at the cost of only being able to describe scenes shot from one fixed camera.

Every measured quantity carries its own uncertainty. A position error smaller
than the uncertainty of the thing it is measured against is not a result.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANNOTATION_SCHEMA_VERSION = 1

CONVENTION = "camera_relative_opencv_metres"


class AnnotationError(Exception):
    """Raised when an annotation file is unreadable or contradicts itself."""


class Marker(BaseModel):
    """One measured spot on the floor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    # Metres from the camera: +X right, +Z forward. Y is the floor and is not
    # measured - a marker is on it by definition.
    x_metres: float
    z_metres: float = Field(gt=0.0)
    # How well the tape was read, and how well the mark was placed. One number
    # for both because they are not separable in practice.
    uncertainty_metres: float = Field(gt=0.0)


class StationaryInterval(BaseModel):
    """A stretch where somebody stood still on a named marker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    marker: str
    # Source-relative seconds, matching the artifact's own timestamps.
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    # Which reconstructed track is this person. Left out when the recording has
    # one participant and the answer is "whoever is there".
    track_id: int | None = Field(default=None, ge=0)
    # How confidently the start and end were read off the footage.
    boundary_uncertainty_seconds: float = Field(default=0.1, ge=0.0)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            msg = (
                f"interval on {self.marker} ends at {self.end_seconds} s, "
                f"which is not after {self.start_seconds} s"
            )
            raise ValueError(msg)
        return self


class MotionInterval(BaseModel):
    """A stretch where somebody walked between two named markers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_marker: str
    to_marker: str
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    track_id: int | None = Field(default=None, ge=0)
    boundary_uncertainty_seconds: float = Field(default=0.1, ge=0.0)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            msg = (
                f"walk from {self.from_marker} to {self.to_marker} ends at "
                f"{self.end_seconds} s, which is not after {self.start_seconds} s"
            )
            raise ValueError(msg)
        if self.from_marker == self.to_marker:
            msg = f"walk from {self.from_marker} to itself is not a walk"
            raise ValueError(msg)
        return self


class Scene(BaseModel):
    """The recording conditions, for the threats-to-validity section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str = ""
    # Where the camera sat. Not used in any calculation, because markers are
    # measured from the camera; recorded so the scene can be rebuilt.
    camera_height_metres: float | None = Field(default=None, gt=0.0)
    camera_placement_uncertainty_metres: float | None = Field(default=None, gt=0.0)
    participants: int = Field(default=1, ge=1)
    annotated_by: str = ""
    annotated_on: str = ""


class Annotation(BaseModel):
    """One controlled recording, described well enough to score against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=ANNOTATION_SCHEMA_VERSION, ge=1)
    # Repeated in the file so it can be read without consulting this module.
    convention: str = CONVENTION
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    scene: Scene = Scene()
    markers: list[Marker]
    stationary: list[StationaryInterval] = []
    motion: list[MotionInterval] = []

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != ANNOTATION_SCHEMA_VERSION:
            msg = (
                f"annotation schema_version {self.schema_version} is not readable "
                f"by this build (expected {ANNOTATION_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        if self.convention != CONVENTION:
            msg = f"annotations are {CONVENTION}; this file says {self.convention}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_markers_exist(self) -> Self:
        """Every interval names a marker the file declares.

        A typo would otherwise silently drop an interval out of the analysis,
        which is a smaller sample reported as a full one.
        """
        known = {marker.id for marker in self.markers}
        if len(known) != len(self.markers):
            raise ValueError("two markers share an id")

        named = {interval.marker for interval in self.stationary}
        for walk in self.motion:
            named |= {walk.from_marker, walk.to_marker}
        unknown = sorted(named - known)
        if unknown:
            msg = f"intervals name marker(s) {unknown} that the file does not declare"
            raise ValueError(msg)
        return self

    def marker(self, marker_id: str) -> Marker:
        for marker in self.markers:
            if marker.id == marker_id:
                return marker
        raise KeyError(f"no marker named {marker_id!r}")


def load_annotation(path: Path) -> Annotation:
    """Parse and validate an annotation TOML."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise AnnotationError(f"cannot read annotations {path}: {error}") from error

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise AnnotationError(f"{path} is not valid TOML: {error}") from error

    try:
        return Annotation.model_validate(data)
    except ValueError as error:
        raise AnnotationError(f"{path} failed validation:\n{error}") from error
