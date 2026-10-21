"""On-disk contract produced by stage [B].

Three files: this one holds what a reader has to understand, `bodies.bin` holds
the bulk arrays and `mesh.bin` the mesh. A person is around 5,700 floats of pose,
and JSON is the wrong container for that.
"""

from __future__ import annotations

from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 2 added the mesh: topology, skinning weights, and a rest pose per track.
BODIES_SCHEMA_VERSION = 2

ARTIFACT_FILENAME = "bodies.json"
PARAMETERS_FILENAME = "bodies.bin"
MESH_FILENAME = "mesh.bin"

# How many bones may pull on one vertex. Anny gives twelve; a GPU skins four,
# and dropping to four moves the worst vertex by 30 mm on a 1700 mm body.
MESH_INFLUENCES = 4

# Little-endian float32 throughout. A raw buffer rather than .npz because both
# readers want it: numpy takes it with frombuffer, and the viewer fetches it into
# a Float32Array without a zip reader in the way.
PARAMETER_DTYPE: Final = "<f4"

# Camera-relative: there is no world frame, only the camera that shot the
# footage. OpenCV axes, x right, y down, z forward. Metres, because the body
# prior fixes the scale.
#
# Settled by measurement rather than by reading upstream: joints projected
# through the reported intrinsics land on the reported 2D joints to 0.00 px,
# and reading them as pelvis-relative gives 45 px.
CONVENTION: Final = "camera_relative_opencv_metres"


class BodyModelInfo(BaseModel):
    """Which parametric body the parameters drive, and how its skeleton hangs together.

    Recorded rather than assumed from the name. A default `Anny()` has 104 bones
    and the one this estimator configures has 163, so a reader that assumed
    would be wrong in a way that still renders.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["anny"]
    version: str
    bone_count: int = Field(gt=0)
    shape_count: int = Field(gt=0)
    # Parent of each bone, -1 for a root. Without this the pose arrays are a
    # list of transforms with no tree, and nothing can be drawn from them.
    bone_parents: list[int]

    @model_validator(mode="after")
    def _check_tree(self) -> Self:
        if len(self.bone_parents) != self.bone_count:
            msg = f"{len(self.bone_parents)} parents for {self.bone_count} bones"
            raise ValueError(msg)
        for bone, parent in enumerate(self.bone_parents):
            # A parent that came later would need two passes to resolve, and a
            # cycle would need a great many more.
            if parent >= bone:
                msg = f"bone {bone} claims parent {parent}, which does not precede it"
                raise ValueError(msg)
            if parent < -1:
                msg = f"bone {bone} has parent {parent}; -1 is the only value below zero"
                raise ValueError(msg)
        return self


class EstimatorInfo(BaseModel):
    """Which model produced the parameters, pinned tightly enough to reproduce."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["multihmr2"]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # The backbone is fetched as source at build time, so the ref it was fetched
    # at is part of what ran.
    backbone_repo: str
    backbone_ref: str
    config_hash: str


class CameraIntrinsics(BaseModel):
    """The camera the translations are expressed relative to.

    Per frame, because this estimator predicts a field of view per frame.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0.0)
    fy: float = Field(gt=0.0)
    cx: float
    cy: float
    # A file whose assumptions have to be guessed will eventually be read wrong.
    # `known`: the operator supplied the camera and the model was told it,
    # rather than predicting one.
    source: Literal["estimated", "assumed", "known"]


class PersonRecord(BaseModel):
    """One detected person in one frame.

    Pose, shape and translation are rows in the sidecar, addressed by
    `parameter_index`. Everything small enough to read by eye is here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_index: int = Field(ge=0)
    # What the tracker asserted, not the truth: a new id after an occlusion is a
    # new person as far as this file is concerned.
    track_id: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    box_xyxy: tuple[float, float, float, float]


class BodyFrame(BaseModel):
    """Every person found in one frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_index: int = Field(ge=0)
    # Index in the original footage, so a body can be traced back to its frame.
    source_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    camera: CameraIntrinsics
    people: list[PersonRecord]


class ParameterArray(BaseModel):
    """One array inside the sidecar, described well enough to slice out blind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    dtype: Literal["<f4"] = PARAMETER_DTYPE
    # First dimension is always the person; the rest are the array's own shape.
    shape: tuple[int, ...]
    byte_offset: int = Field(ge=0)
    byte_length: int = Field(ge=0)


class ParameterArrays(BaseModel):
    """The sidecar's layout, so a reader never has to guess at the bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    # Rows across every array: the total number of people over all frames.
    person_count: int = Field(ge=0)
    arrays: list[ParameterArray]

    @model_validator(mode="after")
    def _check_layout(self) -> Self:
        offset = 0
        for array in self.arrays:
            if array.shape and array.shape[0] != self.person_count:
                msg = (
                    f"array {array.name!r} has {array.shape[0]} rows against "
                    f"{self.person_count} people"
                )
                raise ValueError(msg)
            expected = 4
            for dimension in array.shape:
                expected *= dimension
            if array.byte_length != expected:
                msg = f"array {array.name!r} declares {array.byte_length} bytes, needs {expected}"
                raise ValueError(msg)
            if array.byte_offset != offset:
                msg = (
                    f"array {array.name!r} starts at {array.byte_offset}, "
                    f"leaving a gap or overlap at {offset}"
                )
                raise ValueError(msg)
            offset += array.byte_length
        return self

    def named(self, name: str) -> ParameterArray:
        for array in self.arrays:
            if array.name == name:
                return array
        raise KeyError(f"the sidecar holds no array named {name!r}")


class MeshArray(BaseModel):
    """One array inside `mesh.bin`, described well enough to slice out blind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    # Vertex and bone indices are counts, not measurements, and fit in a short.
    dtype: Literal["<f4", "<u2"]
    shape: tuple[int, ...]
    byte_offset: int = Field(ge=0)
    byte_length: int = Field(ge=0)

    @property
    def item_size(self) -> int:
        return 4 if self.dtype == "<f4" else 2


class MeshAssets(BaseModel):
    """The body the poses drive: topology, skinning, and a rest pose per track.

    Everything here is what the parameters are *missing* in order to be a body
    rather than a list of joint transforms. Shared across every frame, because
    topology and skinning do not change and a person's shape does not either.

    Rest vertices are per track rather than per detection, from the track's
    median shape. That is not only a saving - 9 MB against 863 - it is steadier:
    measured on the reference clip, the model's shape estimate varies as much
    across frames of one person as it does between different people, so a
    per-frame build is mostly noise. It costs 15 mm on a median vertex and 122
    at worst, against a depth error that is already 300 mm at twelve metres.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    vertex_count: int = Field(gt=0)
    face_count: int = Field(gt=0)
    influences: int = Field(gt=0)
    # Which track each row of `rest_vertices` belongs to, in order.
    tracks: list[int]
    arrays: list[MeshArray]

    @model_validator(mode="after")
    def _check_layout(self) -> Self:
        offset = 0
        for array in self.arrays:
            expected = array.item_size
            for dimension in array.shape:
                expected *= dimension
            if array.byte_length != expected:
                msg = f"array {array.name!r} declares {array.byte_length} bytes, needs {expected}"
                raise ValueError(msg)
            if array.byte_offset != offset:
                msg = (
                    f"array {array.name!r} starts at {array.byte_offset}, "
                    f"leaving a gap or overlap at {offset}"
                )
                raise ValueError(msg)
            offset += array.byte_length
        return self

    @model_validator(mode="after")
    def _check_rest_rows(self) -> Self:
        """One rest pose per track, and the mesh and its bind bones agreeing.

        The two are taken from the same shape on purpose. A rest mesh paired
        with bind bones from a different shape is what turns a millimetre of
        error into fifteen, so a file where the counts disagree is a file where
        that has probably happened.
        """
        if len(set(self.tracks)) != len(self.tracks):
            raise ValueError("two rows of rest_vertices claim the same track")
        for name in ("rest_vertices", "rest_bone_poses"):
            array = next((a for a in self.arrays if a.name == name), None)
            if array is None:
                raise ValueError(f"the mesh has no {name}")
            if array.shape[:1] != (len(self.tracks),):
                msg = f"{name} has {array.shape[0]} rows against {len(self.tracks)} tracks"
                raise ValueError(msg)
        return self

    def named(self, name: str) -> MeshArray:
        for array in self.arrays:
            if array.name == name:
                return array
        raise KeyError(f"the mesh holds no array named {name!r}")


class DetectionSummary(BaseModel):
    """How much was found, and how much of it was believed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frames_total: int = Field(ge=0)
    frames_with_people: int = Field(ge=0)
    people_total: int = Field(ge=0)
    max_people_in_frame: int = Field(ge=0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    # Much larger than `max_people_in_frame` means identities are churning.
    track_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if self.frames_with_people > self.frames_total:
            msg = (
                f"{self.frames_with_people} frames with people out of "
                f"{self.frames_total}, which is more than were processed"
            )
            raise ValueError(msg)
        return self


class BodiesArtifact(BaseModel):
    """Stage [B] output descriptor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=BODIES_SCHEMA_VERSION, ge=1)
    # Repeated in the file so it can be read without consulting the source.
    convention: Literal["camera_relative_opencv_metres"] = CONVENTION
    # Stage [B] reads stage [1] directly; keyframe selection discards exactly the
    # frames motion needs.
    ingest_hash: str
    body_model: BodyModelInfo
    estimator: EstimatorInfo
    parameters: ParameterArrays
    # Absent when the estimator could not describe its own body, which is a
    # degraded run rather than a broken one: the skeleton still draws.
    mesh: MeshAssets | None = None
    frames: list[BodyFrame]
    summary: DetectionSummary

    def expected_files(self) -> list[tuple[str, int]]:
        """Every sidecar this artifact promises, with the size its layout implies."""
        expected = [
            (
                self.parameters.path,
                sum(array.byte_length for array in self.parameters.arrays),
            )
        ]
        if self.mesh is not None:
            expected.append((self.mesh.path, sum(array.byte_length for array in self.mesh.arrays)))
        return expected

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != BODIES_SCHEMA_VERSION:
            msg = (
                f"bodies artifact schema_version {self.schema_version} is not readable "
                f"by this build (expected {BODIES_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_parameter_rows(self) -> Self:
        """Every person addresses one row, and every row a person.

        A mismatch reads as one person wearing another's pose rather than as an
        error, so it is caught here.
        """
        indices = [person.parameter_index for frame in self.frames for person in frame.people]
        if len(indices) != self.parameters.person_count:
            msg = (
                f"{len(indices)} people listed but the sidecar declares "
                f"{self.parameters.person_count} rows"
            )
            raise ValueError(msg)
        if sorted(indices) != list(range(self.parameters.person_count)):
            msg = "parameter_index values are not a permutation of the sidecar's rows"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_summary(self) -> Self:
        if len(self.frames) != self.summary.frames_total:
            msg = (
                f"{len(self.frames)} frames stored but the summary reports "
                f"{self.summary.frames_total}"
            )
            raise ValueError(msg)
        counted = sum(len(frame.people) for frame in self.frames)
        if counted != self.summary.people_total:
            msg = f"{counted} people stored but the summary reports {self.summary.people_total}"
            raise ValueError(msg)
        return self
