"""Tests for the bodies.json contract.

The file is read from another process and another language, so the validators
are what stop the stage writing something a reader would have to guess at. The
estimator itself is not asserted on; that is the evaluation's job.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from unproject.bodies.artifacts import (
    BODIES_SCHEMA_VERSION,
    CONVENTION,
    BodiesArtifact,
    BodyFrame,
    BodyModelInfo,
    CameraIntrinsics,
    DetectionSummary,
    EstimatorInfo,
    MeshArray,
    MeshAssets,
    ParameterArray,
    ParameterArrays,
    PersonRecord,
)


def _layout(person_count: int) -> ParameterArrays:
    """A minimal but self-consistent sidecar description."""
    return ParameterArrays(
        path="bodies.bin",
        person_count=person_count,
        arrays=[
            ParameterArray(
                name="root_translation",
                shape=(person_count, 3),
                byte_offset=0,
                byte_length=person_count * 3 * 4,
            )
        ],
    )


_DIGEST = "fd2d9ab7010a5f590a3db2480f5f476fd3ec6afbbde80babbec0d821ba9763d6"


def _body_model(bone_count: int = 4) -> BodyModelInfo:
    """A small but well-formed skeleton: a root and a chain hanging off it."""
    return BodyModelInfo(
        name="anny",
        version="0.6.0",
        bone_count=bone_count,
        shape_count=11,
        bone_parents=[-1, *range(bone_count - 1)],
    )


def _camera() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=1280, height=720, fx=6510.3, fy=6510.3, cx=640.0, cy=360.0, source="estimated"
    )


def _person(parameter_index: int, track_id: int = 0) -> PersonRecord:
    return PersonRecord(
        parameter_index=parameter_index,
        track_id=track_id,
        confidence=0.99,
        box_xyxy=(10.0, 20.0, 110.0, 320.0),
    )


def _frame(frame_index: int, people: list[PersonRecord]) -> BodyFrame:
    return BodyFrame(
        frame_index=frame_index,
        source_index=frame_index * 2,
        timestamp_seconds=frame_index * 0.08,
        camera=_camera(),
        people=people,
    )


def _artifact(frames: list[BodyFrame], **overrides: object) -> BodiesArtifact:
    people = sum(len(frame.people) for frame in frames)
    counts = [len(frame.people) for frame in frames]
    defaults: dict[str, object] = {
        "ingest_hash": "0140aa2c6860",
        "body_model": _body_model(),
        "estimator": EstimatorInfo(
            name="multihmr2",
            checkpoint_sha256=_DIGEST,
            backbone_repo="facebookresearch/dinov3",
            backbone_ref="6876159a11b4df116f30f667f8c9888617df0751",
            config_hash="deadbeef",
        ),
        "parameters": _layout(people),
        "frames": frames,
        "summary": DetectionSummary(
            frames_total=len(frames),
            frames_with_people=sum(1 for count in counts if count > 0),
            people_total=people,
            max_people_in_frame=max(counts, default=0),
            confidence_threshold=0.4,
            track_count=len({p.track_id for f in frames for p in f.people}),
        ),
    }
    return BodiesArtifact.model_validate(defaults | overrides)


def test_a_complete_artifact_round_trips() -> None:
    artifact = _artifact([_frame(0, [_person(0), _person(1, track_id=1)]), _frame(1, [_person(2)])])
    restored = BodiesArtifact.model_validate_json(artifact.model_dump_json())
    assert restored == artifact


def test_the_convention_is_named_in_the_file() -> None:
    """A file whose frame has to be guessed will eventually be read wrong."""
    artifact = _artifact([_frame(0, [_person(0)])])
    assert artifact.convention == CONVENTION == "camera_relative_opencv_metres"
    assert "convention" in artifact.model_dump_json()


def test_intrinsics_record_whether_they_were_estimated() -> None:
    assert _camera().source == "estimated"
    with pytest.raises(ValidationError):
        CameraIntrinsics(
            width=1280,
            height=720,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            source="guessed",
        )


def test_every_person_addresses_exactly_one_parameter_row() -> None:
    """A disagreement here reads as wrong parameters, not as an error."""
    frames = [_frame(0, [_person(0), _person(1)])]
    with pytest.raises(ValidationError, match="sidecar declares"):
        _artifact(frames, parameters=_layout(3))


def test_duplicate_parameter_rows_are_refused() -> None:
    # Two people pointing at one row would render one of them as the other.
    frames = [_frame(0, [_person(0), _person(0, track_id=1)])]
    with pytest.raises(ValidationError, match="permutation"):
        _artifact(frames)


def test_a_gap_in_the_parameter_rows_is_refused() -> None:
    frames = [_frame(0, [_person(0), _person(2)])]
    with pytest.raises(ValidationError, match="permutation"):
        _artifact(frames)


def test_the_summary_must_match_the_frames_it_summarises() -> None:
    frames = [_frame(0, [_person(0)])]
    with pytest.raises(ValidationError, match="summary reports"):
        _artifact(
            frames,
            summary=DetectionSummary(
                frames_total=9,
                frames_with_people=1,
                people_total=1,
                max_people_in_frame=1,
                confidence_threshold=0.4,
                track_count=1,
            ),
        )


def test_more_frames_with_people_than_frames_is_refused() -> None:
    with pytest.raises(ValidationError):
        DetectionSummary(
            frames_total=2,
            frames_with_people=3,
            people_total=3,
            max_people_in_frame=2,
            confidence_threshold=0.4,
            track_count=3,
        )


def test_a_future_schema_version_is_not_read() -> None:
    """Better to refuse than to read a file whose fields have moved."""
    with pytest.raises(ValidationError, match="not readable"):
        _artifact([_frame(0, [_person(0)])], schema_version=BODIES_SCHEMA_VERSION + 1)


def test_an_unknown_key_is_an_error_not_a_default() -> None:
    # extra="forbid": a typo must fail rather than fall back silently.
    with pytest.raises(ValidationError):
        _artifact([_frame(0, [_person(0)])], peopel=[])


def test_confidence_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(ValidationError):
        PersonRecord(parameter_index=0, track_id=0, confidence=1.5, box_xyxy=(0.0, 0.0, 1.0, 1.0))


def test_a_frame_with_nobody_in_it_is_legal() -> None:
    """An empty frame is a result, not a failure."""
    artifact = _artifact([_frame(0, []), _frame(1, [_person(0)])])
    assert artifact.summary.frames_with_people == 1
    assert artifact.summary.frames_total == 2


def test_the_sidecar_layout_must_be_contiguous() -> None:
    """A gap means every array after it is read from the wrong offset."""
    with pytest.raises(ValidationError, match="gap or overlap"):
        ParameterArrays(
            path="bodies.bin",
            person_count=2,
            arrays=[
                ParameterArray(name="a", shape=(2, 3), byte_offset=0, byte_length=24),
                ParameterArray(name="b", shape=(2, 3), byte_offset=99, byte_length=24),
            ],
        )


def test_a_declared_length_that_does_not_match_the_shape_is_refused() -> None:
    with pytest.raises(ValidationError, match="declares"):
        ParameterArrays(
            path="bodies.bin",
            person_count=2,
            arrays=[ParameterArray(name="a", shape=(2, 3), byte_offset=0, byte_length=8)],
        )


def test_an_array_must_have_a_row_per_person() -> None:
    with pytest.raises(ValidationError, match="rows against"):
        ParameterArrays(
            path="bodies.bin",
            person_count=5,
            arrays=[ParameterArray(name="a", shape=(2, 3), byte_offset=0, byte_length=24)],
        )


def test_an_array_can_be_looked_up_by_name() -> None:
    layout = _layout(4)
    assert layout.named("root_translation").shape == (4, 3)
    with pytest.raises(KeyError):
        layout.named("nope")


def test_a_bone_may_not_parent_itself_or_a_later_bone() -> None:
    """A forward reference would need two passes; a cycle would need many."""
    with pytest.raises(ValidationError, match="does not precede"):
        BodyModelInfo(
            name="anny", version="0.6.0", bone_count=3, shape_count=11, bone_parents=[-1, 2, 1]
        )


def test_the_parent_list_must_cover_every_bone() -> None:
    with pytest.raises(ValidationError, match="parents for"):
        BodyModelInfo(
            name="anny", version="0.6.0", bone_count=5, shape_count=11, bone_parents=[-1, 0]
        )


def test_minus_one_is_the_only_root_marker() -> None:
    with pytest.raises(ValidationError, match="only value below zero"):
        BodyModelInfo(
            name="anny", version="0.6.0", bone_count=2, shape_count=11, bone_parents=[-3, 0]
        )


def _mesh(tracks: list[int] | None = None, vertices: int = 3, faces: int = 1) -> MeshAssets:
    rows = [0] if tracks is None else tracks
    sizes = [
        ("faces", "<u2", (faces, 3), 2),
        ("skin_indices", "<u2", (vertices, 4), 2),
        ("skin_weights", "<f4", (vertices, 4), 4),
        ("rest_vertices", "<f4", (len(rows), vertices, 3), 4),
        ("rest_bone_poses", "<f4", (len(rows), 4, 4, 4), 4),
    ]
    arrays: list[MeshArray] = []
    offset = 0
    for name, dtype, shape, item in sizes:
        length = item
        for dimension in shape:
            length *= dimension
        arrays.append(
            MeshArray(
                name=name,
                dtype=dtype,
                shape=shape,
                byte_offset=offset,
                byte_length=length,
            )
        )
        offset += length
    return MeshAssets(
        path="mesh.bin",
        vertex_count=vertices,
        face_count=faces,
        influences=4,
        tracks=rows,
        arrays=arrays,
    )


class TestMesh:
    def test_a_complete_mesh_validates(self) -> None:
        assert _mesh([3, 9]).named("rest_vertices").shape[0] == 2

    def test_a_gap_between_arrays_is_refused(self) -> None:
        # The reader slices blind, so a wrong offset is wrong geometry rather
        # than an error.
        raw = _mesh().model_dump()
        raw["arrays"][1]["byte_offset"] = 999
        with pytest.raises(ValidationError, match="gap or overlap"):
            MeshAssets.model_validate(raw)

    def test_a_wrong_byte_length_is_refused(self) -> None:
        mesh = _mesh()
        arrays = [a.model_dump() for a in mesh.arrays]
        arrays[0]["byte_length"] = 7
        with pytest.raises(ValidationError, match="declares 7 bytes"):
            MeshAssets.model_validate(mesh.model_dump() | {"arrays": arrays})

    def test_rest_rows_must_match_the_tracks(self) -> None:
        mesh = _mesh([1, 2])
        with pytest.raises(ValidationError, match="against 3 tracks"):
            MeshAssets.model_validate(mesh.model_dump() | {"tracks": [1, 2, 3]})

    def test_two_rows_cannot_claim_the_same_track(self) -> None:
        # One of the two people would silently wear the other's body.
        raw = _mesh([4, 7]).model_dump()
        raw["tracks"] = [4, 4]
        with pytest.raises(ValidationError, match="same track"):
            MeshAssets.model_validate(raw)

    def test_a_run_without_a_mesh_is_still_readable(self) -> None:
        # The skeleton draws either way; a missing mesh is a degraded run.
        assert _artifact([_frame(0, [_person(0)])]).mesh is None

    def test_an_unknown_array_is_an_error_not_a_none(self) -> None:
        with pytest.raises(KeyError):
            _mesh().named("normals")

    def test_the_mesh_and_its_bind_bones_must_agree(self) -> None:
        # They come from one shape; different row counts mean they did not.
        raw = _mesh([1, 2]).model_dump()
        raw["arrays"][4]["shape"] = (3, 4, 4, 4)
        raw["arrays"][4]["byte_length"] = 3 * 4 * 4 * 4 * 4
        with pytest.raises(ValidationError, match="rest_bone_poses has 3 rows"):
            MeshAssets.model_validate(raw)


class TestExpectedFiles:
    """What the reuse check reads before trusting a descriptor.

    The bulk of a bodies run is in the sidecars, so a readable `bodies.json`
    says nothing about whether the parameters survived.
    """

    def test_the_parameter_sidecar_is_promised_with_its_size(self) -> None:
        artifact = _artifact([_frame(0, [_person(0), _person(1, track_id=1)])])

        assert artifact.expected_files() == [("bodies.bin", 2 * 3 * 4)]

    def test_the_mesh_is_promised_too_when_there_is_one(self) -> None:
        mesh = _mesh([3, 9])
        artifact = _artifact([_frame(0, [_person(0)])], mesh=mesh)

        promised = dict(artifact.expected_files())
        assert promised["mesh.bin"] == sum(array.byte_length for array in mesh.arrays)

    def test_a_run_without_a_mesh_promises_only_the_parameters(self) -> None:
        """A degraded run still draws skeletons; it must not promise a mesh."""
        artifact = _artifact([_frame(0, [_person(0)])])

        assert [name for name, _ in artifact.expected_files()] == ["bodies.bin"]
