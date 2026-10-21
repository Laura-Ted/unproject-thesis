"""Stage [B] driver: fit a body model to the people in every ingested frame.

A sibling of stage [2], not a successor: keyframe selection discards frames that
are too similar, which is the opposite of what motion capture wants. So this
stage reads stage [1] directly.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from unproject.bodies.artifacts import (
    ARTIFACT_FILENAME,
    MESH_FILENAME,
    PARAMETER_DTYPE,
    PARAMETERS_FILENAME,
    BodiesArtifact,
    BodyFrame,
    CameraIntrinsics,
    DetectionSummary,
    MeshArray,
    MeshAssets,
    ParameterArray,
    ParameterArrays,
    PersonRecord,
)
from unproject.bodies.multihmr import (
    BodyEstimationError,
    EstimatorSession,
    FloatArray,
    MultiHmrEstimator,
    PersonEstimate,
)
from unproject.config import BodiesConfig, ExperimentConfig, config_hash
from unproject.ingest.artifacts import IngestArtifact
from unproject.ingest.stage import run_ingest
from unproject.run.manifest import StageRecorder, write_manifest
from unproject.run.publish import staged_artifact
from unproject.run.reuse import unusable_reason

STAGE_NAME = "bodies"

_HASH_PREFIX_LENGTH = 12

logger = logging.getLogger(__name__)


class BodiesError(Exception):
    """Raised when the stage cannot produce a complete artifact."""


@dataclass(frozen=True, slots=True)
class BodiesResult:
    artifact_dir: Path
    artifact: BodiesArtifact
    reused: bool


def bodies_hash(ingest_hash: str, config: BodiesConfig) -> str:
    """Identity of a bodies run: its own config and the frames it was given.

    Without the ingest hash, bodies fitted to one set of frames would be reused
    for another. Changing the ingest stride changes the frames, so it has to
    change this too.
    """
    combined = f"{ingest_hash}:{config_hash(config)}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def bodies_artifact_dir(artifacts_root: Path, experiment: str, chained_hash: str) -> Path:
    return artifacts_root / experiment / STAGE_NAME / chained_hash[:_HASH_PREFIX_LENGTH]


def run_bodies(
    config: ExperimentConfig,
    *,
    config_dir: Path,
    artifacts_root: Path,
    overwrite: bool = False,
) -> BodiesResult:
    """Estimate bodies for the stage [1] frames and write them out.

    Runs stage [1] first, reusing its artifact when it exists.
    """
    ingested = run_ingest(
        config, config_dir=config_dir, artifacts_root=artifacts_root, overwrite=overwrite
    )
    upstream_hash = ingested.artifact_dir.name

    chained = bodies_hash(upstream_hash, config.bodies)
    artifact_dir = bodies_artifact_dir(artifacts_root, config.name, chained)

    if not overwrite:
        previous = _reusable(artifact_dir)
        if previous is not None:
            logger.info("reusing existing bodies artifact at %s", artifact_dir)
            return BodiesResult(artifact_dir=artifact_dir, artifact=previous, reused=True)

    estimator = MultiHmrEstimator(config.bodies, config_dir=config_dir)
    recorder = StageRecorder(
        STAGE_NAME, experiment=config.name, config_hash=chained, seed=config.seed
    )
    with staged_artifact(artifact_dir, descriptor=ARTIFACT_FILENAME) as staging:
        try:
            with recorder:
                artifact = _estimate(
                    estimator,
                    ingested.artifact,
                    ingested.artifact_dir,
                    upstream_hash,
                    staging,
                    config.bodies,
                )
        finally:
            _record_run(recorder, staging)

        (staging / ARTIFACT_FILENAME).write_text(
            artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    summary = artifact.summary
    logger.info(
        "found %d people over %d frames (max %d in one frame, %d identities) in %s",
        summary.people_total,
        summary.frames_total,
        summary.max_people_in_frame,
        summary.track_count,
        artifact_dir,
    )
    return BodiesResult(artifact_dir=artifact_dir, artifact=artifact, reused=False)


def _estimate(
    estimator: MultiHmrEstimator,
    ingested: IngestArtifact,
    frames_dir: Path,
    upstream_hash: str,
    artifact_dir: Path,
    config: BodiesConfig,
) -> BodiesArtifact:
    records = ingested.frames
    if not records:
        raise BodiesError(f"{frames_dir} holds no frames to estimate bodies for")

    frame_paths = [frames_dir / record.path for record in records]
    # The frames' own size: the source's, unless the ingest applied a geometry.
    frame_width, frame_height = ingested.frame_size()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    frames: list[BodyFrame] = []
    accumulator = _ParameterAccumulator()
    # Every shape a track was ever given, so the mesh can use its median rather
    # than whichever frame happened to be last.
    shapes: dict[int, list[FloatArray]] = defaultdict(list)

    try:
        estimates = estimator.estimate(frame_paths)
        for position, (record, estimate) in enumerate(zip(records, estimates, strict=True)):
            people = [
                PersonRecord(
                    parameter_index=accumulator.append(person),
                    track_id=person.track_id,
                    confidence=person.confidence,
                    box_xyxy=person.box_xyxy,
                )
                for person in estimate.people
            ]
            for person in estimate.people:
                shapes[person.track_id].append(person.shape)
            focal_x, focal_y, centre_x, centre_y = estimate.intrinsics
            frames.append(
                BodyFrame(
                    frame_index=position,
                    source_index=record.source_index,
                    timestamp_seconds=record.timestamp_seconds,
                    camera=CameraIntrinsics(
                        width=frame_width,
                        height=frame_height,
                        fx=focal_x,
                        fy=focal_y,
                        cx=centre_x,
                        cy=centre_y,
                        # The model predicts a field of view per frame unless
                        # the config hands it one.
                        source="known" if config.intrinsics is not None else "estimated",
                    ),
                    people=people,
                )
            )
    except BodyEstimationError as error:
        raise BodiesError(f"multihmr2 could not estimate bodies: {error}") from error

    parameters = accumulator.write(artifact_dir / PARAMETERS_FILENAME)
    # The model is already loaded, so this is a fresh tracker over the same
    # weights; nothing here advances it.
    mesh = _write_mesh(estimator.open(), shapes, artifact_dir)
    people_per_frame = [len(frame.people) for frame in frames]

    body_model = estimator.body_model()
    if (accumulator.bone_count, accumulator.shape_count) != (
        body_model.bone_count,
        body_model.shape_count,
    ):
        # The skeleton is read off the model and the arrays come from its
        # output; if they disagree the file would describe a different body from
        # the one it holds.
        raise BodiesError(
            f"the model declares {body_model.bone_count} bones and "
            f"{body_model.shape_count} shape parameters, but the arrays hold "
            f"{accumulator.bone_count} and {accumulator.shape_count}"
        )

    return BodiesArtifact(
        ingest_hash=upstream_hash,
        body_model=body_model,
        estimator=estimator.describe(),
        parameters=parameters,
        mesh=mesh,
        frames=frames,
        summary=DetectionSummary(
            frames_total=len(frames),
            frames_with_people=sum(1 for count in people_per_frame if count > 0),
            people_total=sum(people_per_frame),
            max_people_in_frame=max(people_per_frame, default=0),
            confidence_threshold=config.confidence_threshold,
            track_count=len(shapes),
        ),
    )


def _write_mesh(
    session: EstimatorSession, shapes: dict[int, list[FloatArray]], artifact_dir: Path
) -> MeshAssets | None:
    """Write the body itself: topology, skinning, and a rest pose per track.

    The median shape of a track rather than any one frame's. Shape is a property
    of a person and the estimator re-guesses it every frame, so the median is
    both steadier and cheaper than storing one mesh per detection - a fortieth
    of the space on this clip.
    """
    if not shapes:
        return None

    topology = session.topology()
    tracks = sorted(shapes)
    # One shape per track, and both rest arrays taken from it: pairing the two
    # keeps the reconstruction to about a millimetre.
    medians = [np.median(np.stack(shapes[track]), axis=0) for track in tracks]
    rest = np.stack([session.rest_vertices(shape)[topology.kept_vertices] for shape in medians])
    rest_bones = np.stack([session.rest_bone_poses(shape) for shape in medians])

    arrays: list[MeshArray] = []
    offset = 0
    payload: list[bytes] = []
    for name, block, dtype in (
        ("faces", topology.faces, "<u2"),
        ("skin_indices", topology.skin_indices, "<u2"),
        ("skin_weights", topology.skin_weights, "<f4"),
        ("rest_vertices", rest.astype(np.float32), "<f4"),
        ("rest_bone_poses", rest_bones.astype(np.float32), "<f4"),
    ):
        raw = np.ascontiguousarray(block).tobytes()
        arrays.append(
            MeshArray(
                name=name,
                dtype=cast(Literal["<f4", "<u2"], dtype),
                shape=tuple(int(size) for size in block.shape),
                byte_offset=offset,
                byte_length=len(raw),
            )
        )
        payload.append(raw)
        offset += len(raw)

    (artifact_dir / MESH_FILENAME).write_bytes(b"".join(payload))
    return MeshAssets(
        path=MESH_FILENAME,
        vertex_count=int(rest.shape[1]),
        face_count=int(topology.faces.shape[0]),
        influences=int(topology.skin_weights.shape[1]),
        tracks=tracks,
        arrays=arrays,
    )


class _ParameterAccumulator:
    """Collects the bulk arrays, one row per person, in the order they are seen.

    The rows are what `PersonRecord.parameter_index` addresses. Held in memory
    until the end so each array lands contiguously; the run manifest records
    the peak if that ever becomes the binding constraint.
    """

    def __init__(self) -> None:
        self._bone_poses: list[FloatArray] = []
        self._rest_bone_poses: list[FloatArray] = []
        self._shape: list[FloatArray] = []
        self._root_translation: list[FloatArray] = []
        self._joints_3d: list[FloatArray] = []

    def append(self, person: PersonEstimate) -> int:
        index = len(self._bone_poses)
        self._bone_poses.append(person.bone_poses)
        self._rest_bone_poses.append(person.rest_bone_poses)
        self._shape.append(person.shape)
        self._root_translation.append(person.root_translation)
        self._joints_3d.append(person.joints_3d)
        return index

    @property
    def person_count(self) -> int:
        return len(self._bone_poses)

    @property
    def bone_count(self) -> int:
        return int(self._bone_poses[0].shape[0]) if self._bone_poses else 0

    @property
    def shape_count(self) -> int:
        return int(self._shape[0].shape[0]) if self._shape else 0

    def write(self, path: Path) -> ParameterArrays:
        """Write the flat buffer and describe what went into it."""
        if not self._bone_poses:
            raise BodiesError(
                "no people were found in any frame, so there are no parameters to write"
            )
        stacked = {
            "bone_poses": np.stack(self._bone_poses),
            "rest_bone_poses": np.stack(self._rest_bone_poses),
            "shape": np.stack(self._shape),
            "root_translation": np.stack(self._root_translation),
            "joints_3d": np.stack(self._joints_3d),
        }

        described: list[ParameterArray] = []
        offset = 0
        with path.open("wb") as handle:
            for name, array in stacked.items():
                payload = np.ascontiguousarray(array, dtype=PARAMETER_DTYPE).tobytes()
                handle.write(payload)
                described.append(
                    ParameterArray(
                        name=name,
                        shape=tuple(int(size) for size in array.shape),
                        byte_offset=offset,
                        byte_length=len(payload),
                    )
                )
                offset += len(payload)

        return ParameterArrays(path=path.name, person_count=self.person_count, arrays=described)


def read_artifact(path: Path) -> BodiesArtifact | None:
    """Load a previous artifact, or None if this build cannot use it."""
    try:
        return BodiesArtifact.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        logger.info("cannot reuse %s, re-running the stage: %s", path, error)
        return None


def _reusable(artifact_dir: Path) -> BodiesArtifact | None:
    """Load a previous artifact, or None if it is unreadable or incomplete.

    The descriptor is JSON and the bulk is in sidecars beside it, so a readable
    descriptor says nothing about whether the parameters survived.
    """
    path = artifact_dir / ARTIFACT_FILENAME
    if not path.is_file():
        return None
    artifact = read_artifact(path)
    if artifact is None:
        return None

    reason = unusable_reason(artifact_dir, artifact.expected_files())
    if reason is not None:
        logger.info("cannot reuse %s, re-running the stage: %s", artifact_dir, reason)
        return None
    return artifact


def _record_run(recorder: StageRecorder, artifact_dir: Path) -> None:
    try:
        write_manifest(recorder.finish(), artifact_dir)
    except OSError as error:
        logger.warning("could not write the run manifest to %s: %s", artifact_dir, error)
