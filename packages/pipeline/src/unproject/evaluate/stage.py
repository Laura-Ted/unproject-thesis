"""Stage [5] driver: measure one pose backend against the other.

Takes two experiment configs that differ only in their pose section and reports
how far apart their runs put the same cameras. Requiring the configs to agree
everywhere else is the point: if the two runs saw different keyframes, the
difference between them is not the backend.

Unlike the earlier stages this one does **not** run its predecessors. pycolmap
and torch each bundle a copy of libomp and abort the process if both initialise
(see `docs/reproduce.md` §6), so a step that needs results from both backends
cannot produce them itself. It reads what is on disk and says which command to
run when something is missing.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unproject.config import ExperimentConfig, config_hash
from unproject.evaluate.alignment import (
    AlignmentError,
    Points,
    rotation_error_degrees,
    similarity_transform,
    trajectory_extent,
)
from unproject.evaluate.artifacts import (
    ARTIFACT_FILENAME,
    KeyframeError,
    MetricsArtifact,
    PoseComparison,
    PoseErrors,
    RunReference,
)
from unproject.pose.artifacts import ARTIFACT_FILENAME as POSE_ARTIFACT_FILENAME
from unproject.pose.artifacts import CameraPose, PoseArtifact
from unproject.pose.geometry import camera_center, quaternion_to_matrix
from unproject.pose.stage import read_artifact, resolve_artifact_dir
from unproject.run.manifest import StageRecorder, write_manifest

STAGE_NAME = "evaluate"

_HASH_PREFIX_LENGTH = 12

logger = logging.getLogger(__name__)


class EvaluateError(Exception):
    """Raised when two runs cannot be compared."""


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """A pose run read back off disk."""

    artifact_dir: Path
    artifact: PoseArtifact


@dataclass(frozen=True, slots=True)
class EvaluateResult:
    artifact_dir: Path
    artifact: MetricsArtifact


def run_evaluate(
    reference: ExperimentConfig,
    candidate: ExperimentConfig,
    *,
    reference_dir: Path,
    candidate_dir: Path,
    artifacts_root: Path,
) -> EvaluateResult:
    """Compare two pose backends on the same scene.

    Each config directory is needed to locate its run: the ingest identity
    depends on the source content, so it cannot be derived from the config text
    alone.
    """
    _check_comparable(reference, candidate)

    reference_run = _load_run(reference, artifacts_root, reference_dir)
    candidate_run = _load_run(candidate, artifacts_root, candidate_dir)

    upstream = reference_run.artifact.preprocess_hash
    if candidate_run.artifact.preprocess_hash != upstream:
        raise EvaluateError(
            "the two runs were given different keyframes "
            f"({upstream} against {candidate_run.artifact.preprocess_hash}), "
            "so the difference between them is not the pose backend"
        )

    chained = hashlib.sha256(
        f"{reference_run.artifact_dir.name}:{candidate_run.artifact_dir.name}".encode()
    ).hexdigest()
    artifact_dir = artifacts_root / reference.name / STAGE_NAME / chained[:_HASH_PREFIX_LENGTH]

    recorder = StageRecorder(
        STAGE_NAME, experiment=reference.name, config_hash=chained, seed=reference.seed
    )
    try:
        with recorder:
            comparison = _compare(reference_run, candidate_run)
    finally:
        _record_run(recorder, artifact_dir)

    artifact = MetricsArtifact(experiment=reference.name, preprocess_hash=upstream, pose=comparison)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ARTIFACT_FILENAME).write_text(
        artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    errors = comparison.errors
    logger.info(
        "%s against %s on %d keyframes: %.4f mean position error (%.2f%% of extent), "
        "%.3f degrees mean rotation",
        comparison.candidate.backend,
        comparison.reference.backend,
        comparison.keyframes_compared,
        errors.translation_mean,
        100.0 * errors.translation_mean_fraction,
        errors.rotation_mean_degrees,
    )
    return EvaluateResult(artifact_dir=artifact_dir, artifact=artifact)


def _check_comparable(reference: ExperimentConfig, candidate: ExperimentConfig) -> None:
    """Refuse a comparison whose two sides did not see the same scene."""
    if reference.name != candidate.name:
        raise EvaluateError(
            f"the configs name different experiments, {reference.name!r} and {candidate.name!r}"
        )
    if config_hash(reference.ingest) != config_hash(candidate.ingest):
        raise EvaluateError("the configs differ in their ingest section")
    if config_hash(reference.preprocess) != config_hash(candidate.preprocess):
        raise EvaluateError("the configs differ in their preprocess section")
    if reference.pose.backend == candidate.pose.backend:
        logger.warning(
            "both sides use the %s backend; this measures run-to-run variation, "
            "not one backend against the other",
            reference.pose.backend,
        )


def _load_run(config: ExperimentConfig, artifacts_root: Path, config_dir: Path) -> LoadedRun:
    """Read a pose run this config would have produced, or say how to make it."""
    artifact_dir = resolve_artifact_dir(config, artifacts_root, config_dir)
    artifact = read_artifact(artifact_dir / POSE_ARTIFACT_FILENAME)
    if artifact is None:
        raise EvaluateError(
            f"no readable {config.pose.backend} pose run at {artifact_dir}; "
            f"run `unproject pose` on that config first"
        )
    return LoadedRun(artifact_dir=artifact_dir, artifact=artifact)


def _compare(reference: LoadedRun, candidate: LoadedRun) -> PoseComparison:
    shared = sorted(
        {pose.keyframe_index for pose in reference.artifact.poses}
        & {pose.keyframe_index for pose in candidate.artifact.poses}
    )
    if not shared:
        raise EvaluateError("the two runs placed no keyframe in common")

    reference_poses = {pose.keyframe_index: pose for pose in reference.artifact.poses}
    candidate_poses = {pose.keyframe_index: pose for pose in candidate.artifact.poses}

    reference_centres = _centres(reference_poses, shared)
    candidate_centres = _centres(candidate_poses, shared)

    try:
        similarity = similarity_transform(candidate_centres, reference_centres)
    except AlignmentError as error:
        raise EvaluateError(f"cannot align the two runs: {error}") from error

    aligned = similarity.apply(candidate_centres)
    translation_errors = np.linalg.norm(aligned - reference_centres, axis=1)

    rotation_errors = np.array(
        [
            rotation_error_degrees(
                # The candidate's world is the reference's turned by the
                # alignment, so its cameras have to be turned with it before
                # their orientations can be compared.
                quaternion_to_matrix(np.array(candidate_poses[index].quaternion_wxyz))
                @ similarity.rotation.T,
                quaternion_to_matrix(np.array(reference_poses[index].quaternion_wxyz)),
            )
            for index in shared
        ]
    )

    extent = trajectory_extent(reference_centres)
    return PoseComparison(
        reference=_describe(reference),
        candidate=_describe(candidate),
        keyframes_compared=len(shared),
        keyframes_reference_only=len(reference_poses) - len(shared),
        keyframes_candidate_only=len(candidate_poses) - len(shared),
        scale=similarity.scale,
        reference_extent=extent,
        errors=PoseErrors(
            translation_mean=float(translation_errors.mean()),
            translation_median=float(np.median(translation_errors)),
            translation_max=float(translation_errors.max()),
            translation_mean_fraction=(
                float(translation_errors.mean() / extent) if extent > 0.0 else 0.0
            ),
            rotation_mean_degrees=float(rotation_errors.mean()),
            rotation_median_degrees=float(np.median(rotation_errors)),
            rotation_max_degrees=float(rotation_errors.max()),
        ),
        per_keyframe=[
            KeyframeError(
                keyframe_index=index,
                translation=float(translation),
                rotation_degrees=float(rotation),
            )
            for index, translation, rotation in zip(
                shared, translation_errors, rotation_errors, strict=True
            )
        ],
    )


def _centres(poses: dict[int, CameraPose], indices: list[int]) -> Points:
    return np.array(
        [
            camera_center(
                np.array(poses[index].quaternion_wxyz, dtype=np.float64),
                np.array(poses[index].translation, dtype=np.float64),
            )
            for index in indices
        ],
        dtype=np.float64,
    )


def _describe(result: LoadedRun) -> RunReference:
    artifact: PoseArtifact = result.artifact
    return RunReference(
        backend=artifact.backend.name,
        version=artifact.backend.version,
        artifact_dir=result.artifact_dir.as_posix(),
        keyframes_registered=artifact.registration.keyframes_registered,
    )


def _record_run(recorder: StageRecorder, artifact_dir: Path) -> None:
    try:
        write_manifest(recorder.finish(), artifact_dir)
    except OSError as error:
        logger.warning("could not write the run manifest to %s: %s", artifact_dir, error)
