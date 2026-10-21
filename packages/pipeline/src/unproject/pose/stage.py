"""Stage [3] driver: estimate camera poses for the keyframes."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from unproject.config import ExperimentConfig, PoseConfig, config_hash
from unproject.ingest.stage import ingest_artifact_dir, ingest_identity
from unproject.pose.artifacts import (
    ARTIFACT_FILENAME,
    POINTS_FILENAME,
    PoseArtifact,
    RegistrationSummary,
)
from unproject.pose.backend import PoseBackend, PoseBackendError, PoseEstimate
from unproject.pose.pointcloud import write_ply
from unproject.preprocess.artifacts import PreprocessArtifact
from unproject.preprocess.stage import (
    preprocess_artifact_dir,
    preprocess_hash,
    run_preprocess,
)
from unproject.run.manifest import StageRecorder, write_manifest
from unproject.run.publish import staged_artifact
from unproject.run.reuse import unusable_reason

STAGE_NAME = "pose"

_HASH_PREFIX_LENGTH = 12

logger = logging.getLogger(__name__)


class PoseError(Exception):
    """Raised when the stage cannot produce a complete artifact."""


@dataclass(frozen=True, slots=True)
class PoseResult:
    artifact_dir: Path
    artifact: PoseArtifact
    reused: bool


def pose_hash(preprocess_hash_value: str, config: PoseConfig) -> str:
    """Identity of a pose run: its own config and the keyframes it was given.

    Same reasoning as stage [2]. A different keyframe set means different poses,
    whatever the backend settings say.
    """
    combined = f"{preprocess_hash_value}:{config_hash(config)}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def pose_artifact_dir(artifacts_root: Path, experiment: str, chained_hash: str) -> Path:
    return artifacts_root / experiment / STAGE_NAME / chained_hash[:_HASH_PREFIX_LENGTH]


def resolve_artifact_dir(config: ExperimentConfig, artifacts_root: Path, config_dir: Path) -> Path:
    """Where this config's pose artifact lands, without running anything.

    Stage [5] needs this: it compares two backends, and it cannot load both
    their libraries into one process. Working the location out costs a read of
    the source, since the ingest identity depends on its content, but no stage
    runs. `test_pose_stage` pins it against what `run_pose` actually writes.
    """
    ingest = ingest_artifact_dir(
        artifacts_root, config.name, ingest_identity(config.ingest, config_dir)
    )
    preprocess = preprocess_artifact_dir(
        artifacts_root, config.name, preprocess_hash(ingest.name, config.preprocess)
    )
    return pose_artifact_dir(artifacts_root, config.name, pose_hash(preprocess.name, config.pose))


def run_pose(
    config: ExperimentConfig,
    *,
    config_dir: Path,
    artifacts_root: Path,
    overwrite: bool = False,
) -> PoseResult:
    """Estimate poses for the stage [2] keyframes and write them out.

    Runs the earlier stages first, reusing their artifacts when they exist.
    """
    preprocessed = run_preprocess(
        config, config_dir=config_dir, artifacts_root=artifacts_root, overwrite=overwrite
    )
    upstream_hash = preprocessed.artifact_dir.name

    chained = pose_hash(upstream_hash, config.pose)
    artifact_dir = pose_artifact_dir(artifacts_root, config.name, chained)

    if not overwrite:
        previous = _reusable(artifact_dir)
        if previous is not None:
            logger.info("reusing existing pose artifact at %s", artifact_dir)
            return PoseResult(artifact_dir=artifact_dir, artifact=previous, reused=True)

    recorder = StageRecorder(
        STAGE_NAME, experiment=config.name, config_hash=chained, seed=config.seed
    )
    with staged_artifact(artifact_dir, descriptor=ARTIFACT_FILENAME) as staging:
        # The backend works inside the staging directory too: COLMAP's database
        # and intermediate reconstruction are built there and published with the
        # artifact, never written into an already published one.
        backend = _open_backend(config.pose, preprocessed.artifact, staging, config_dir)
        try:
            with recorder:
                artifact = _estimate(
                    backend,
                    preprocessed.artifact,
                    preprocessed.artifact_dir,
                    upstream_hash,
                    staging,
                )
        finally:
            _record_run(recorder, staging)

        (staging / ARTIFACT_FILENAME).write_text(
            artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    summary = artifact.registration
    logger.info(
        "registered %d of %d keyframes in %s",
        summary.keyframes_registered,
        summary.keyframes_total,
        artifact_dir,
    )
    return PoseResult(artifact_dir=artifact_dir, artifact=artifact, reused=False)


def _open_backend(
    config: PoseConfig,
    preprocessed: PreprocessArtifact,
    artifact_dir: Path,
    config_dir: Path,
) -> PoseBackend:
    """Build the chosen backend, loading only that one's dependencies.

    pycolmap and torch each bundle their own copy of libomp, and loading both
    into one process aborts it on macOS (see `docs/reproduce.md` §6). A run uses
    one backend, so importing them here rather than at module scope is what
    keeps them apart.
    """
    if config.backend == "colmap":
        from unproject.pose.colmap import ColmapBackend

        return ColmapBackend(
            config.colmap, work_dir=artifact_dir, source_kind=preprocessed.source.kind
        )

    from unproject.pose.lingbot import LingbotBackend

    return LingbotBackend(config.lingbot, config_dir=config_dir)


def _estimate(
    backend: PoseBackend,
    preprocessed: PreprocessArtifact,
    preprocess_dir: Path,
    upstream_hash: str,
    artifact_dir: Path,
) -> PoseArtifact:
    keyframes = preprocessed.keyframes
    if not keyframes:
        raise PoseError(f"{preprocess_dir} holds no keyframes to estimate poses for")

    image_paths = [preprocess_dir / record.path for record in keyframes]
    indices = [record.keyframe_index for record in keyframes]

    try:
        estimate = backend.estimate(image_paths, indices)
    except PoseBackendError as error:
        raise PoseError(f"{backend.describe().name} could not estimate poses: {error}") from error

    if not estimate.poses:
        raise PoseError(
            f"{backend.describe().name} registered none of the {len(keyframes)} keyframes"
        )

    written = _write_points(estimate, artifact_dir)
    return PoseArtifact(
        preprocess_hash=upstream_hash,
        backend=backend.describe(),
        cameras=estimate.cameras,
        poses=estimate.poses,
        registration=RegistrationSummary(
            keyframes_total=len(keyframes),
            keyframes_registered=len(estimate.poses),
            points_count=0 if estimate.points is None else len(estimate.points),
            mean_reprojection_error_px=estimate.mean_reprojection_error_px,
        ),
        points_path=None if written is None else written[0],
        points_byte_length=None if written is None else written[1],
    )


def _write_points(estimate: PoseEstimate, artifact_dir: Path) -> tuple[str, int] | None:
    """Write the sparse cloud and return its name and size, or None if empty."""
    if estimate.points is None or estimate.colors is None or len(estimate.points) == 0:
        return None
    path = artifact_dir / POINTS_FILENAME
    write_ply(path, estimate.points, estimate.colors)
    return POINTS_FILENAME, path.stat().st_size


def read_artifact(path: Path) -> PoseArtifact | None:
    """Load a previous artifact, or None if this build cannot use it."""
    try:
        return PoseArtifact.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        logger.info("cannot reuse %s, re-running the stage: %s", path, error)
        return None


def _reusable(artifact_dir: Path) -> PoseArtifact | None:
    """Load a previous artifact, or None if it is unreadable or incomplete."""
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
