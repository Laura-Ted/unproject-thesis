"""Stage [2] driver: reduce ingested frames to keyframes on disk."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from unproject.config import ExperimentConfig, PreprocessConfig, config_hash
from unproject.ingest.artifacts import IngestArtifact
from unproject.ingest.stage import run_ingest
from unproject.preprocess.artifacts import (
    ARTIFACT_FILENAME,
    KEYFRAMES_DIRNAME,
    KeyframeRecord,
    PreprocessArtifact,
    ResizeSummary,
    SelectionSummary,
)
from unproject.preprocess.metrics import GrayArray, to_working_gray, working_scale
from unproject.preprocess.selection import select_keyframes
from unproject.run.manifest import StageRecorder, write_manifest
from unproject.run.publish import staged_artifact
from unproject.run.reuse import unusable_reason

STAGE_NAME = "preprocess"

_HASH_PREFIX_LENGTH = 12

logger = logging.getLogger(__name__)


class PreprocessError(Exception):
    """Raised when the stage cannot produce a complete artifact."""


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    artifact_dir: Path
    artifact: PreprocessArtifact
    reused: bool


def preprocess_hash(ingest_hash: str, config: PreprocessConfig) -> str:
    """Identity of a preprocess run: its own config and its input.

    Without the ingest hash, a keyframe set computed from one set of frames
    would be reused for another. Changing the ingest stride changes the frames,
    so it has to change this too.
    """
    combined = f"{ingest_hash}:{config_hash(config)}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def preprocess_artifact_dir(artifacts_root: Path, experiment: str, chained_hash: str) -> Path:
    return artifacts_root / experiment / STAGE_NAME / chained_hash[:_HASH_PREFIX_LENGTH]


def run_preprocess(
    config: ExperimentConfig,
    *,
    config_dir: Path,
    artifacts_root: Path,
    overwrite: bool = False,
) -> PreprocessResult:
    """Select keyframes from the ingested frames and write them out.

    Runs stage [1] first, which reuses its artifact when one is already there.
    """
    ingested = run_ingest(
        config, config_dir=config_dir, artifacts_root=artifacts_root, overwrite=overwrite
    )
    ingest_hash = ingested.artifact_dir.name

    chained = preprocess_hash(ingest_hash, config.preprocess)
    artifact_dir = preprocess_artifact_dir(artifacts_root, config.name, chained)

    if not overwrite:
        previous = _reusable(artifact_dir)
        if previous is not None:
            logger.info("reusing existing preprocess artifact at %s", artifact_dir)
            return PreprocessResult(artifact_dir=artifact_dir, artifact=previous, reused=True)

    recorder = StageRecorder(
        STAGE_NAME, experiment=config.name, config_hash=chained, seed=config.seed
    )
    with staged_artifact(artifact_dir, descriptor=ARTIFACT_FILENAME) as staging:
        try:
            with recorder:
                artifact = _select(
                    ingested.artifact,
                    ingested.artifact_dir,
                    config.preprocess,
                    ingest_hash,
                    staging,
                )
        finally:
            _record_run(recorder, staging)

        # Written last, so an interrupted run leaves a visibly incomplete directory.
        (staging / ARTIFACT_FILENAME).write_text(
            artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    logger.info("kept %d keyframes in %s", artifact.selection.keyframes_kept, artifact_dir)
    return PreprocessResult(artifact_dir=artifact_dir, artifact=artifact, reused=False)


def _select(
    ingested: IngestArtifact,
    ingest_dir: Path,
    config: PreprocessConfig,
    ingest_hash: str,
    artifact_dir: Path,
) -> PreprocessArtifact:
    frames = [_read_frame(ingest_dir / record.path) for record in ingested.frames]
    if not frames:
        raise PreprocessError(f"{ingest_dir} holds no frames to select from")

    height, width = frames[0].shape[0], frames[0].shape[1]
    output_scale = working_scale(
        int(width), int(height), config.max_long_edge or max(width, height)
    )
    metric_scale = working_scale(int(width), int(height), config.metric_long_edge)
    # Thresholds are stated in output pixels but measured on the metric image,
    # so a displacement needs converting before it can be compared.
    disparity_scale = output_scale / metric_scale

    grays: list[GrayArray] = [to_working_gray(frame, config.metric_long_edge) for frame in frames]
    keyframes = select_keyframes(grays, config, disparity_scale=disparity_scale)
    # Triangulation needs two views at minimum, so ending with one means the
    # configuration is wrong rather than the scene being thin.
    if len(keyframes) < 2:
        raise PreprocessError(
            f"only {len(keyframes)} keyframe(s) selected from {ingest_dir}, which cannot "
            f"be reconstructed; min_disparity_px={config.min_disparity_px} is likely too "
            "high for how far the camera moved"
        )

    keyframes_dir = artifact_dir / KEYFRAMES_DIRNAME
    keyframes_dir.mkdir(parents=True)

    held_out = _held_out_positions(len(keyframes), config.held_out_every)
    training = len(keyframes) - len(held_out)
    if training < 2:
        raise PreprocessError(
            f"held_out_every={config.held_out_every} leaves {training} keyframe(s) to train "
            f"on out of {len(keyframes)}; the scene needs more keyframes or a wider split"
        )

    records: list[KeyframeRecord] = []
    for position, keyframe in enumerate(keyframes):
        frame_record = ingested.frames[keyframe.index]
        relative = f"{KEYFRAMES_DIRNAME}/{position:06d}.png"
        written_bytes = _write_png(
            _resized(frames[keyframe.index], output_scale), artifact_dir / relative
        )
        records.append(
            KeyframeRecord(
                keyframe_index=position,
                source_index=frame_record.source_index,
                timestamp_seconds=frame_record.timestamp_seconds,
                path=relative,
                byte_length=written_bytes,
                sharpness=keyframe.sharpness,
                displacement_px=keyframe.displacement_px,
                tracked_ratio=keyframe.tracked_ratio,
                forced=keyframe.forced,
                held_out=position in held_out,
            )
        )

    written = _read_frame(artifact_dir / records[0].path)
    return PreprocessArtifact(
        ingest_hash=ingest_hash,
        source=ingested.source,
        resize=ResizeSummary(
            scale=output_scale,
            width=int(written.shape[1]),
            height=int(written.shape[0]),
        ),
        selection=SelectionSummary(
            frames_considered=len(frames),
            keyframes_kept=len(records),
            forced_keyframes=sum(1 for record in records if record.forced),
            held_out_every=config.held_out_every,
            held_out_count=len(held_out),
            min_disparity_px=config.min_disparity_px,
            min_tracked_ratio=config.min_tracked_ratio,
            blur_percentile=config.blur_percentile,
            blur_floor=min(record.sharpness for record in records),
            metric_long_edge=config.metric_long_edge,
        ),
        keyframes=records,
    )


def _held_out_positions(count: int, every: int | None) -> frozenset[int]:
    """Which keyframes are withheld from training.

    Every Nth by position, following the convention the 3DGS papers evaluate
    against. Deterministic, so no seed is involved and two runs of the same
    config cannot disagree.

    Caveat for the write-up: in densely sampled footage a withheld view can sit
    almost on top of a training one, which makes the test easier than it looks.
    """
    if every is None:
        return frozenset()
    return frozenset(position for position in range(count) if position % every == 0)


def _reusable(artifact_dir: Path) -> PreprocessArtifact | None:
    """Load a previous artifact, or None if this build cannot use it.

    Unreadable, from an older schema, or no longer backed by the keyframes it
    describes all mean the same thing: run the stage again.
    """
    path = artifact_dir / ARTIFACT_FILENAME
    if not path.is_file():
        return None
    try:
        artifact = PreprocessArtifact.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        logger.info("cannot reuse %s, re-running the stage: %s", path, error)
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


def _read_frame(path: Path) -> NDArray[np.uint8]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessError(f"cannot decode {path}")
    return cast(NDArray[np.uint8], image)


def _resized(image: NDArray[np.uint8], scale: float) -> NDArray[np.uint8]:
    if scale == 1.0:
        return image
    # INTER_AREA averages over the source region; INTER_LINEAR aliases when
    # shrinking by more than a factor of two.
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cast(NDArray[np.uint8], resized)


def _write_png(image: NDArray[np.uint8], path: Path) -> int:
    """Write one keyframe and return the size the artifact will promise for it."""
    if not cv2.imwrite(str(path), image):
        raise PreprocessError(f"failed to write {path}")
    return path.stat().st_size
