"""Stage [1] driver: write a `FrameSource` to disk and describe the result."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2

from unproject.config import ExperimentConfig, IngestConfig, config_hash
from unproject.ingest.artifacts import (
    ARTIFACT_FILENAME,
    FRAMES_DIRNAME,
    FrameRecord,
    IngestArtifact,
    SelectionSummary,
)
from unproject.ingest.geometry import GeometryError, apply_geometry, output_size
from unproject.ingest.image_dir import ImageDirectorySource
from unproject.ingest.source import FrameSource
from unproject.ingest.types import Frame, SourceMetadata
from unproject.ingest.video import VideoFileSource
from unproject.run.manifest import StageRecorder, write_manifest
from unproject.run.publish import staged_artifact
from unproject.run.reuse import unusable_reason

STAGE_NAME = "ingest"

# Enough to stay collision-free over the number of runs this project will do.
_HASH_PREFIX_LENGTH = 12

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Raised when the stage cannot produce a complete artifact."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    artifact_dir: Path
    artifact: IngestArtifact
    # Set when a complete artifact was already on disk and nothing was decoded.
    reused: bool


def ingest_hash(config: IngestConfig, content_sha256: str) -> str:
    """Identity of an ingest run: its own config and the bytes it read.

    The config alone is not an input identity. Two different videos can sit at
    the same path, and a directory of images can have one of them replaced, and
    in both cases the frames on disk would otherwise be reused for footage they
    did not come from.
    """
    combined = f"{config_hash(config)}:{content_sha256}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def ingest_identity(config: IngestConfig, config_dir: Path) -> str:
    """The ingest hash for a config, by looking at the source it names.

    Unlike every later stage this is not a pure function of the config file:
    reading the source is the only way to learn its digest.
    """
    source = _open_source(config, config_dir)
    return ingest_hash(config, source.metadata().content_sha256)


def ingest_artifact_dir(artifacts_root: Path, experiment: str, ingest_hash: str) -> Path:
    """Where a given ingest run lands.

    Keyed by the stage's own identity rather than the whole experiment's, so
    changing a training parameter does not invalidate frames that are still good.
    """
    return artifacts_root / experiment / STAGE_NAME / ingest_hash[:_HASH_PREFIX_LENGTH]


def run_ingest(
    config: ExperimentConfig,
    *,
    config_dir: Path,
    artifacts_root: Path,
    overwrite: bool = False,
) -> IngestResult:
    """Decode the configured slice of the source into a frames directory.

    Relative paths in the config resolve against `config_dir`, the directory the
    config file itself lives in.
    """
    source = _open_source(config.ingest, config_dir)
    metadata = source.metadata()

    stage_hash = ingest_hash(config.ingest, metadata.content_sha256)
    artifact_dir = ingest_artifact_dir(artifacts_root, config.name, stage_hash)

    if not overwrite:
        previous = _reusable(artifact_dir)
        if previous is not None:
            logger.info("reusing existing ingest artifact at %s", artifact_dir)
            return IngestResult(artifact_dir=artifact_dir, artifact=previous, reused=True)

    recorder = StageRecorder(
        STAGE_NAME,
        experiment=config.name,
        config_hash=stage_hash,
        seed=config.seed,
    )
    # Built beside the target and moved in at the end, so a run that dies
    # halfway cannot leave frames that a descriptor already vouches for.
    with staged_artifact(artifact_dir, descriptor=ARTIFACT_FILENAME) as staging:
        try:
            with recorder:
                artifact = _extract(source, metadata, config.ingest, staging)
        finally:
            # Runs on the failure path too: a stage that died should still record
            # how long it ran and what stopped it.
            _record_run(recorder, staging)

        # Written last: while it is missing the directory is visibly incomplete,
        # so an interrupted run cannot be mistaken for a finished one.
        (staging / ARTIFACT_FILENAME).write_text(
            artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    logger.info("wrote %d frames to %s", artifact.selection.frames_written, artifact_dir)
    return IngestResult(artifact_dir=artifact_dir, artifact=artifact, reused=False)


def _reusable(artifact_dir: Path) -> IngestArtifact | None:
    """Load a previous artifact, or None if this build cannot use it.

    An artifact is always reproducible from its config, so one that is
    unreadable, written by an older schema, or no longer backed by the frames it
    describes is a reason to run the stage again rather than to fail.
    """
    path = artifact_dir / ARTIFACT_FILENAME
    if not path.is_file():
        return None
    try:
        artifact = IngestArtifact.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        logger.info("cannot reuse %s, re-running the stage: %s", path, error)
        return None

    reason = unusable_reason(artifact_dir, artifact.expected_files())
    if reason is not None:
        logger.info("cannot reuse %s, re-running the stage: %s", artifact_dir, reason)
        return None
    return artifact


def open_source(config: IngestConfig, config_dir: Path) -> FrameSource:
    """The source an ingest config names, opened but not read.

    For a caller that wants the input's identity or size without running the
    stage: the benchmark preflight checks the bytes on a machine against the
    ones the campaign expects this way.
    """
    return _open_source(config, config_dir)


def _open_source(config: IngestConfig, config_dir: Path) -> FrameSource:
    """Pick an implementation from what the path actually is.

    A directory of images and a video file are unambiguous on disk, so the
    config does not repeat the distinction and cannot contradict it.
    """
    path = _resolve(config.source_path, config_dir)
    uri = str(config.source_path)

    if path.is_dir():
        if config.image_fps is None:
            raise IngestError(
                f"{path} is a directory of images; set image_fps in the config, "
                "since images carry no frame rate of their own"
            )
        return ImageDirectorySource(path, fps=config.image_fps, uri=uri)
    return VideoFileSource(path, uri=uri)


def _record_run(recorder: StageRecorder, artifact_dir: Path) -> None:
    """Write the manifest without letting an I/O error hide a stage error.

    This runs on the failure path, where the disk is one of the things that may
    have gone wrong; the original exception is the one worth propagating.
    """
    try:
        write_manifest(recorder.finish(), artifact_dir)
    except OSError as error:
        logger.warning("could not write the run manifest to %s: %s", artifact_dir, error)


def _extract(
    source: FrameSource,
    metadata: SourceMetadata,
    config: IngestConfig,
    artifact_dir: Path,
) -> IngestArtifact:
    frames_dir = artifact_dir / FRAMES_DIRNAME
    frames_dir.mkdir(parents=True)

    try:
        frame_width, frame_height = output_size(metadata.width, metadata.height, config.geometry)
    except GeometryError as error:
        raise IngestError(f"{metadata.uri}: {error}") from error

    counter = _DecodeCounter()
    records: list[FrameRecord] = []
    truncated = False

    for frame in _select(counter.wrap(source.frames()), config):
        if config.max_frames is not None and len(records) >= config.max_frames:
            truncated = True
            break
        if config.geometry is not None:
            try:
                frame = Frame(
                    source_index=frame.source_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    image=apply_geometry(frame.image, config.geometry),
                )
            except GeometryError as error:
                raise IngestError(
                    f"frame {frame.source_index} of {metadata.uri}: {error}"
                ) from error
        relative = f"{FRAMES_DIRNAME}/{frame.source_index:06d}.png"
        written = _write_png(frame, artifact_dir / relative)
        records.append(
            FrameRecord(
                source_index=frame.source_index,
                timestamp_seconds=frame.timestamp_seconds,
                path=relative,
                byte_length=written,
            )
        )

    if not records:
        raise IngestError(
            f"no frames selected from {metadata.uri}; check the time window and stride"
        )

    return IngestArtifact(
        source=metadata,
        selection=SelectionSummary(
            frame_stride=config.frame_stride,
            start_time_seconds=config.start_time_seconds,
            end_time_seconds=config.end_time_seconds,
            max_frames=config.max_frames,
            frames_decoded=counter.count,
            frames_written=len(records),
            truncated=truncated,
        ),
        geometry=config.geometry,
        frame_width=frame_width,
        frame_height=frame_height,
        frames=records,
    )


class _DecodeCounter:
    """Counts frames pulled off the source.

    `max_frames` and the time window stop the loop early, so the number of
    frames written says nothing about how much of the video was read.
    """

    def __init__(self) -> None:
        self.count = 0

    def wrap(self, frames: Iterator[Frame]) -> Iterator[Frame]:
        for frame in frames:
            self.count += 1
            yield frame


def _select(frames: Iterator[Frame], config: IngestConfig) -> Iterator[Frame]:
    """Apply the time window, then decimate by stride.

    The stride counts from the first frame inside the window, so moving the
    window start does not reshuffle which frames get picked within it.
    """
    start = config.start_time_seconds
    end = config.end_time_seconds
    kept_in_window = 0

    for frame in frames:
        if start is not None and frame.timestamp_seconds < start:
            continue
        if end is not None and frame.timestamp_seconds >= end:
            # Frames arrive in decode order, so nothing later can qualify.
            break
        if kept_in_window % config.frame_stride == 0:
            yield frame
        kept_in_window += 1


def _write_png(frame: Frame, path: Path) -> int:
    """Write one frame and return the size the artifact will promise for it."""
    if not cv2.imwrite(str(path), frame.image):
        raise IngestError(f"failed to write frame {frame.source_index} to {path}")
    return path.stat().st_size


def _resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()
