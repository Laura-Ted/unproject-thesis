"""Running one dataset sequence through stage [B].

ADR-0004 §9 WP2 asks for two things that pull against each other: reach every
sequence through the *existing* estimator path rather than a second copy of it,
and keep the sequences apart so no identity survives from the end of one to the
start of the next.

Both fall out of one decision. A sequence is turned into an ordinary experiment
config whose ingest source is that sequence's frame directory, and stage [B] is
invoked on it exactly as `unproject bodies` would. The estimator, and with it
the tracker, is built inside `run_bodies`, so a call per sequence is a fresh
tracker per sequence with nothing to reset and nothing to leak.

The per-sequence name goes into the artifact path, so each sequence's bodies
land in their own directory and are reused independently.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from unproject.bodies.artifacts import BodiesArtifact
from unproject.bodies.stage import run_bodies
from unproject.config import ExperimentConfig, KnownIntrinsics
from unproject.ground_truth.config import CameraKind, FrameTransform
from unproject.ground_truth.dataset import DatasetRecord, GroundTruthError
from unproject.ground_truth.mupots import SequenceIntrinsics
from unproject.ingest.artifacts import ARTIFACT_FILENAME as INGEST_ARTIFACT_FILENAME
from unproject.ingest.stage import ingest_artifact_dir

logger = logging.getLogger(__name__)

# Frame directories carry no timing, so ingest demands a rate. Anything the
# estimator does is per-frame, so this only decides what a timestamp means -
# which is why a wrong value here is invisible until a velocity is taken.
_UNSET_FPS = "the experiment config for a frame-directory dataset must set ingest.image_fps"

# Appended to the per-sequence experiment name, and so to the artifact
# directory, when the estimator is told the annotation-derived camera, and
# when the frames were letterboxed. Both are different inputs and must not
# share a directory with the plain run.
ANNOTATION_DERIVED_SUFFIX = "-annotation-derived-camera"
LETTERBOX_SUFFIX = "-letterbox-square"


@dataclass(frozen=True, slots=True)
class SequenceRun:
    """One sequence, estimated and paired with what it should have been."""

    sequence_id: str
    # The per-sequence experiment name, which is also the artifact directory it
    # writes under. Carried so nothing has to re-derive it and guess wrong.
    experiment_name: str
    artifact_dir: Path
    artifact: BodiesArtifact
    records: tuple[DatasetRecord, ...]
    intrinsics: SequenceIntrinsics
    reused: bool

    @property
    def annotated_people(self) -> int:
        return sum(len(record.people) for record in self.records)


def sequence_experiment(
    base: ExperimentConfig,
    *,
    sequence_id: str,
    frames_dir: Path,
    camera: KnownIntrinsics | None = None,
    transform: FrameTransform = "none",
) -> ExperimentConfig:
    """The base experiment, pointed at one sequence's frames.

    Only the source and the name change - and the camera, when the evaluation
    asks for the annotation-derived one. Every other estimator setting -
    checkpoint digest, backbone revision, thresholds - is inherited unaltered,
    so the sequences are comparable with each other and with the project's own
    runs.

    The name carries the camera choice so the two conditions never share an
    artifact directory: the bodies hash already differs, and the name makes it
    visible on disk as well. The suffix names where the camera came from, not
    a "released" one MuPoTS does not publish.
    """
    if base.ingest.image_fps is None:
        raise GroundTruthError(f"{_UNSET_FPS}; {base.name} does not")
    if base.bodies.intrinsics is not None:
        raise GroundTruthError(
            f"{base.name} sets bodies.intrinsics itself. A ground-truth run supplies "
            "the camera per sequence; a fixed one in the experiment would be applied "
            "to every sequence regardless of which camera shot it."
        )

    suffix = "" if camera is None else ANNOTATION_DERIVED_SUFFIX
    if transform == "letterbox_square":
        suffix += LETTERBOX_SUFFIX
    return base.model_copy(
        update={
            "name": f"{base.name}{suffix}-{sequence_id}",
            "ingest": base.ingest.model_copy(update={"source_path": frames_dir}),
            "bodies": base.bodies.model_copy(update={"intrinsics": camera}),
        }
    )


def annotation_derived_camera(
    intrinsics: SequenceIntrinsics, *, width: int, height: int
) -> KnownIntrinsics:
    """The camera the annotations imply, in the form the estimator accepts.

    Recovered by least squares from the released 2D and 3D, per sequence. It is
    not a published calibration - MuPoTS ships none - and it is not "the true
    camera": it is the pinhole that best explains the annotations, with the
    residual recorded beside it.
    """
    return KnownIntrinsics(
        width=width,
        height=height,
        fx=intrinsics.camera.fx,
        fy=intrinsics.camera.fy,
        cx=intrinsics.camera.cx,
        cy=intrinsics.camera.cy,
    )


def run_sequence(
    base: ExperimentConfig,
    *,
    sequence_id: str,
    frames_dir: Path,
    records: list[DatasetRecord],
    intrinsics: SequenceIntrinsics,
    config_dir: Path,
    artifacts_root: Path,
    overwrite: bool = False,
    camera: CameraKind = "model_estimated",
    transform: FrameTransform = "none",
) -> SequenceRun:
    """Estimate one sequence and return it beside its ground truth.

    Refuses when the frame count and the annotation count disagree. That
    mismatch is the one failure a later metric cannot see: every frame would
    still be compared against *some* annotation, just not its own, and the
    result would be a plausible-looking error rather than an exception.
    """
    known = None
    if camera == "annotation_derived":
        if not records:
            raise GroundTruthError(f"{sequence_id} has no frames to take an image size from")
        known = annotation_derived_camera(
            intrinsics, width=records[0].image_width, height=records[0].image_height
        )
    experiment = sequence_experiment(
        base, sequence_id=sequence_id, frames_dir=frames_dir, camera=known, transform=transform
    )
    result = run_bodies(
        experiment,
        config_dir=config_dir,
        artifacts_root=artifacts_root,
        overwrite=overwrite,
    )

    estimated = result.artifact.summary.frames_total
    if estimated != len(records):
        raise GroundTruthError(
            f"{sequence_id}: stage [B] saw {estimated} frame(s) but the annotations "
            f"describe {len(records)}. Frame N would be scored against annotation N "
            "for a different moment. Check ingest.frame_stride and max_frames are "
            "unset for ground-truth runs."
        )

    return SequenceRun(
        sequence_id=sequence_id,
        experiment_name=experiment.name,
        artifact_dir=result.artifact_dir,
        artifact=result.artifact,
        records=tuple(records),
        intrinsics=intrinsics,
        reused=result.reused,
    )


def release_frames(run: SequenceRun, *, artifacts_root: Path) -> int:
    """Delete the decoded frames of a sequence whose bodies are already written.

    Stage [1] re-encodes every source frame as lossless PNG, which for a
    directory of JPEGs is a copy roughly five times the size of files that are
    still on disk. Held for all twenty MuPoTS sequences that is 27 GB of
    intermediate; released after each one it is 2.5 GB.

    Nothing measured is lost. The bodies artifact - the actual result - is
    untouched, and the frames regenerate from the dataset by re-running the
    stage. Returns the bytes freed.

    Deliberately narrow: it resolves exactly the one directory the artifact
    names as its own input, refuses anything that is not a complete ingest
    artifact, and never walks a tree it was not given.
    """
    directory = ingest_artifact_dir(artifacts_root, run.experiment_name, run.artifact.ingest_hash)
    if not (directory / INGEST_ARTIFACT_FILENAME).is_file():
        # Already released, or never there: nothing to free, and not an error.
        return 0

    freed = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    shutil.rmtree(directory)
    logger.info("released %.1f GB of decoded frames from %s", freed / 1e9, directory)
    return freed
