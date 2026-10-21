"""A controlled transformation of a sequence's frames: letterbox to a square.

ADR-0005 WP5 asks that the square-frame effect be tested with the *same
content* transformed in a controlled way, not by comparing TS1-TS5 (square,
indoor, other actors) with TS6-TS20 (wide, outdoor), where scene, subjects and
aspect ratio all change at once.

So: take a wide sequence, add black rows above and below until it is square,
and score it against the same annotations moved down by the padding. The
people, the camera and the 3D annotations are untouched - camera-space 3D does
not know about pixels - and the only thing that changes on the way to the
model is the shape of the tensor. If the depth bias moves, the aspect-ratio
path did it; if it does not, the square sequences' extra offset is something
about those sequences.

The padded frames are written as lossless PNG under `artifacts/`, never
beside the dataset, and a run that asked for `--release-frames` deletes them
per sequence once its bodies exist. The ingest stage hashes their content, so
a transformed sequence is a different input with a different identity.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from unproject.ground_truth.dataset import Camera, DatasetRecord, GroundTruthError
from unproject.ground_truth.mupots import SequenceIntrinsics

logger = logging.getLogger(__name__)

DERIVED_DIRECTORY = "_derived"
LETTERBOX_SQUARE = "letterbox-square"


def letterbox_padding(width: int, height: int) -> tuple[int, int]:
    """Rows to add above and below a frame to make it square: (top, bottom).

    Symmetric where the difference is even; the odd row goes below, and the
    principal point moves by the top pad, so a reader can recompute it.
    """
    if width < height:
        raise GroundTruthError(
            f"a {width}x{height} frame is taller than it is wide; letterboxing to a square "
            "adds rows and would not make it square"
        )
    extra = width - height
    return extra // 2, extra - extra // 2


def letterbox_record(record: DatasetRecord, top: int) -> DatasetRecord:
    """The same annotations, on a frame with `top` black rows above it."""
    shift = np.array([0.0, float(top)])
    people = tuple(
        replace(person, joints_2d=None if person.joints_2d is None else person.joints_2d + shift)
        for person in record.people
    )
    return replace(
        record,
        image_width=record.image_width,
        image_height=record.image_width,
        people=people,
        camera=None if record.camera is None else _shift(record.camera, top),
    )


def letterbox_intrinsics(intrinsics: SequenceIntrinsics, top: int) -> SequenceIntrinsics:
    return replace(intrinsics, camera=_shift(intrinsics.camera, top))


def _shift(camera: Camera, top: int) -> Camera:
    return Camera(fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy + top)


def derived_frames_dir(artifacts_root: Path, dataset: str, sequence_id: str) -> Path:
    return artifacts_root / DERIVED_DIRECTORY / dataset / LETTERBOX_SQUARE / sequence_id


def write_letterboxed_frames(
    records: list[DatasetRecord], target: Path, *, top: int, bottom: int
) -> Path:
    """Every frame of the sequence, padded, as PNG named like the original.

    Reused when complete: a marker file records how many frames were written
    and with what padding, and a directory that matches is not rewritten.
    """
    import cv2

    marker = target / "letterbox.txt"
    stamp = f"frames={len(records)} top={top} bottom={bottom}\n"
    if marker.is_file() and marker.read_text("utf-8") == stamp:
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for record in records:
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise GroundTruthError(f"cannot read {record.image_path}")
        if (image.shape[1], image.shape[0]) != (record.image_width, record.image_height):
            raise GroundTruthError(
                f"{record.image_path} is {image.shape[1]}x{image.shape[0]}, but the release "
                f"says {record.image_width}x{record.image_height}"
            )
        padded = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if padded.shape[0] != padded.shape[1]:
            raise GroundTruthError("padding did not produce a square frame; this is a bug")
        # PNG, lossless: the content is the release's own pixels plus black.
        if not cv2.imwrite(str(target / (record.image_path.stem + ".png")), padded):
            raise GroundTruthError(f"cannot write {target / record.image_path.stem}.png")
    marker.write_text(stamp, encoding="utf-8")
    return target


def letterbox_sequence(
    records: list[DatasetRecord],
    intrinsics: SequenceIntrinsics,
    *,
    artifacts_root: Path,
    dataset: str,
    sequence_id: str,
) -> tuple[Path, list[DatasetRecord], SequenceIntrinsics, int]:
    """Pad one sequence to a square. Returns the frames directory, the moved
    annotations, the moved intrinsics, and the top padding.

    A sequence that is already square is returned unchanged with padding 0
    and no derived frames: there is nothing to transform, and writing a copy
    would only claim there was.
    """
    if not records:
        raise GroundTruthError(f"{sequence_id} has no frames")
    width, height = records[0].image_width, records[0].image_height
    if width == height:
        return records[0].image_path.parent, records, intrinsics, 0
    top, bottom = letterbox_padding(width, height)
    target = write_letterboxed_frames(
        records,
        derived_frames_dir(artifacts_root, dataset, sequence_id),
        top=top,
        bottom=bottom,
    )
    moved = [
        replace(
            letterbox_record(record, top),
            image_path=target / (record.image_path.stem + ".png"),
        )
        for record in records
    ]
    return target, moved, letterbox_intrinsics(intrinsics, top), top


def release_derived_frames(artifacts_root: Path, dataset: str, sequence_id: str) -> int:
    """Delete one sequence's padded frames. Returns the bytes freed."""
    directory = derived_frames_dir(artifacts_root, dataset, sequence_id)
    if not directory.is_dir():
        return 0
    freed = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    shutil.rmtree(directory)
    logger.info("released %.1f GB of letterboxed frames from %s", freed / 1e9, directory)
    return freed


__all__ = [
    "DERIVED_DIRECTORY",
    "LETTERBOX_SQUARE",
    "derived_frames_dir",
    "letterbox_intrinsics",
    "letterbox_padding",
    "letterbox_record",
    "letterbox_sequence",
    "release_derived_frames",
    "write_letterboxed_frames",
]
