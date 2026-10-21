"""Reading MuPoTS-3D into the canonical contract.

The release is 20 sequences of numbered JPEGs with one `annot.mat` each, MATLAB
format 5.0. No pickle, so `scipy.io.loadmat` reads it safely; ADR-0004 §5.7's
warning about executing dataset-provided code does not bite here.

Everything this module asserts about the release was read from the files or from
the official evaluator, and is recorded in
`docs/notes/2026-09-10-mupots-acquired.md`. Three facts matter most:

- **Units are millimetres.** Converted to metres here, once, at the boundary.
- **`annot3` is subject scale, `univ_annot3` is a canonical skeleton.** They
  share the root exactly - the pelvis is bit-identical between them across all
  20,760 annotated people - and differ on the limbs by a median 89 mm. Absolute
  work uses `annot3`; the official root-relative PCK uses `univ_annot3`.
- **No intrinsics are released.** They are recovered here by least squares from
  the released 3D and 2D, which fits to under a pixel on most subjects. See
  `recover_intrinsics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from unproject.ground_truth.dataset import (
    CANONICAL_CONVENTION,
    Camera,
    DatasetRecord,
    FloatArray,
    GroundTruthError,
    GroundTruthPerson,
    validate_record,
)

DATASET_NAME = "mupots-3d"

ANNOTATION_FILENAME = "annot.mat"

# The 17-joint "relevant" set, in the order the official evaluator uses. Read
# from mpii_get_joints.m; the pelvis at index 14 is what the evaluator subtracts
# for root alignment.
JOINT_NAMES: tuple[str, ...] = (
    "head_top",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "pelvis",
    "spine",
    "head",
)

PELVIS_INDEX = JOINT_NAMES.index("pelvis")

SEQUENCE_COUNT = 20

# From mpii_get_multiperson_test_seq_info.m. It returns [rows cols], so its
# [1080 1920] is 1920 wide by 1080 tall; these are stored (width, height) to
# match what every caller wants and to stop the swap happening twice.
_SQUARE_SEQUENCES = 5
_SQUARE_SIZE = (2048, 2048)
_WIDE_SIZE = (1920, 1080)

_MM_PER_METRE = 1000.0


@dataclass(frozen=True, slots=True)
class SequenceIntrinsics:
    """Intrinsics recovered from a sequence, with the evidence for them."""

    camera: Camera
    # Root-mean-square reprojection residual over every annotated joint, in
    # pixels. Sub-pixel on every sequence (docs/notes/2026-09-11-mupots-full-run.md).
    residual_px: float
    samples: int


def recover_intrinsics(joints_3d: FloatArray, joints_2d: FloatArray) -> SequenceIntrinsics:
    """Fit a pinhole to the released 3D and 2D.

    MuPoTS publishes no calibration, so the camera has to be recovered. With
    `u = fx·X/Z + cx` the problem is linear in `(fx, cx)`, and separately in
    `(fy, cy)`, so this is two least-squares fits and not an optimisation.

    The residual is returned rather than discarded: it is the check that
    `annot3` really is camera-relative in the frame this project assumes. A
    wrong axis or a world-space annotation does not fit a pinhole to under a
    pixel by luck.

    `joints_3d` is (N, 3) in metres, `joints_2d` is (N, 2) in pixels.
    """
    if joints_3d.shape[0] != joints_2d.shape[0] or joints_3d.shape[0] < 2:
        raise GroundTruthError(
            f"cannot recover intrinsics from {joints_3d.shape[0]} 3D and "
            f"{joints_2d.shape[0]} 2D point(s); they must correspond and there "
            "must be at least two"
        )
    depth = joints_3d[:, 2]
    if not np.all(depth > 0):
        raise GroundTruthError(
            "some annotated joints sit at or behind the camera plane, so they "
            "cannot be projected. The 3D annotations are not camera-relative "
            "with +Z forward, or a unit conversion is wrong."
        )

    horizontal = np.stack([joints_3d[:, 0] / depth, np.ones_like(depth)], axis=1)
    vertical = np.stack([joints_3d[:, 1] / depth, np.ones_like(depth)], axis=1)
    (fx, cx), *_ = np.linalg.lstsq(horizontal, joints_2d[:, 0], rcond=None)
    (fy, cy), *_ = np.linalg.lstsq(vertical, joints_2d[:, 1], rcond=None)

    residual = float(
        np.sqrt(
            np.mean(
                (horizontal @ [fx, cx] - joints_2d[:, 0]) ** 2
                + (vertical @ [fy, cy] - joints_2d[:, 1]) ** 2
            )
        )
    )
    if fx <= 0 or fy <= 0:
        raise GroundTruthError(
            f"the recovered focal length is not positive (fx={fx:.1f}, fy={fy:.1f}). "
            "That means the 2D and 3D annotations disagree about which way an "
            "axis points."
        )
    return SequenceIntrinsics(
        camera=Camera(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy)),
        residual_px=residual,
        samples=int(joints_3d.shape[0]),
    )


def image_size(sequence_id: str) -> tuple[int, int]:
    """Width and height, from the official sequence table rather than the JPEG.

    Two sizes, not one: the first five sequences are square and the rest are
    landscape. Reading it from the metadata keeps a corrupt frame from silently
    changing the geometry a metric depends on.
    """
    index = _sequence_index(sequence_id)
    return _SQUARE_SIZE if index <= _SQUARE_SEQUENCES else _WIDE_SIZE


class MupotsDataset:
    """The 20 official test sequences, read on demand."""

    def __init__(self, root: Path, *, release_identity: str, universal: bool = False) -> None:
        """`root` holds the `TS1`…`TS20` directories.

        `universal` selects `univ_annot3` over `annot3`. It exists for the
        official root-relative protocol; anything measuring absolute placement
        must leave it false, because the universal skeleton is rescaled.
        """
        self._root = root
        self._release_identity = release_identity
        self._universal = universal

    @property
    def name(self) -> str:
        return DATASET_NAME

    @property
    def release_identity(self) -> str:
        return self._release_identity

    @cached_property
    def _present(self) -> list[str]:
        found = [
            f"TS{index}"
            for index in range(1, SEQUENCE_COUNT + 1)
            if (self._root / f"TS{index}" / ANNOTATION_FILENAME).is_file()
        ]
        if not found:
            raise GroundTruthError(
                f"no MuPoTS sequences under {self._root}. Expected TS1…TS20, each with "
                f"an {ANNOTATION_FILENAME}. Check the archive was extracted and that "
                "this is the MultiPersonTestSet directory rather than its parent."
            )
        return found

    def sequences(self) -> list[str]:
        """Every sequence actually on disk, in numeric order.

        Missing sequences are reported by the caller rather than silently
        skipped: an evaluation over 12 of 20 is not the benchmark.
        """
        return list(self._present)

    def records(self, sequence_id: str) -> list[DatasetRecord]:
        """Every frame of one sequence, validated, in frame order."""
        path = self._root / sequence_id / ANNOTATION_FILENAME
        annotations = _load_annotations(path)
        width, height = image_size(sequence_id)
        field = "univ_annot3" if self._universal else "annot3"

        records: list[DatasetRecord] = []
        for frame_index in range(annotations.shape[0]):
            people = tuple(
                person
                for slot in range(annotations.shape[1])
                if (person := _read_person(annotations[frame_index, slot], field, slot)) is not None
            )
            records.append(
                validate_record(
                    DatasetRecord(
                        dataset_name=DATASET_NAME,
                        release_identity=self._release_identity,
                        sample_id=f"{sequence_id}/{frame_index:06d}",
                        sequence_id=sequence_id,
                        frame_index=frame_index,
                        image_path=self._root / sequence_id / f"img_{frame_index:06d}.jpg",
                        image_width=width,
                        image_height=height,
                        people=people,
                        convention=CANONICAL_CONVENTION,
                    ),
                    source=path,
                )
            )
        return records

    def intrinsics(self, sequence_id: str) -> SequenceIntrinsics:
        """Recover the camera for one sequence from all its annotated joints."""
        records = self.records(sequence_id)
        three = [person.joints_3d for record in records for person in record.people]
        two = [person.joints_2d for record in records for person in record.people]
        if not three or any(item is None for item in two):
            raise GroundTruthError(f"{sequence_id} has no annotated joints to calibrate from")
        return recover_intrinsics(
            np.concatenate(three), np.concatenate([item for item in two if item is not None])
        )


def _sequence_index(sequence_id: str) -> int:
    if not sequence_id.startswith("TS") or not sequence_id[2:].isdigit():
        raise GroundTruthError(
            f"{sequence_id!r} is not a MuPoTS sequence identifier; they are TS1 to TS20"
        )
    index = int(sequence_id[2:])
    if not 1 <= index <= SEQUENCE_COUNT:
        raise GroundTruthError(f"{sequence_id} is outside TS1…TS{SEQUENCE_COUNT}")
    return index


def _load_annotations(path: Path) -> Any:  # noqa: ANN401 - a MATLAB object array
    """Read one `annot.mat`, checking it is shaped the way the release is."""
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise GroundTruthError(
            "reading MuPoTS needs the ground-truth extra: uv sync --extra ground-truth"
        ) from error

    try:
        contents = loadmat(path)
    except (OSError, ValueError) as error:
        raise GroundTruthError(f"cannot read {path}: {error}") from error

    if "annotations" not in contents:
        raise GroundTruthError(
            f"{path} has no 'annotations' array; its keys are "
            f"{sorted(key for key in contents if not key.startswith('__'))}. This is "
            "not a MuPoTS annotation file."
        )
    annotations = contents["annotations"]
    if annotations.ndim != 2:
        raise GroundTruthError(
            f"{path} holds an annotations array of {annotations.ndim} dimension(s); "
            "the release stores frames by person slots, which is two."
        )
    return annotations


def _read_person(cell: Any, field: str, slot: int) -> GroundTruthPerson | None:  # noqa: ANN401
    """One person in one frame, or None where the slot holds nobody."""
    try:
        valid_frame = int(np.asarray(cell["isValidFrame"][0, 0]).ravel()[0])
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise GroundTruthError(
            f"an annotation cell has no readable isValidFrame: {error}"
        ) from error
    if not valid_frame:
        return None

    joints_3d = _array(cell, field, expected=3)
    joints_2d = _array(cell, "annot2", expected=2)
    if joints_3d.shape[1] != len(JOINT_NAMES):
        raise GroundTruthError(
            f"{field} has {joints_3d.shape[1]} joint(s); the evaluation set has "
            f"{len(JOINT_NAMES)}. The release changed, or this is a different joint set."
        )

    return GroundTruthPerson(
        # The slot is stable within a sequence, which is what the temporal
        # metrics need. It is a slot, not a verified identity, and nothing here
        # claims otherwise.
        ground_truth_person_id=f"slot{slot}",
        joints_3d=(joints_3d.T / _MM_PER_METRE).astype(np.float64),
        joint_names=JOINT_NAMES,
        # The release marks a whole person valid or not, never one joint, so
        # every joint of a present person is annotated.
        joint_valid=np.ones(len(JOINT_NAMES), dtype=np.bool_),
        joints_2d=joints_2d.T.astype(np.float64),
        root_translation=(joints_3d[:, PELVIS_INDEX] / _MM_PER_METRE).astype(np.float64),
    )


def _array(cell: Any, field: str, *, expected: int) -> FloatArray:  # noqa: ANN401
    try:
        value = np.asarray(cell[field][0, 0], dtype=np.float64)
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise GroundTruthError(f"an annotation cell has no readable {field}: {error}") from error
    if value.ndim != 2 or value.shape[0] != expected:
        raise GroundTruthError(
            f"{field} has shape {value.shape}; the release stores it as ({expected}, joints)"
        )
    return value


__all__ = [
    "DATASET_NAME",
    "JOINT_NAMES",
    "PELVIS_INDEX",
    "SEQUENCE_COUNT",
    "MupotsDataset",
    "SequenceIntrinsics",
    "image_size",
    "recover_intrinsics",
]
