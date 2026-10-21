"""The shape every dataset is converted into before anything measures it.

Two releases with different conventions, units and joint sets meet here and
nowhere else. A metric function never sees a MuPoTS array or an Anny-One array;
it sees a `DatasetRecord` that has already been checked.

The checking is the point. A silent unit error, a flipped axis or a
root-relative annotation read as camera-space all produce plausible numbers, and
a plausible wrong number is the failure mode this whole evaluation exists to
avoid. So conversion fails closed and says what it wanted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# The one frame everything is converted into. Repeated as a literal in every
# record so a reader never has to infer it.
CANONICAL_CONVENTION = "camera_relative_opencv_metres"

# Anything beyond this is a person on the far side of a stadium, and far more
# likely a millimetre annotation read as metres.
_MAX_PLAUSIBLE_DEPTH_METRES = 200.0


class GroundTruthError(Exception):
    """Raised when a release cannot be read into the canonical contract."""


@dataclass(frozen=True, slots=True)
class Camera:
    """Pinhole intrinsics at the resolution of the image they belong to."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True, slots=True)
class GroundTruthPerson:
    """One annotated person in one image."""

    # Null where the release does not carry stable identity. Temporal metrics
    # need it and are skipped rather than guessed when it is absent.
    ground_truth_person_id: str | None
    # (J, 3), metres, camera-relative OpenCV: +X right, +Y down, +Z forward.
    joints_3d: FloatArray
    joint_names: tuple[str, ...]
    # (J,) False where the release says the joint is unannotated. Invalid joints
    # never enter a metric; they are counted.
    joint_valid: BoolArray
    # (J, 2) in pixels, when the release provides it. Person association uses
    # this and never the 3D positions.
    joints_2d: FloatArray | None = None
    joint_visible: BoolArray | None = None
    root_translation: FloatArray | None = None
    # Native Anny parameters, only where the release is Anny-One.
    anny_pose: FloatArray | None = None
    anny_shape: FloatArray | None = None

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One image or video frame, with everything annotated in it."""

    dataset_name: str
    release_identity: str
    sample_id: str
    # Null for a still-image release. Sequence boundaries decide where a tracker
    # session starts, so they are never inferred from file order.
    sequence_id: str | None
    frame_index: int | None
    image_path: Path
    image_width: int
    image_height: int
    people: tuple[GroundTruthPerson, ...]
    camera: Camera | None = None
    convention: str = CANONICAL_CONVENTION


class GroundTruthDataset(Protocol):
    """What both adapters must offer, and nothing more.

    Two implementations justify this; it is not a plugin point. Anything only
    one adapter needs belongs on that adapter.
    """

    @property
    def name(self) -> str: ...

    @property
    def release_identity(self) -> str: ...

    def sequences(self) -> list[str]:
        """Sequence identifiers, or a single synthetic one for a still release."""
        ...

    def records(self, sequence_id: str) -> list[DatasetRecord]:
        """Every validated record in one sequence, in frame order."""
        ...


def validate_record(record: DatasetRecord, *, source: Path | str) -> DatasetRecord:
    """Check one record and return it, or say exactly what is wrong.

    `source` is named in every message: an error that does not say which file it
    came from costs more to act on than it saves to raise.
    """
    where = f"{record.dataset_name} record {record.sample_id} from {source}"

    if record.convention != CANONICAL_CONVENTION:
        raise GroundTruthError(
            f"{where} declares convention {record.convention!r}; the evaluator "
            f"only reads {CANONICAL_CONVENTION!r}. Convert in the adapter, and "
            "test the conversion against a synthetic transform rather than by "
            "looking at a rendered skeleton."
        )
    if record.image_width <= 0 or record.image_height <= 0:
        raise GroundTruthError(
            f"{where} has image size {record.image_width}x{record.image_height}; "
            "both sides must be positive. Read them from the released metadata "
            "rather than from the decoded image."
        )
    for index, person in enumerate(record.people):
        _validate_person(person, where=f"{where}, person {index}")
    return record


def _validate_person(person: GroundTruthPerson, *, where: str) -> None:
    count = person.joint_count
    if count == 0:
        raise GroundTruthError(f"{where} has no joints, so nothing about it can be measured")

    _check_array(person.joints_3d, (count, 3), where=where, field="joints_3d")
    if person.joint_valid.shape != (count,):
        raise GroundTruthError(
            f"{where} has {count} joint name(s) but a validity mask of shape "
            f"{person.joint_valid.shape}; they describe the same joints and must "
            "have the same length."
        )
    if len(set(person.joint_names)) != count:
        raise GroundTruthError(
            f"{where} repeats a joint name. Names address joints in the mapping, "
            "so a duplicate silently sends two joints to one target."
        )

    # Only the valid joints have to be finite: a release is free to park an
    # unannotated joint at NaN, and dropping it is the correct reading.
    valid = person.joints_3d[person.joint_valid]
    if valid.size and not np.isfinite(valid).all():
        raise GroundTruthError(
            f"{where} has a non-finite value among the joints it marks valid. "
            "Either the validity mask or the conversion is wrong."
        )
    if valid.size and float(np.abs(valid).max()) > _MAX_PLAUSIBLE_DEPTH_METRES:
        raise GroundTruthError(
            f"{where} has a joint {float(np.abs(valid).max()):.1f} from the camera. "
            f"The contract is metres and the limit is {_MAX_PLAUSIBLE_DEPTH_METRES:.0f}; "
            "this is what a millimetre annotation looks like when it is not converted."
        )

    if person.joints_2d is not None:
        _check_array(person.joints_2d, (count, 2), where=where, field="joints_2d")
    if person.joint_visible is not None and person.joint_visible.shape != (count,):
        raise GroundTruthError(
            f"{where} has a visibility mask of shape {person.joint_visible.shape}, "
            f"expected ({count},)"
        )
    if person.root_translation is not None:
        _check_array(person.root_translation, (3,), where=where, field="root_translation")


def _check_array(array: FloatArray, expected: tuple[int, ...], *, where: str, field: str) -> None:
    if array.shape != expected:
        raise GroundTruthError(f"{where} has {field} of shape {array.shape}, expected {expected}")
    if not np.issubdtype(array.dtype, np.floating):
        raise GroundTruthError(
            f"{where} has {field} of dtype {array.dtype}; a float array is required, "
            "because an integer array here means a unit was truncated somewhere."
        )
