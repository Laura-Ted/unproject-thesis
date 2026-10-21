"""The raw per-person record, and the tables derived from it.

ADR-0004 §7.2 makes this the source of truth: `observations.npz` holds one row
per matched person and one row per frame, and every table and figure is a view
computed from it. Nothing is summarised on the way in, so a question nobody
thought to ask before the run can still be answered after it.

Everything is a plain numeric array. No object dtype, no pickle - an NPZ with
object arrays is a pickle wearing a different extension, and this one is written
by a long run that nobody wants to repeat because a reader refused it.

Version 2 of the file renames the two focal-length arrays to say where each
came from - `annotation_derived` and `model_estimated`, never `released` - and
adds a body-scale proxy per matched person, on both sides, for the hypothesis
ADR-0005 WP5 tests. A version 1 file is read under the new names; a proxy it
never recorded is NaN, and `has_scale_proxy` says so.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from unproject.ground_truth.artifacts import Coverage
from unproject.ground_truth.config import MetricThresholds
from unproject.ground_truth.dataset import BoolArray, FloatArray, GroundTruthError
from unproject.ground_truth.metrics import (
    Aggregate,
    DepthBin,
    PersonErrors,
    aggregate,
    depth_bins,
    pck,
    per_joint,
    root_error,
)

IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int32]]

OBSERVATIONS_SCHEMA_VERSION = 2

# Written in this order so the file's key order does not depend on dict
# iteration; a reader checks for exactly these.
_PERSON_KEYS = (
    "sequence_index",
    "frame_index",
    "ground_truth_slot",
    "prediction_track",
    "absolute_mm",
    "root_relative_mm",
    "joint_valid",
    "root_error_xyz_mm",
    "depth_m",
    # Bilateral femur plus tibia, metres, on each side of the match; NaN when a
    # leg joint is invalid or the file predates the field. See
    # `metrics.scale_proxy`.
    "scale_proxy_predicted_m",
    "scale_proxy_annotated_m",
)
_FRAME_KEYS = (
    "frame_sequence_index",
    "frame_number",
    "annotated_people",
    "predicted_people",
    "matched_people",
)
# One value per sequence. The annotation-derived focal length is recovered by
# least squares from the released 2D and 3D; the model-estimated one is what the
# estimator invented for the same frames. Their ratio is the mechanism behind
# the depth bias, so it travels with the observations rather than being
# recomputed from the dataset later.
_SEQUENCE_KEYS = ("sequence_fx_annotation_derived", "sequence_fx_model_estimated")

# The v1 spellings of the renamed keys, so a v1 file still loads.
_LEGACY_KEYS = {
    "sequence_fx_released": "sequence_fx_annotation_derived",
    "sequence_fx_estimated": "sequence_fx_model_estimated",
}
_ADDED_IN_V2 = ("scale_proxy_predicted_m", "scale_proxy_annotated_m")


@dataclass(frozen=True, slots=True)
class Observations:
    """One row per matched person, one row per frame, and nothing aggregated."""

    joint_names: tuple[str, ...]
    sequence_names: tuple[str, ...]

    sequence_index: IntArray
    frame_index: IntArray
    ground_truth_slot: IntArray
    prediction_track: IntArray
    absolute_mm: FloatArray
    root_relative_mm: FloatArray
    joint_valid: BoolArray
    root_error_xyz_mm: FloatArray
    depth_m: FloatArray
    scale_proxy_predicted_m: FloatArray
    scale_proxy_annotated_m: FloatArray

    # Every frame, matched or not. Without these a sequence where nothing was
    # detected would simply be absent from the results rather than reported as
    # a failure, which is the exact reading ADR-0004 §6.1 forbids.
    frame_sequence_index: IntArray
    frame_number: IntArray
    annotated_people: IntArray
    predicted_people: IntArray
    matched_people: IntArray

    sequence_fx_annotation_derived: FloatArray
    sequence_fx_model_estimated: FloatArray

    # False when read from a v1 file, whose rows carry no proxy. A NaN column
    # would otherwise be indistinguishable from "every leg joint was invalid".
    has_scale_proxy: bool = True

    def __post_init__(self) -> None:
        people = self.sequence_index.shape[0]
        joints = len(self.joint_names)
        for name in (
            "frame_index",
            "ground_truth_slot",
            "prediction_track",
            "depth_m",
            "scale_proxy_predicted_m",
            "scale_proxy_annotated_m",
        ):
            if getattr(self, name).shape != (people,):
                raise GroundTruthError(f"{name} has {getattr(self, name).shape}, expected {people}")
        for name in ("absolute_mm", "root_relative_mm", "joint_valid"):
            if getattr(self, name).shape != (people, joints):
                raise GroundTruthError(
                    f"{name} is {getattr(self, name).shape}, expected ({people}, {joints})"
                )
        if self.root_error_xyz_mm.shape != (people, 3):
            raise GroundTruthError(f"root errors are {self.root_error_xyz_mm.shape}")
        frames = self.frame_sequence_index.shape[0]
        for name in _FRAME_KEYS[1:]:
            if getattr(self, name).shape != (frames,):
                raise GroundTruthError(f"{name} has {getattr(self, name).shape}, expected {frames}")
        for name in _SEQUENCE_KEYS:
            if getattr(self, name).shape != (len(self.sequence_names),):
                raise GroundTruthError(
                    f"{name} has {getattr(self, name).shape}, expected one per sequence "
                    f"({len(self.sequence_names)})"
                )
        if int(self.matched_people.sum()) != people:
            raise GroundTruthError(
                f"{int(self.matched_people.sum())} matches counted per frame but "
                f"{people} person rows were written; one was lost between the two"
            )

    @property
    def coverage(self) -> Coverage:
        """The books, balanced. Unmatched predictions fill `false_positives` as a
        count, not a verdict: whether an unmatched prediction is an error depends
        on whether the release annotates everyone, and MuPoTS does not."""
        matched = int(self.matched_people.sum())
        return Coverage(
            samples=self.frame_sequence_index.shape[0],
            ground_truth_people=int(self.annotated_people.sum()),
            predictions=int(self.predicted_people.sum()),
            matched=matched,
            false_negatives=int(self.annotated_people.sum()) - matched,
            false_positives=int(self.predicted_people.sum()) - matched,
            valid_joints=int(self.joint_valid.sum()),
        )

    def for_sequence(self, name: str) -> Self:
        """The same observations, narrowed to one sequence."""
        if name not in self.sequence_names:
            raise GroundTruthError(f"{name!r} is not among {list(self.sequence_names)}")
        index = self.sequence_names.index(name)
        people = self.sequence_index == index
        frames = self.frame_sequence_index == index
        return type(self)(
            joint_names=self.joint_names,
            sequence_names=(name,),
            sequence_index=np.zeros(int(people.sum()), dtype=np.int32),
            frame_index=self.frame_index[people],
            ground_truth_slot=self.ground_truth_slot[people],
            prediction_track=self.prediction_track[people],
            absolute_mm=self.absolute_mm[people],
            root_relative_mm=self.root_relative_mm[people],
            joint_valid=self.joint_valid[people],
            root_error_xyz_mm=self.root_error_xyz_mm[people],
            depth_m=self.depth_m[people],
            scale_proxy_predicted_m=self.scale_proxy_predicted_m[people],
            scale_proxy_annotated_m=self.scale_proxy_annotated_m[people],
            frame_sequence_index=np.zeros(int(frames.sum()), dtype=np.int32),
            frame_number=self.frame_number[frames],
            annotated_people=self.annotated_people[frames],
            predicted_people=self.predicted_people[frames],
            matched_people=self.matched_people[frames],
            sequence_fx_annotation_derived=self.sequence_fx_annotation_derived[index : index + 1],
            sequence_fx_model_estimated=self.sequence_fx_model_estimated[index : index + 1],
            has_scale_proxy=self.has_scale_proxy,
        )

    def save(self, path: Path) -> None:
        arrays = {key: getattr(self, key) for key in (*_PERSON_KEYS, *_FRAME_KEYS, *_SEQUENCE_KEYS)}
        arrays["schema_version"] = np.array(OBSERVATIONS_SCHEMA_VERSION, dtype=np.int32)
        arrays["joint_names"] = np.array(self.joint_names, dtype="U")
        arrays["sequence_names"] = np.array(self.sequence_names, dtype="U")
        with path.open("wb") as handle:
            np.savez(handle, **arrays)

    @classmethod
    def load(cls, path: Path) -> Self:
        try:
            with np.load(path, allow_pickle=False) as data:
                stored = {key: data[key] for key in data.files}
        except (OSError, ValueError) as error:
            raise GroundTruthError(f"cannot read {path}: {error}") from error

        version = int(stored.pop("schema_version", np.array(1)))
        if version not in (1, OBSERVATIONS_SCHEMA_VERSION):
            raise GroundTruthError(
                f"{path} is observations schema {version}; this build reads 1 and "
                f"{OBSERVATIONS_SCHEMA_VERSION}"
            )
        has_scale_proxy = version >= 2
        if version == 1:
            # Labels only: the arrays are the same numbers under their new names.
            for old_key, new_key in _LEGACY_KEYS.items():
                if old_key in stored:
                    stored[new_key] = stored.pop(old_key)
            rows = stored["sequence_index"].shape[0] if "sequence_index" in stored else 0
            for key in _ADDED_IN_V2:
                stored.setdefault(key, np.full(rows, np.nan))

        missing = sorted(
            {*_PERSON_KEYS, *_FRAME_KEYS, *_SEQUENCE_KEYS, "joint_names"} - set(stored)
        )
        if missing:
            raise GroundTruthError(f"{path} is missing {missing}; it is not an observations file")
        return cls(
            joint_names=tuple(str(name) for name in stored.pop("joint_names")),
            sequence_names=tuple(str(name) for name in stored.pop("sequence_names")),
            has_scale_proxy=has_scale_proxy,
            **stored,
        )


def collect(
    joint_names: tuple[str, ...],
    sequence_names: tuple[str, ...],
    people: list[tuple[int, int, int, int, PersonErrors]],
    frames: list[tuple[int, int, int, int, int]],
    focal_lengths: list[tuple[float, float]],
) -> Observations:
    """Stack the per-person and per-frame rows into arrays.

    `people` rows are (sequence, frame, ground-truth slot, prediction track,
    errors); `frames` rows are (sequence, frame, annotated, predicted, matched);
    `focal_lengths` is (annotation-derived fx, model-estimated fx) per
    sequence, in order.
    """
    if len(focal_lengths) != len(sequence_names):
        raise GroundTruthError(
            f"{len(focal_lengths)} focal-length pairs for {len(sequence_names)} sequences"
        )
    focal = np.array(focal_lengths, dtype=np.float64).reshape(-1, 2)
    joints = len(joint_names)
    columns = np.array([row[:4] for row in people], dtype=np.int32).reshape(-1, 4)
    frame_columns = np.array(frames, dtype=np.int32).reshape(-1, 5)
    return Observations(
        joint_names=joint_names,
        sequence_names=sequence_names,
        sequence_index=columns[:, 0],
        frame_index=columns[:, 1],
        ground_truth_slot=columns[:, 2],
        prediction_track=columns[:, 3],
        absolute_mm=np.array([row[4].absolute_mm for row in people]).reshape(-1, joints),
        root_relative_mm=np.array([row[4].root_relative_mm for row in people]).reshape(-1, joints),
        joint_valid=np.array([row[4].valid for row in people], dtype=np.bool_).reshape(-1, joints),
        root_error_xyz_mm=np.array([row[4].root_error_xyz_mm for row in people]).reshape(-1, 3),
        depth_m=np.array([row[4].depth_m for row in people], dtype=np.float64).reshape(-1),
        scale_proxy_predicted_m=np.array(
            [row[4].scale_proxy_predicted_m for row in people], dtype=np.float64
        ).reshape(-1),
        scale_proxy_annotated_m=np.array(
            [row[4].scale_proxy_annotated_m for row in people], dtype=np.float64
        ).reshape(-1),
        frame_sequence_index=frame_columns[:, 0],
        frame_number=frame_columns[:, 1],
        annotated_people=frame_columns[:, 2],
        predicted_people=frame_columns[:, 3],
        matched_people=frame_columns[:, 4],
        sequence_fx_annotation_derived=focal[:, 0],
        sequence_fx_model_estimated=focal[:, 1],
    )


@dataclass(frozen=True, slots=True)
class Summary:
    """Every headline number for one set of observations."""

    coverage: Coverage
    recall: float
    # The official protocol's headline (relative), and the one this project
    # cares about (absolute).
    pck_relative: Aggregate
    pck_absolute: Aggregate
    mpjpe_mm: Aggregate
    mpjpe_relative_mm: Aggregate
    root_mae_xyz_mm: FloatArray
    root_bias_xyz_mm: FloatArray
    depth: list[DepthBin]


def summarise(observations: Observations, thresholds: MetricThresholds) -> Summary:
    """Reduce raw observations to the reported numbers, and nothing else.

    Recall is computed here rather than left to the reader because §6.1 forbids
    an accuracy figure that travels without one.
    """
    valid = observations.joint_valid
    annotated = int(observations.annotated_people.sum())
    matched = int(observations.matched_people.sum())
    return Summary(
        coverage=observations.coverage,
        recall=matched / annotated if annotated else float("nan"),
        pck_relative=pck(observations.root_relative_mm, valid, radius_mm=thresholds.pck_radius_mm),
        pck_absolute=pck(observations.absolute_mm, valid, radius_mm=thresholds.pck_radius_mm),
        mpjpe_mm=aggregate(observations.absolute_mm, valid),
        mpjpe_relative_mm=aggregate(observations.root_relative_mm, valid),
        root_mae_xyz_mm=root_error(observations.root_error_xyz_mm)[0],
        root_bias_xyz_mm=root_error(observations.root_error_xyz_mm)[1],
        depth=depth_bins(
            observations.depth_m, observations.root_error_xyz_mm, thresholds.depth_bins_metres
        ),
    )


def write_per_sequence(
    observations: Observations, thresholds: MetricThresholds, path: Path
) -> None:
    """One row per sequence, plus the total. A derived view, never the source."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "sequence",
                "frames",
                "annotated",
                "predicted",
                "matched",
                "recall",
                "pck_rel_micro",
                "pck_abs_micro",
                "mpjpe_mm",
                "mpjpe_rel_mm",
                "root_bias_z_mm",
                "joints",
                "fx_annotation_derived",
                "fx_model_estimated",
                "focal_ratio_estimated_over_derived",
            ]
        )
        for name in (*observations.sequence_names, "ALL"):
            subset = observations if name == "ALL" else observations.for_sequence(name)
            summary = summarise(subset, thresholds)
            writer.writerow(
                [
                    name,
                    subset.frame_sequence_index.shape[0],
                    summary.coverage.ground_truth_people,
                    summary.coverage.predictions,
                    summary.coverage.matched,
                    _number(summary.recall, 4),
                    _number(summary.pck_relative.micro, 2),
                    _number(summary.pck_absolute.micro, 2),
                    _number(summary.mpjpe_mm.micro, 1),
                    _number(summary.mpjpe_relative_mm.micro, 1),
                    _number(float(summary.root_bias_xyz_mm[2]), 1),
                    summary.coverage.valid_joints,
                    *_focal_columns(subset),
                ]
            )


def write_per_joint(observations: Observations, thresholds: MetricThresholds, path: Path) -> None:
    """One row per joint of the common skeleton."""
    absolute, present = per_joint(observations.absolute_mm, observations.joint_valid)
    relative, _ = per_joint(observations.root_relative_mm, observations.joint_valid)
    inside = np.where(
        observations.joint_valid, observations.root_relative_mm <= thresholds.pck_radius_mm, False
    )
    hits, _ = per_joint(inside.astype(np.float64) * 100.0, observations.joint_valid)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["joint", "observations", "mpjpe_mm", "mpjpe_rel_mm", "pck_rel"])
        for index, name in enumerate(observations.joint_names):
            writer.writerow(
                [
                    name,
                    int(observations.joint_valid[:, index].sum()),
                    _number(float(absolute[index]), 1) if present[index] else "",
                    _number(float(relative[index]), 1) if present[index] else "",
                    _number(float(hits[index]), 2) if present[index] else "",
                ]
            )


def _focal_columns(subset: Observations) -> list[str]:
    """Annotation-derived and model-estimated fx, and their ratio; blank on the
    ALL row, where a single focal length would be a mean over different cameras."""
    if len(subset.sequence_names) != 1:
        return ["", "", ""]
    derived = float(subset.sequence_fx_annotation_derived[0])
    estimated = float(subset.sequence_fx_model_estimated[0])
    return [_number(derived, 1), _number(estimated, 1), _number(estimated / derived, 4)]


def _number(value: float, places: int) -> str:
    """Fixed decimals so a table regenerates byte for byte from the same file."""
    return "" if not np.isfinite(value) else f"{value:.{places}f}"
