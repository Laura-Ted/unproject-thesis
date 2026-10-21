"""Two evaluations side by side. ADR-0005 WP4.

Two questions, and the same tool answers both.

**Did a migration change anything but labels?** WP1 renamed the vocabulary and
promised the numbers would not move; WP4 regenerates the thesis evaluations
from a clean commit and has to show that only identifiers and the fields the
migration added differ. `compare` puts every observation array of one
evaluation beside the other's and says whether they are bit-identical, and
lists every descriptor field that differs. A migration that moves a number is
then visible as exactly that, and the ADR says what to do about it: a new
protocol identifier and an explanation, not a quiet edit.

**What did the intervention do, per sequence?** The camera experiment is a
control and a treatment - `model_estimated` against `annotation_derived` - and
no single evaluation can draw the paired figure. This can: for every sequence
both evaluations scored, the depth bias, the placement PCK and the MPJPE of one
against the other, as a CSV and a figure, from the two observation files and
nothing else.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from unproject.ground_truth.artifacts import EvaluationDescriptor
from unproject.ground_truth.dataset import GroundTruthError
from unproject.ground_truth.evaluate import read_evaluation
from unproject.ground_truth.observations import Observations, summarise

if TYPE_CHECKING:
    from matplotlib.figure import Figure

PAIRED_CSV = "paired-per-sequence.csv"
PAIRED_FIGURE = "paired-depth-bias.svg"
DIFF_FILENAME = "descriptor-diff.txt"

# The person and frame arrays whose equality decides "same numbers". The
# sequence-level focal lengths are included; the joint and sequence name lists
# are compared as labels.
_NUMERIC_KEYS = (
    "sequence_index",
    "frame_index",
    "ground_truth_slot",
    "prediction_track",
    "absolute_mm",
    "root_relative_mm",
    "joint_valid",
    "root_error_xyz_mm",
    "depth_m",
    "frame_sequence_index",
    "frame_number",
    "annotated_people",
    "predicted_people",
    "matched_people",
    "sequence_fx_annotation_derived",
    "sequence_fx_model_estimated",
)
# Present only from observations schema 2; compared when both sides carry it.
_OPTIONAL_KEYS = ("scale_proxy_predicted_m", "scale_proxy_annotated_m")

_HASH_SALT = "unproject-ground-truth-compare"


@dataclass(frozen=True, slots=True)
class ArrayDifference:
    key: str
    # Both shapes, so a length change reads as what it is.
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    # Largest absolute difference where the shapes agree; None otherwise.
    max_abs_difference: float | None


@dataclass(frozen=True, slots=True)
class Comparison:
    a: Path
    b: Path
    # Every numeric array both sides carry is bit-identical.
    identical_numbers: bool
    array_differences: tuple[ArrayDifference, ...]
    # Arrays one side has and the other lacks (schema 1 against 2).
    only_in_a: tuple[str, ...]
    only_in_b: tuple[str, ...]
    same_labels: bool
    # Descriptor fields that differ, as "field: a -> b" lines.
    descriptor_differences: tuple[str, ...]
    shared_sequences: tuple[str, ...]


def compare_numbers(a: Observations, b: Observations) -> Comparison:
    """Are these the same numbers under different labels?"""
    differences: list[ArrayDifference] = []
    for key in _NUMERIC_KEYS:
        differences.extend(_compare_key(key, getattr(a, key), getattr(b, key)))
    only_a: list[str] = []
    only_b: list[str] = []
    for key in _OPTIONAL_KEYS:
        if a.has_scale_proxy and b.has_scale_proxy:
            differences.extend(_compare_key(key, getattr(a, key), getattr(b, key)))
        elif a.has_scale_proxy:
            only_a.append(key)
        elif b.has_scale_proxy:
            only_b.append(key)
    return Comparison(
        a=Path(),
        b=Path(),
        identical_numbers=not differences,
        array_differences=tuple(differences),
        only_in_a=tuple(only_a),
        only_in_b=tuple(only_b),
        same_labels=(a.joint_names == b.joint_names and a.sequence_names == b.sequence_names),
        descriptor_differences=(),
        shared_sequences=tuple(name for name in a.sequence_names if name in b.sequence_names),
    )


def _compare_key(key: str, left: np.ndarray, right: np.ndarray) -> list[ArrayDifference]:
    if left.shape != right.shape:
        return [ArrayDifference(key, tuple(left.shape), tuple(right.shape), None)]
    if left.dtype == np.bool_ or right.dtype == np.bool_:
        if np.array_equal(left, right):
            return []
        return [ArrayDifference(key, tuple(left.shape), tuple(right.shape), 1.0)]
    # NaN against NaN is equal here: an invalid joint is invalid on both sides.
    if np.array_equal(left, right, equal_nan=True):
        return []
    with np.errstate(invalid="ignore"):
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    largest = float(np.nanmax(delta)) if np.isfinite(delta).any() else float("inf")
    return [ArrayDifference(key, tuple(left.shape), tuple(right.shape), largest)]


def compare_descriptors(a: EvaluationDescriptor, b: EvaluationDescriptor) -> tuple[str, ...]:
    """Every top-level field that differs, one line each, nested fields flattened."""
    left = a.model_dump(mode="json")
    right = b.model_dump(mode="json")
    lines: list[str] = []
    _walk(left, right, "", lines)
    return tuple(lines)


def _walk(left: object, right: object, prefix: str, lines: list[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left:
                lines.append(f"{path}: (absent) -> {right[key]!r}")
            elif key not in right:
                lines.append(f"{path}: {left[key]!r} -> (absent)")
            else:
                _walk(left[key], right[key], path, lines)
        return
    if left != right:
        lines.append(f"{prefix}: {left!r} -> {right!r}")


def compare_evaluations(directory_a: Path, directory_b: Path) -> Comparison:
    descriptor_a, observations_a = read_evaluation(directory_a)
    descriptor_b, observations_b = read_evaluation(directory_b)
    numbers = compare_numbers(observations_a, observations_b)
    return Comparison(
        a=directory_a,
        b=directory_b,
        identical_numbers=numbers.identical_numbers,
        array_differences=numbers.array_differences,
        only_in_a=numbers.only_in_a,
        only_in_b=numbers.only_in_b,
        same_labels=numbers.same_labels,
        descriptor_differences=compare_descriptors(descriptor_a, descriptor_b),
        shared_sequences=numbers.shared_sequences,
    )


def write_paired(
    directory_a: Path, directory_b: Path, out: Path, *, label_a: str, label_b: str
) -> list[Path]:
    """The per-sequence paired table and figure, b beside a.

    Only sequences both evaluations scored appear; the two must share the
    joint set, or the numbers are not comparable and the call is refused.
    """
    descriptor_a, observations_a = read_evaluation(directory_a)
    descriptor_b, observations_b = read_evaluation(directory_b)
    if observations_a.joint_names != observations_b.joint_names:
        raise GroundTruthError(
            "the two evaluations use different common skeletons "
            f"({list(observations_a.joint_names)} against {list(observations_b.joint_names)}); "
            "their numbers are not comparable"
        )
    shared = [
        name for name in observations_a.sequence_names if name in observations_b.sequence_names
    ]
    if not shared:
        raise GroundTruthError("the two evaluations share no sequence")

    out.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []
    biases: list[tuple[str, float, float]] = []
    for name in shared:
        sa = summarise(observations_a.for_sequence(name), descriptor_a.config.thresholds)
        sb = summarise(observations_b.for_sequence(name), descriptor_b.config.thresholds)
        bias_a = float(sa.root_bias_xyz_mm[2])
        bias_b = float(sb.root_bias_xyz_mm[2])
        biases.append((name, bias_a, bias_b))
        rows.append(
            [
                name,
                str(sa.coverage.matched),
                str(sb.coverage.matched),
                f"{bias_a:.1f}",
                f"{bias_b:.1f}",
                f"{bias_b - bias_a:+.1f}",
                f"{sa.pck_absolute.micro:.2f}",
                f"{sb.pck_absolute.micro:.2f}",
                f"{sa.pck_relative.micro:.2f}",
                f"{sb.pck_relative.micro:.2f}",
                f"{sa.mpjpe_mm.micro:.1f}",
                f"{sb.mpjpe_mm.micro:.1f}",
            ]
        )
    table = out / PAIRED_CSV
    with table.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "sequence",
                f"matched_{label_a}",
                f"matched_{label_b}",
                f"root_bias_z_mm_{label_a}",
                f"root_bias_z_mm_{label_b}",
                "root_bias_z_mm_difference",
                f"pck_abs_{label_a}",
                f"pck_abs_{label_b}",
                f"pck_rel_{label_a}",
                f"pck_rel_{label_b}",
                f"mpjpe_mm_{label_a}",
                f"mpjpe_mm_{label_b}",
            ]
        )
        writer.writerows(rows)
    figure = _paired_figure(biases, label_a=label_a, label_b=label_b)
    figure_path = out / PAIRED_FIGURE
    figure.savefig(figure_path, format="svg", bbox_inches="tight", metadata={"Date": None})
    _close(figure)
    return [table, figure_path]


def _paired_figure(biases: list[tuple[str, float, float]], *, label_a: str, label_b: str) -> Figure:
    """Depth bias per sequence, one bar pair each, zero drawn."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = _HASH_SALT
    import matplotlib.pyplot as plt

    names = [name for name, _, _ in biases]
    a = np.array([value for _, value, _ in biases])
    b = np.array([value for _, _, value in biases])
    positions = np.arange(len(names))
    width = 0.38
    figure, axes = plt.subplots(figsize=(max(7.0, 0.45 * len(names) + 2.0), 4.0))
    axes.bar(positions - width / 2, a, width, color="#4878a8", label=label_a)
    axes.bar(positions + width / 2, b, width, color="#a84848", label=label_b)
    axes.axhline(0.0, color="#333333", linewidth=1.0)
    axes.set_xticks(positions, names, rotation=45, fontsize=8, ha="right")
    axes.set_ylabel("mean signed depth error (mm)")
    axes.set_title(
        f"Depth bias per sequence, {label_b} beside {label_a}\n"
        "camera-relative OpenCV axes: +Z away from the camera",
        fontsize=9,
    )
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure


def _close(figure: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)


def describe(comparison: Comparison) -> str:
    """The comparison as text, for the terminal and for the diff file."""
    lines = [f"a: {comparison.a}", f"b: {comparison.b}"]
    if comparison.identical_numbers:
        lines.append(
            "numbers: identical - every observation array both sides carry is bit-for-bit equal"
        )
    else:
        lines.append("numbers: DIFFERENT")
        for difference in comparison.array_differences:
            if difference.max_abs_difference is None:
                lines.append(
                    f"  {difference.key}: shape {difference.shape_a} -> {difference.shape_b}"
                )
            else:
                lines.append(
                    f"  {difference.key}: max |a - b| = {difference.max_abs_difference:.6g}"
                )
    if comparison.only_in_a:
        lines.append(f"arrays only in a: {', '.join(comparison.only_in_a)}")
    if comparison.only_in_b:
        lines.append(f"arrays only in b: {', '.join(comparison.only_in_b)}")
    lines.append(f"labels (joints, sequences): {'same' if comparison.same_labels else 'DIFFERENT'}")
    lines.append(f"descriptor fields that differ: {len(comparison.descriptor_differences)}")
    lines.extend(f"  {line}" for line in comparison.descriptor_differences)
    return "\n".join(lines)


__all__ = [
    "DIFF_FILENAME",
    "PAIRED_CSV",
    "PAIRED_FIGURE",
    "ArrayDifference",
    "Comparison",
    "compare_descriptors",
    "compare_evaluations",
    "compare_numbers",
    "describe",
    "write_paired",
]
