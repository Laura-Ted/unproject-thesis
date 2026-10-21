"""What the errors are, once the people have been paired.

Pure arrays in, numbers out: no file reading, no dataset knowledge, no torch.
That is what makes the boundary cases in ADR-0004 §8.1 testable by hand - a
149 mm joint and a 151 mm joint, a root offset that must move the absolute
error and leave the relative one alone.

Two distinctions run through the whole module and are never collapsed:

**Absolute against root-relative.** Root-relative removes where the person is
and asks only about the shape of the pose. Absolute keeps it. The official
MuPoTS protocol reports root-relative; the question this project exists to
answer is absolute. Reporting one under the other's name would answer the wrong
question convincingly.

**Micro against macro.** Micro pools every joint of every person, so a crowded
frame counts for more. Macro averages per person and then over people, so
everyone counts once. They differ whenever detection is uneven, which is exactly
when the difference matters, so both are always carried.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from unproject.ground_truth.dataset import BoolArray, FloatArray, GroundTruthError

_MM_PER_METRE = 1000.0


@dataclass(frozen=True, slots=True)
class PersonErrors:
    """One matched person's errors, per joint, in millimetres."""

    # (J,) absolute error in camera space, and after subtracting each side's own
    # root. Invalid joints hold NaN and are excluded by `valid`.
    absolute_mm: FloatArray
    root_relative_mm: FloatArray
    valid: BoolArray
    # (3,) signed prediction minus ground truth, in millimetres.
    root_error_xyz_mm: FloatArray
    # Ground-truth depth of the root, in metres, for the distance bins.
    depth_m: float
    # Bilateral femur plus tibia on each side, metres; NaN without the six leg
    # joints. See `scale_proxy`.
    scale_proxy_predicted_m: float = float("nan")
    scale_proxy_annotated_m: float = float("nan")


def person_errors(
    gt_3d: FloatArray,
    pred_3d: FloatArray,
    valid: BoolArray,
    *,
    root_index: int,
    names: tuple[str, ...] | None = None,
) -> PersonErrors:
    """Compare one matched pair, in metres in, millimetres out.

    Both sides must already be on the common skeleton and in the same camera
    frame. The root joint must be valid: without it there is nothing to align on,
    and a root-relative error taken against a missing root would silently become
    an absolute one.

    With `names`, the body-scale proxy is measured on both sides as well; it is
    a length, so it is taken before anything is aligned or subtracted.
    """
    if gt_3d.shape != pred_3d.shape or gt_3d.shape[1] != 3:
        raise GroundTruthError(
            f"expected matching (J, 3) arrays, got {gt_3d.shape}/{pred_3d.shape}"
        )
    if valid.shape != (gt_3d.shape[0],):
        raise GroundTruthError(f"{valid.shape} validity for {gt_3d.shape[0]} joints")
    if not 0 <= root_index < gt_3d.shape[0]:
        raise GroundTruthError(f"root index {root_index} is outside {gt_3d.shape[0]} joints")
    if not bool(valid[root_index]):
        raise GroundTruthError(
            "the root joint is not valid for this person, so there is nothing to "
            "align on. Root-relative error against a missing root is absolute error."
        )

    absolute = np.linalg.norm(pred_3d - gt_3d, axis=1) * _MM_PER_METRE
    centred = (pred_3d - pred_3d[root_index]) - (gt_3d - gt_3d[root_index])
    relative = np.linalg.norm(centred, axis=1) * _MM_PER_METRE

    absolute = np.where(valid, absolute, np.nan)
    relative = np.where(valid, relative, np.nan)
    predicted_scale = float("nan")
    annotated_scale = float("nan")
    if names is not None:
        predicted_scale = scale_proxy(pred_3d, valid, names)
        annotated_scale = scale_proxy(gt_3d, valid, names)
    return PersonErrors(
        absolute_mm=absolute,
        root_relative_mm=relative,
        valid=valid.copy(),
        root_error_xyz_mm=(pred_3d[root_index] - gt_3d[root_index]) * _MM_PER_METRE,
        depth_m=float(gt_3d[root_index, 2]),
        scale_proxy_predicted_m=predicted_scale,
        scale_proxy_annotated_m=annotated_scale,
    )


@dataclass(frozen=True, slots=True)
class Aggregate:
    """One error measure, both ways of averaging it.

    `micro` pools joints; `macro` averages within a person first. `people` and
    `joints` are carried so a value computed over three joints cannot be quoted
    as if it were computed over three thousand.
    """

    micro: float
    macro: float
    people: int
    joints: int


def aggregate(errors: FloatArray, valid: BoolArray) -> Aggregate:
    """Mean error, pooled and per person. `errors` is (people, joints)."""
    _check_table(errors, valid)
    counts = valid.sum(axis=1)
    if not valid.any():
        return Aggregate(micro=float("nan"), macro=float("nan"), people=0, joints=0)

    pooled = float(np.nanmean(np.where(valid, errors, np.nan)))
    per_person = np.full(errors.shape[0], np.nan)
    contributing = counts > 0
    per_person[contributing] = np.nanmean(np.where(valid, errors, np.nan)[contributing], axis=1)
    return Aggregate(
        micro=pooled,
        macro=float(np.nanmean(per_person[contributing])),
        people=int(contributing.sum()),
        joints=int(counts.sum()),
    )


def pck(errors: FloatArray, valid: BoolArray, *, radius_mm: float) -> Aggregate:
    """Percentage of joints inside the radius, pooled and per person.

    The comparison is `<=`, matching the released evaluator. At the 150 mm
    threshold that decides a joint sitting at exactly 150 mm, which is why
    ADR-0004 §8.1 asks for a 149 and a 151 case rather than trusting the reading.
    """
    _check_table(errors, valid)
    if radius_mm <= 0.0:
        raise GroundTruthError(f"the PCK radius must be positive, got {radius_mm}")
    inside = np.where(valid, errors <= radius_mm, False)
    return aggregate(inside.astype(np.float64) * 100.0, valid)


def per_joint(errors: FloatArray, valid: BoolArray) -> tuple[FloatArray, BoolArray]:
    """Mean error for each joint across people, and which joints had any.

    A joint nobody annotated must come back flagged rather than as zero: zero is
    a perfect score, and an absent joint is not one.
    """
    _check_table(errors, valid)
    counts = valid.sum(axis=0)
    means = np.full(errors.shape[1], np.nan)
    present = counts > 0
    if present.any():
        means[present] = np.nanmean(np.where(valid, errors, np.nan)[:, present], axis=0)
    return means, present


def root_error(root_errors_mm: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Per-axis mean absolute error and signed bias, in millimetres.

    Both, because they answer different questions. A model that puts everyone
    30 cm too far away and a model that scatters people by 30 cm in either
    direction have the same MAE; only the bias tells them apart, and a bias is
    what a wrong focal length produces.
    """
    if root_errors_mm.ndim != 2 or root_errors_mm.shape[1] != 3:
        raise GroundTruthError(f"expected (people, 3) root errors, got {root_errors_mm.shape}")
    if root_errors_mm.shape[0] == 0:
        empty = np.full(3, np.nan)
        return empty, empty.copy()
    return np.abs(root_errors_mm).mean(axis=0), root_errors_mm.mean(axis=0)


@dataclass(frozen=True, slots=True)
class DepthBin:
    """Depth error for the people annotated within one distance band."""

    lower_m: float
    upper_m: float
    people: int
    absolute_error_mm: float
    signed_bias_mm: float
    # Bias as a fraction of the true distance, which is the form a focal-length
    # error takes: a constant proportion rather than a constant offset.
    relative_bias: float


def depth_bins(
    depths_m: FloatArray, root_errors_mm: FloatArray, edges_m: list[float]
) -> list[DepthBin]:
    """Depth error split by how far away the person actually was.

    Edges are interior boundaries, so N edges give N+1 bins with the outermost
    two open. Bins are half-open at the top, `lower <= d < upper`, so a person at
    exactly an edge lands in one bin rather than two.
    """
    if depths_m.shape[0] != root_errors_mm.shape[0]:
        raise GroundTruthError(
            f"{depths_m.shape[0]} depths against {root_errors_mm.shape[0]} root errors"
        )
    if edges_m != sorted(edges_m) or len(set(edges_m)) != len(edges_m):
        raise GroundTruthError(f"depth bin edges must strictly ascend, got {edges_m}")

    bounds = [-np.inf, *edges_m, np.inf]
    bins: list[DepthBin] = []
    for lower, upper in pairwise(bounds):
        inside = (depths_m >= lower) & (depths_m < upper)
        count = int(inside.sum())
        if count == 0:
            bins.append(
                DepthBin(
                    lower_m=float(lower),
                    upper_m=float(upper),
                    people=0,
                    absolute_error_mm=float("nan"),
                    signed_bias_mm=float("nan"),
                    relative_bias=float("nan"),
                )
            )
            continue
        depth_error = root_errors_mm[inside, 2]
        true_mm = depths_m[inside] * _MM_PER_METRE
        bins.append(
            DepthBin(
                lower_m=float(lower),
                upper_m=float(upper),
                people=count,
                absolute_error_mm=float(np.abs(depth_error).mean()),
                signed_bias_mm=float(depth_error.mean()),
                relative_bias=float((depth_error / true_mm).mean()),
            )
        )
    return bins


# A depth error beyond this is a gross mis-association rather than a placement
# error - a person put at nine metres who was annotated at five. Counted, never
# excluded: the means keep them, and the count says how many there were.
GROSS_DEPTH_ERROR_MM = 1000.0

# Scales a median absolute deviation to the standard deviation of a normal
# distribution, so "3 MAD" reads like "3 sigma" for a reader who expects one.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True, slots=True)
class RobustDepth:
    """The depth error described by its distribution, not only its mean.

    ADR-0004 §6.5 asks for median and interquartile range beside the mean of a
    skewed error, and ADR-0005 WP4 for the outliers to be preserved and
    counted. Every field is over the signed root depth error, prediction minus
    annotation, in millimetres, unless named `abs_`.
    """

    people: int
    mean_mm: float
    median_mm: float
    p25_mm: float
    p75_mm: float
    p10_mm: float
    p90_mm: float
    mad_mm: float
    abs_median_mm: float
    abs_p90_mm: float
    # Median of error over annotated distance: the proportional reading.
    relative_median: float
    # |error| beyond GROSS_DEPTH_ERROR_MM, and beyond three scaled MADs from
    # the median. Two definitions, one physical and one statistical.
    gross_outliers: int
    mad_outliers: int

    @property
    def iqr_mm(self) -> float:
        return self.p75_mm - self.p25_mm


def robust_depth(depths_m: FloatArray, root_errors_mm: FloatArray) -> RobustDepth:
    """Distribution of the signed depth error. Nearest-rank percentiles."""
    if depths_m.shape[0] != root_errors_mm.shape[0]:
        raise GroundTruthError(
            f"{depths_m.shape[0]} depths against {root_errors_mm.shape[0]} root errors"
        )
    if root_errors_mm.ndim != 2 or root_errors_mm.shape[1] != 3:
        raise GroundTruthError(f"expected (people, 3) root errors, got {root_errors_mm.shape}")
    error = root_errors_mm[:, 2].astype(np.float64)
    count = int(error.shape[0])
    if count == 0:
        nan = float("nan")
        return RobustDepth(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0, 0)
    ordered = np.sort(error)
    median = _nearest_rank(ordered, 50)
    mad = float(np.median(np.abs(error - median)))
    absolute = np.sort(np.abs(error))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = error / (depths_m * _MM_PER_METRE)
    return RobustDepth(
        people=count,
        mean_mm=float(error.mean()),
        median_mm=median,
        p25_mm=_nearest_rank(ordered, 25),
        p75_mm=_nearest_rank(ordered, 75),
        p10_mm=_nearest_rank(ordered, 10),
        p90_mm=_nearest_rank(ordered, 90),
        mad_mm=mad,
        abs_median_mm=_nearest_rank(absolute, 50),
        abs_p90_mm=_nearest_rank(absolute, 90),
        relative_median=_nearest_rank(np.sort(relative), 50),
        gross_outliers=int((np.abs(error) > GROSS_DEPTH_ERROR_MM).sum()),
        mad_outliers=int((np.abs(error - median) > 3.0 * _MAD_TO_SIGMA * mad).sum())
        if mad > 0.0
        else 0,
    )


def _nearest_rank(ordered: FloatArray, percentile: int) -> float:
    """A value that was measured, as every other percentile in the project."""
    rank = max(1, -(-percentile * ordered.shape[0] // 100))
    return float(ordered[min(rank, ordered.shape[0]) - 1])


# The segments the body-scale proxy is built from, hip to knee and knee to
# ankle on each side. Declared here, before any analysis reads it, per ADR-0005
# WP5: a proxy chosen after seeing the residual would be fitted to it.
SCALE_PROXY_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def scale_proxy(joints: FloatArray, valid: BoolArray, names: tuple[str, ...]) -> float:
    """Bilateral femur plus tibia length, in metres, from one skeleton.

    A proxy for body size that both sides define the same way: hip, knee and
    ankle are direct joints on the common skeleton, and a bone length does not
    move with where the person is put. It is what the body-scale hypothesis is
    tested with - the ratio of predicted to annotated proxy against the relative
    depth error - so it is recorded per person at evaluation time rather than
    reconstructed later from something else.

    NaN when any of the six joints is invalid: a leg with a missing knee is not
    a shorter leg.
    """
    if joints.shape != (len(names), 3) or valid.shape != (len(names),):
        raise GroundTruthError(
            f"expected ({len(names)}, 3) joints and ({len(names)},) validity, got "
            f"{joints.shape} and {valid.shape}"
        )
    index = {name: position for position, name in enumerate(names)}
    missing = sorted({name for pair in SCALE_PROXY_SEGMENTS for name in pair} - set(index))
    if missing:
        raise GroundTruthError(
            f"the common skeleton has no {missing}; the scale proxy needs hips, knees and ankles"
        )
    total = 0.0
    for start, end in SCALE_PROXY_SEGMENTS:
        if not (bool(valid[index[start]]) and bool(valid[index[end]])):
            return float("nan")
        total += float(np.linalg.norm(joints[index[end]] - joints[index[start]]))
    return total


def _check_table(errors: FloatArray, valid: BoolArray) -> None:
    if errors.shape != valid.shape:
        raise GroundTruthError(f"{errors.shape} errors against {valid.shape} validity")
    if errors.ndim != 2:
        raise GroundTruthError(f"expected a (people, joints) table, got {errors.shape}")
    if np.isnan(errors[valid]).any():
        raise GroundTruthError("a joint marked valid has no error; it cannot be both")
