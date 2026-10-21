"""The evaluation's numbers as one JSON file, derived from the observations.

`summary.json` is what a thesis table is read from and what the evidence bundle
carries (ADR-0005 WP7): every headline, the coverage it was computed over, the
protocol and matcher it was computed under, the depth error as a distribution
with its outliers counted, and the per-sequence and per-joint rows. It holds
nothing the observations do not, and regenerates from them byte for byte.

A view, never the source. `observations.npz` is the record; if this file and
that one disagree, this one is wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from unproject.ground_truth.artifacts import Coverage, MatchingIdentity
from unproject.ground_truth.config import CameraKind, MetricProtocol, MetricThresholds
from unproject.ground_truth.metrics import Aggregate, RobustDepth, per_joint, robust_depth
from unproject.ground_truth.observations import Observations, summarise

SUMMARY_SCHEMA_VERSION = 1
SUMMARY_FILENAME = "summary.json"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AggregateRow(_Frozen):
    micro: float | None
    macro: float | None
    people: int
    joints: int


class DepthDistribution(_Frozen):
    """The signed depth error, prediction minus annotation, in millimetres."""

    people: int
    mean_mm: float | None
    median_mm: float | None
    p10_mm: float | None
    p25_mm: float | None
    p75_mm: float | None
    p90_mm: float | None
    iqr_mm: float | None
    mad_mm: float | None
    abs_median_mm: float | None
    abs_p90_mm: float | None
    relative_median: float | None
    gross_outliers: int
    mad_outliers: int


class DepthBinRow(_Frozen):
    lower_m: float | None
    upper_m: float | None
    people: int
    absolute_error_mm: float | None
    signed_bias_mm: float | None
    relative_bias: float | None
    distribution: DepthDistribution


class SequenceRow(_Frozen):
    sequence: str
    frames: int
    annotated: int
    predicted: int
    matched: int
    recall: float | None
    pck_rel_micro: float | None
    pck_abs_micro: float | None
    mpjpe_mm: float | None
    mpjpe_rel_mm: float | None
    root_bias_z_mm: float | None
    root_bias_z_median_mm: float | None
    joints: int
    fx_annotation_derived: float | None
    fx_model_estimated: float | None
    focal_ratio_estimated_over_derived: float | None


class JointRow(_Frozen):
    joint: str
    observations: int
    mpjpe_mm: float | None
    mpjpe_rel_mm: float | None
    pck_rel: float | None


class EvaluationSummary(_Frozen):
    schema_version: int = Field(default=SUMMARY_SCHEMA_VERSION, ge=1)
    evaluation_id: str
    metric_protocol: MetricProtocol
    matching: MatchingIdentity
    camera_source: CameraKind
    final: bool
    thresholds: MetricThresholds
    coverage: Coverage
    recall: float | None
    pck_relative: AggregateRow
    pck_absolute: AggregateRow
    mpjpe_mm: AggregateRow
    mpjpe_relative_mm: AggregateRow
    root_mae_xyz_mm: list[float | None]
    root_bias_xyz_mm: list[float | None]
    root_bias_median_xyz_mm: list[float | None]
    depth: DepthDistribution
    depth_bins: list[DepthBinRow]
    has_scale_proxy: bool
    per_sequence: list[SequenceRow]
    per_joint: list[JointRow]


def build_summary(
    observations: Observations,
    thresholds: MetricThresholds,
    *,
    evaluation_id: str,
    metric_protocol: MetricProtocol,
    matching: MatchingIdentity,
    camera_source: CameraKind,
    final: bool,
) -> EvaluationSummary:
    """Every number, from the observations and nothing else."""
    summary = summarise(observations, thresholds)
    depth = robust_depth(observations.depth_m, observations.root_error_xyz_mm)
    return EvaluationSummary(
        evaluation_id=evaluation_id,
        metric_protocol=metric_protocol,
        matching=matching,
        camera_source=camera_source,
        final=final,
        thresholds=thresholds,
        coverage=summary.coverage,
        recall=_finite(summary.recall),
        pck_relative=_aggregate(summary.pck_relative),
        pck_absolute=_aggregate(summary.pck_absolute),
        mpjpe_mm=_aggregate(summary.mpjpe_mm),
        mpjpe_relative_mm=_aggregate(summary.mpjpe_relative_mm),
        root_mae_xyz_mm=[_finite(float(v)) for v in summary.root_mae_xyz_mm],
        root_bias_xyz_mm=[_finite(float(v)) for v in summary.root_bias_xyz_mm],
        root_bias_median_xyz_mm=_median_xyz(observations),
        depth=_distribution(depth),
        depth_bins=[
            DepthBinRow(
                lower_m=_finite(bin_.lower_m),
                upper_m=_finite(bin_.upper_m),
                people=bin_.people,
                absolute_error_mm=_finite(bin_.absolute_error_mm),
                signed_bias_mm=_finite(bin_.signed_bias_mm),
                relative_bias=_finite(bin_.relative_bias),
                distribution=_distribution(_bin_depth(observations, bin_.lower_m, bin_.upper_m)),
            )
            for bin_ in summary.depth
        ],
        has_scale_proxy=observations.has_scale_proxy,
        per_sequence=[
            _sequence_row(observations, name, thresholds) for name in observations.sequence_names
        ],
        per_joint=_joint_rows(observations, thresholds),
    )


def write_summary(summary: EvaluationSummary, directory: Path) -> Path:
    path = directory / SUMMARY_FILENAME
    path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _sequence_row(
    observations: Observations, name: str, thresholds: MetricThresholds
) -> SequenceRow:
    subset = observations.for_sequence(name)
    summary = summarise(subset, thresholds)
    derived = float(subset.sequence_fx_annotation_derived[0])
    estimated = float(subset.sequence_fx_model_estimated[0])
    depth = subset.root_error_xyz_mm[:, 2]
    return SequenceRow(
        sequence=name,
        frames=int(subset.frame_sequence_index.shape[0]),
        annotated=summary.coverage.ground_truth_people,
        predicted=summary.coverage.predictions,
        matched=summary.coverage.matched,
        recall=_finite(summary.recall),
        pck_rel_micro=_finite(summary.pck_relative.micro),
        pck_abs_micro=_finite(summary.pck_absolute.micro),
        mpjpe_mm=_finite(summary.mpjpe_mm.micro),
        mpjpe_rel_mm=_finite(summary.mpjpe_relative_mm.micro),
        root_bias_z_mm=_finite(float(summary.root_bias_xyz_mm[2])),
        root_bias_z_median_mm=_finite(float(np.median(depth))) if depth.size else None,
        joints=summary.coverage.valid_joints,
        fx_annotation_derived=_finite(derived),
        fx_model_estimated=_finite(estimated),
        focal_ratio_estimated_over_derived=_finite(estimated / derived) if derived else None,
    )


def _joint_rows(observations: Observations, thresholds: MetricThresholds) -> list[JointRow]:
    absolute, present = per_joint(observations.absolute_mm, observations.joint_valid)
    relative, _ = per_joint(observations.root_relative_mm, observations.joint_valid)
    inside = np.where(
        observations.joint_valid, observations.root_relative_mm <= thresholds.pck_radius_mm, False
    )
    hits, _ = per_joint(inside.astype(np.float64) * 100.0, observations.joint_valid)
    return [
        JointRow(
            joint=name,
            observations=int(observations.joint_valid[:, index].sum()),
            mpjpe_mm=_finite(float(absolute[index])) if present[index] else None,
            mpjpe_rel_mm=_finite(float(relative[index])) if present[index] else None,
            pck_rel=_finite(float(hits[index])) if present[index] else None,
        )
        for index, name in enumerate(observations.joint_names)
    ]


def _bin_depth(observations: Observations, lower: float, upper: float) -> RobustDepth:
    inside = (observations.depth_m >= lower) & (observations.depth_m < upper)
    return robust_depth(observations.depth_m[inside], observations.root_error_xyz_mm[inside])


def _distribution(depth: RobustDepth) -> DepthDistribution:
    return DepthDistribution(
        people=depth.people,
        mean_mm=_finite(depth.mean_mm),
        median_mm=_finite(depth.median_mm),
        p10_mm=_finite(depth.p10_mm),
        p25_mm=_finite(depth.p25_mm),
        p75_mm=_finite(depth.p75_mm),
        p90_mm=_finite(depth.p90_mm),
        iqr_mm=_finite(depth.iqr_mm),
        mad_mm=_finite(depth.mad_mm),
        abs_median_mm=_finite(depth.abs_median_mm),
        abs_p90_mm=_finite(depth.abs_p90_mm),
        relative_median=_finite(depth.relative_median),
        gross_outliers=depth.gross_outliers,
        mad_outliers=depth.mad_outliers,
    )


def _median_xyz(observations: Observations) -> list[float | None]:
    if observations.root_error_xyz_mm.shape[0] == 0:
        return [None, None, None]
    return [_finite(float(v)) for v in np.median(observations.root_error_xyz_mm, axis=0)]


def _aggregate(value: Aggregate) -> AggregateRow:
    return AggregateRow(
        micro=_finite(value.micro),
        macro=_finite(value.macro),
        people=value.people,
        joints=value.joints,
    )


def _finite(value: float) -> float | None:
    """JSON has no NaN; an absent number is null, never a string."""
    return None if not np.isfinite(value) else round(float(value), 6)


__all__ = [
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA_VERSION",
    "DepthDistribution",
    "EvaluationSummary",
    "JointRow",
    "SequenceRow",
    "build_summary",
    "write_summary",
]
