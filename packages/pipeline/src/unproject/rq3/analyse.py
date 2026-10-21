"""One controlled recording, scored end to end.

Reads a finished bodies run and its annotations and writes a summary. Runs no
inference and touches no artifact: the analysis is a pure function of a run and
a hand-written file, so it can be rerun until the numbers stop changing without
anything being recomputed.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from unproject.rq3.annotations import CONVENTION, Annotation
from unproject.rq3.metrics import (
    AxisMotion,
    PositionError,
    ReprojectionConsistency,
    ShapeVariation,
    SmoothingEffect,
    TrackFragmentation,
    TrackSummary,
    axis_motion,
    position_errors,
    reprojection_consistency,
    shape_variation,
    smoothing_effect,
    track_fragmentation,
    track_summaries,
)
from unproject.rq3.reader import Run

# 2 split reprojection consistency by whether the 2D box is clipped by the
#   frame edge. The single figure was confounded and read far worse than it was.
#   Added overall_motion and overall_smoothing, so the half of RQ3 that needs no
#   ground truth can be measured on a run that has no annotated intervals.
RQ3_SUMMARY_SCHEMA_VERSION = 2

SUMMARY_FILENAME = "rq3.json"
SUMMARY_CSV_FILENAME = "rq3-positions.csv"


class Rq3Summary(BaseModel):
    """Everything RQ3 has to say about one recording."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=RQ3_SUMMARY_SCHEMA_VERSION, ge=1)
    # Named in the file so a table can state it without consulting the source.
    convention: str = CONVENTION
    units: str = "metres and seconds; pixels where stated"
    annotation: str
    ingest_hash: str
    # Raw, and always present. The filter is scored beside the raw figures, in
    # `overall_smoothing` and `smoothing`, never instead of them.
    positions: list[PositionError]
    # Every detection in the run, respecting track boundaries. Needs no ground
    # truth, so it is the one motion figure an uncontrolled scene can produce.
    overall_motion: AxisMotion
    # The filter scored over the same detections, so what it buys and what it
    # costs can be checked without a controlled scene either.
    overall_smoothing: SmoothingEffect
    stationary_motion: list[AxisMotion]
    walking_motion: list[AxisMotion]
    smoothing: list[SmoothingEffect]
    shapes: list[ShapeVariation]
    # One entry per track, for inspecting them one at a time. ADR-0003 §3.2.
    # A track is not a person; the viewer says "track" for that reason.
    per_track: list[TrackSummary]
    fragmentation: TrackFragmentation
    reprojection: ReprojectionConsistency


def analyse(run: Run, annotation: Annotation) -> Rq3Summary:
    """Score one run against its annotations. Deterministic, given both."""
    stationary = [
        axis_motion(
            run.between(interval.start_seconds, interval.end_seconds, interval.track_id),
            label=f"still at {interval.marker}",
            track_id=interval.track_id,
        )
        for interval in annotation.stationary
    ]
    walking = [
        axis_motion(
            run.between(walk.start_seconds, walk.end_seconds, walk.track_id),
            label=f"{walk.from_marker} to {walk.to_marker}",
            track_id=walk.track_id,
        )
        for walk in annotation.motion
    ]
    smoothed = [
        smoothing_effect(
            run.between(walk.start_seconds, walk.end_seconds, walk.track_id),
            label=f"{walk.from_marker} to {walk.to_marker}",
            track_id=walk.track_id,
        )
        for walk in annotation.motion
    ] + [
        smoothing_effect(
            run.between(interval.start_seconds, interval.end_seconds, interval.track_id),
            label=f"still at {interval.marker}",
            track_id=interval.track_id,
        )
        for interval in annotation.stationary
    ]

    return Rq3Summary(
        annotation=annotation.name,
        ingest_hash=run.artifact.ingest_hash,
        positions=position_errors(run, annotation),
        overall_motion=axis_motion(run.detections, label="whole run", track_id=None),
        overall_smoothing=smoothing_effect(run.detections, label="whole run", track_id=None),
        stationary_motion=stationary,
        walking_motion=walking,
        smoothing=smoothed,
        shapes=shape_variation(run),
        per_track=track_summaries(run),
        fragmentation=track_fragmentation(run, annotation.scene.participants),
        reprojection=reprojection_consistency(run),
    )


def positions_csv(summary: Rq3Summary) -> str:
    """The position table, for inspection and for thesis tooling.

    Carries the marker uncertainty in its own column: a depth error quoted
    without the ruler it was measured against is not reportable.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "marker",
            "track_id",
            "samples",
            "lateral_bias_metres",
            "depth_bias_metres",
            "lateral_abs_median_metres",
            "depth_abs_median_metres",
            "marker_uncertainty_metres",
            "depth_error_exceeds_uncertainty",
        ]
    )
    for row in summary.positions:
        writer.writerow(
            [
                row.marker,
                _cell(row.track_id),
                row.samples,
                _cell(row.lateral_bias_metres),
                _cell(row.depth_bias_metres),
                _cell(row.lateral_abs_median_metres),
                _cell(row.depth_abs_median_metres),
                row.marker_uncertainty_metres,
                str(row.depth_error_exceeds_uncertainty).lower(),
            ]
        )
    return buffer.getvalue()


def _cell(value: float | int | None) -> str:
    """Empty rather than zero: a missing measurement is not a measurement of zero."""
    return "" if value is None else str(value)


def write_summary(summary: Rq3Summary, directory: Path) -> tuple[Path, Path]:
    """Write both forms into the directory the analysis was asked to fill."""
    directory.mkdir(parents=True, exist_ok=True)
    as_json = directory / SUMMARY_FILENAME
    as_csv = directory / SUMMARY_CSV_FILENAME
    as_json.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    as_csv.write_text(positions_csv(summary), encoding="utf-8")
    return as_json, as_csv
