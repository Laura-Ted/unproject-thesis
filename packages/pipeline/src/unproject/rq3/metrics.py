"""What RQ3 asks: how stable, and how wrong, are the bodies.

Every number here is per axis, because that is the whole point. Sideways the
model is good; along the camera axis it is guessing, and a single figure
averaging the two would hide exactly the thing being measured.

**Units and frame.** Metres and seconds, camera-relative OpenCV axes, +X right,
+Y down, +Z forward. Pixels where a number is on the image plane.

Missing detections stay missing. Nothing here interpolates across a frame where
nobody was found: a gap is reported as a gap, per ADR-0002 §8.5.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field

from unproject.rq3.annotations import Annotation, Marker
from unproject.rq3.reader import Detection, Run
from unproject.rq3.smoothing import DEFAULT_SMOOTHING, Smoothing, TrackSmoother

# A pair of detections further apart than this is a gap rather than a step, and
# a velocity across it would describe the gap rather than the person.
MAX_STEP_SECONDS = 0.5

# How close a box has to come to the frame before the body counts as cut off by
# the shot. One pixel: the detector clamps to the image, so a clipped box sits
# exactly on the edge.
_EDGE_PIXELS = 1.0

CONVENTION = "camera_relative_opencv_metres"


class PositionError(BaseModel):
    """How far a standing body was placed from where the tape said it was."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    marker: str
    track_id: int | None
    samples: int = Field(ge=0)
    # Signed medians: the bias, which is what says whether depth is pushed near
    # or far rather than only that it is wrong.
    lateral_bias_metres: float | None = None
    depth_bias_metres: float | None = None
    lateral_abs_median_metres: float | None = None
    depth_abs_median_metres: float | None = None
    # The ruler this was measured against. An error under it is not a result.
    marker_uncertainty_metres: float
    depth_error_exceeds_uncertainty: bool = False


class AxisMotion(BaseModel):
    """Frame-to-frame movement, split by axis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    track_id: int | None
    steps: int = Field(ge=0)
    # Pairs dropped for being a gap rather than a step. Visible, not silent.
    gaps_skipped: int = Field(default=0, ge=0)
    lateral_median_metres_per_second: float | None = None
    depth_median_metres_per_second: float | None = None
    lateral_p95_metres_per_second: float | None = None
    depth_p95_metres_per_second: float | None = None
    # Jitter as a fraction of how far away the body is, which is the form the
    # error actually takes.
    depth_jitter_fraction_of_range: float | None = None


class ShapeVariation(BaseModel):
    """How much one person's estimated body changed while they stood there.

    A person's shape does not change. Variation here is the model re-guessing,
    and it bounds how much a per-frame mesh would be worth.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: int
    samples: int = Field(ge=0)
    max_parameter_std: float | None = None
    mean_parameter_std: float | None = None


class TrackFragmentation(BaseModel):
    """Whether identities survived the recording."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tracks: int = Field(ge=0)
    participants: int = Field(ge=1)
    # Above 1 means identities churned: one person became several tracks.
    tracks_per_participant: float = Field(ge=0.0)
    median_track_seconds: float | None = None
    longest_gap_seconds: float | None = None


class ReprojectionConsistency(BaseModel):
    """Whether the 3D root lands on the person it came from.

    Not an accuracy measure: it checks the recorded intrinsics against the
    recorded 2D box, so it catches a coordinate convention read the wrong way
    round rather than a body in the wrong place.

    **Split by whether the box touches the frame edge, and only the clear figure
    means anything.** A person walking out of the bottom of the shot has their
    legs cut off, so their box stops at the image edge while their pelvis is
    genuinely below it. Counting those as failures put the figure on the crowd
    clip at 58% when the bodies whose boxes were fully in frame were at 100%.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int = Field(ge=0)
    # Detections whose 2D box is fully inside the frame. This is the number to
    # read: nothing about it is confounded by the frame edge.
    clear_samples: int = Field(default=0, ge=0)
    clear_inside: int = Field(default=0, ge=0)
    clear_inside_fraction: float | None = None
    # Detections whose box is clipped by the frame edge. Reported so the split
    # is visible, but a body cut off by the shot may legitimately have its
    # pelvis outside it, so a low figure here is not an inconsistency.
    clipped_samples: int = Field(default=0, ge=0)
    clipped_inside: int = Field(default=0, ge=0)
    clipped_inside_fraction: float | None = None
    median_outside_px: float | None = None


class SmoothingEffect(BaseModel):
    """What the filter bought, and what it cost.

    Reported beside the raw numbers, never instead of them: the raw jitter is
    the project's own result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    raw: AxisMotion
    smoothed: AxisMotion
    # Seconds the smoothed track lags the raw one, by best alignment. The price
    # of the stability above it.
    response_delay_seconds: float | None = None


def position_errors(run: Run, annotation: Annotation) -> list[PositionError]:
    """Lateral and depth error at each annotated standing position."""
    return [
        _one_position_error(
            run.between(interval.start_seconds, interval.end_seconds, interval.track_id),
            annotation.marker(interval.marker),
            interval.track_id,
        )
        for interval in annotation.stationary
    ]


def _one_position_error(
    detections: Sequence[Detection], marker: Marker, track_id: int | None
) -> PositionError:
    if not detections:
        return PositionError(
            marker=marker.id,
            track_id=track_id,
            samples=0,
            marker_uncertainty_metres=marker.uncertainty_metres,
        )

    lateral = [float(detection.root_translation[0]) - marker.x_metres for detection in detections]
    depth = [float(detection.root_translation[2]) - marker.z_metres for detection in detections]
    depth_abs = statistics.median(abs(value) for value in depth)
    return PositionError(
        marker=marker.id,
        track_id=track_id,
        samples=len(detections),
        lateral_bias_metres=_round(statistics.median(lateral)),
        depth_bias_metres=_round(statistics.median(depth)),
        lateral_abs_median_metres=_round(statistics.median(abs(v) for v in lateral)),
        depth_abs_median_metres=_round(depth_abs),
        marker_uncertainty_metres=marker.uncertainty_metres,
        depth_error_exceeds_uncertainty=depth_abs > marker.uncertainty_metres,
    )


def axis_motion(detections: Sequence[Detection], *, label: str, track_id: int | None) -> AxisMotion:
    """Per-axis displacement rate between consecutive detections of each track.

    Grouped by track first. Walking the detections in time order and skipping
    pairs whose track changed looks equivalent and is not: with a dozen people
    in a frame, consecutive entries are different people and every pair is
    skipped. On the crowd clip that produced zero steps out of 2832 detections.
    """
    tracks: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        tracks[detection.track_id].append(detection)

    lateral: list[float] = []
    depth: list[float] = []
    fractions: list[float] = []
    gaps = 0

    for one in tracks.values():
        ordered = sorted(one, key=lambda detection: detection.time_seconds)
        for before, after in pairwise(ordered):
            dt = after.time_seconds - before.time_seconds
            if dt <= 0.0 or dt > MAX_STEP_SECONDS:
                gaps += 1
                continue
            dx = abs(float(after.root_translation[0]) - float(before.root_translation[0]))
            dz = abs(float(after.root_translation[2]) - float(before.root_translation[2]))
            lateral.append(dx / dt)
            depth.append(dz / dt)
            range_metres = max(abs(float(after.root_translation[2])), 1e-6)
            fractions.append(dz / range_metres)

    return AxisMotion(
        label=label,
        track_id=track_id,
        steps=len(lateral),
        gaps_skipped=gaps,
        lateral_median_metres_per_second=_round(_median(lateral)),
        depth_median_metres_per_second=_round(_median(depth)),
        lateral_p95_metres_per_second=_round(_percentile(lateral, 95)),
        depth_p95_metres_per_second=_round(_percentile(depth, 95)),
        depth_jitter_fraction_of_range=_round(_median(fractions)),
    )


class TrackSummary(BaseModel):
    """One track on its own.

    A track is not a person. It is one unbroken run of detections the tracker
    kept together, and ADR-0003 §3.1 is explicit that the id means nothing
    outside this run. When one person becomes four tracks, four of these appear
    and their lifetimes abut - which is the finding, not a defect to hide.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: int
    detections: int = Field(ge=0)
    first_seconds: float = Field(ge=0.0)
    last_seconds: float = Field(ge=0.0)
    # The largest hole inside this track: what a viewer sees as a body
    # disappearing and coming back.
    longest_gap_seconds: float | None = None
    # How far away this track was, which is what the depth error scales with.
    median_range_metres: float | None = None
    motion: AxisMotion
    shape: ShapeVariation


def track_summaries(run: Run) -> list[TrackSummary]:
    """Every track, summarised on its own, in id order.

    Reuses the same per-axis metrics the rest of RQ3 uses, so a number in the
    viewer and a number in the table cannot disagree.
    """
    summaries: list[TrackSummary] = []
    shapes = {variation.track_id: variation for variation in shape_variation(run)}

    for track_id in run.track_ids:
        detections = sorted(run.track(track_id), key=lambda d: d.time_seconds)
        times = [d.time_seconds for d in detections]
        ranges = [abs(float(d.root_translation[2])) for d in detections]
        gaps = [after - before for before, after in pairwise(times)]
        summaries.append(
            TrackSummary(
                track_id=track_id,
                detections=len(detections),
                first_seconds=times[0] if times else 0.0,
                last_seconds=times[-1] if times else 0.0,
                longest_gap_seconds=_round(max(gaps)) if gaps else None,
                median_range_metres=_round(_median(ranges)),
                motion=axis_motion(detections, label=f"track {track_id}", track_id=track_id),
                shape=shapes[track_id],
            )
        )
    return summaries


def shape_variation(run: Run) -> list[ShapeVariation]:
    """Within-track spread of the estimated body shape, per track."""
    variations: list[ShapeVariation] = []
    for track_id in run.track_ids:
        detections = run.track(track_id)
        if len(detections) < 2:
            variations.append(ShapeVariation(track_id=track_id, samples=len(detections)))
            continue
        # One standard deviation per shape parameter, then summarised across
        # them: a single parameter drifting is as much a problem as all of them.
        columns = list(zip(*[detection.shape.tolist() for detection in detections], strict=True))
        deviations = [statistics.stdev(column) for column in columns]
        variations.append(
            ShapeVariation(
                track_id=track_id,
                samples=len(detections),
                max_parameter_std=_round(max(deviations)),
                mean_parameter_std=_round(statistics.fmean(deviations)),
            )
        )
    return variations


def track_fragmentation(run: Run, participants: int) -> TrackFragmentation:
    """Tracks against people, and the worst hole inside a track."""
    tracks = run.track_ids
    lifetimes: list[float] = []
    longest_gap: float | None = None

    for track_id in tracks:
        detections = run.track(track_id)
        times = sorted(detection.time_seconds for detection in detections)
        if len(times) >= 2:
            lifetimes.append(times[-1] - times[0])
            gap = max(after - before for before, after in pairwise(times))
            longest_gap = gap if longest_gap is None else max(longest_gap, gap)

    return TrackFragmentation(
        tracks=len(tracks),
        participants=participants,
        tracks_per_participant=round(len(tracks) / participants, 6),
        median_track_seconds=_round(_median(lifetimes)),
        longest_gap_seconds=_round(longest_gap),
    )


def reprojection_consistency(run: Run) -> ReprojectionConsistency:
    """Project each root through its own frame's intrinsics and see where it lands."""
    cameras = {frame.frame_index: frame.camera for frame in run.artifact.frames}
    outside: list[float] = []
    samples = 0
    clear = [0, 0]
    clipped = [0, 0]

    for detection in run.detections:
        camera = cameras.get(detection.frame_index)
        if camera is None:
            continue
        samples += 1
        left, top, right, bottom = detection.box_xyxy
        bucket = clipped if _touches_edge(detection, camera.width, camera.height) else clear
        bucket[1] += 1

        x, y, z = (float(value) for value in detection.root_translation[:3])
        if z <= 0.0:
            # Behind the camera: nothing to project, and a real result.
            outside.append(float("inf"))
            continue
        u = camera.fx * x / z + camera.cx
        v = camera.fy * y / z + camera.cy
        if left <= u <= right and top <= v <= bottom:
            bucket[0] += 1
        else:
            outside.append(max(left - u, u - right, 0.0) + max(top - v, v - bottom, 0.0))

    finite = [value for value in outside if value != float("inf")]
    return ReprojectionConsistency(
        samples=samples,
        clear_samples=clear[1],
        clear_inside=clear[0],
        clear_inside_fraction=_round(clear[0] / clear[1]) if clear[1] else None,
        clipped_samples=clipped[1],
        clipped_inside=clipped[0],
        clipped_inside_fraction=_round(clipped[0] / clipped[1]) if clipped[1] else None,
        median_outside_px=_round(_median(finite)),
    )


def _touches_edge(detection: Detection, width: int, height: int) -> bool:
    """Whether the 2D box runs into the frame, so the body is cut off by the shot."""
    left, top, right, bottom = detection.box_xyxy
    return (
        left <= _EDGE_PIXELS
        or top <= _EDGE_PIXELS
        or right >= width - 1 - _EDGE_PIXELS
        or bottom >= height - 1 - _EDGE_PIXELS
    )


def smoothing_effect(
    detections: Sequence[Detection],
    *,
    label: str,
    track_id: int | None,
    options: Smoothing = DEFAULT_SMOOTHING,
) -> SmoothingEffect:
    """The same motion metrics before and after the viewer's filter."""
    ordered = sorted(detections, key=lambda detection: detection.time_seconds)
    smoothed = _smooth(ordered, options)
    return SmoothingEffect(
        label=label,
        raw=axis_motion(ordered, label=f"{label} (raw)", track_id=track_id),
        smoothed=axis_motion(smoothed, label=f"{label} (smoothed)", track_id=track_id),
        response_delay_seconds=_response_delay(ordered, smoothed),
    )


def _smooth(ordered: Sequence[Detection], options: Smoothing) -> list[Detection]:
    """Run the detections through the filter, keeping everything else intact."""
    import numpy as np

    smoother = TrackSmoother(options)
    out: list[Detection] = []
    for time in sorted({detection.time_seconds for detection in ordered}):
        here = [d for d in ordered if d.time_seconds == time]
        roots = [
            (
                d.track_id,
                (
                    float(d.root_translation[0]),
                    float(d.root_translation[1]),
                    float(d.root_translation[2]),
                ),
            )
            for d in here
        ]
        for detection, (_, root) in zip(here, smoother.frame(time, roots), strict=True):
            out.append(
                Detection(
                    time_seconds=detection.time_seconds,
                    frame_index=detection.frame_index,
                    track_id=detection.track_id,
                    confidence=detection.confidence,
                    root_translation=np.array(root, dtype=np.float32),
                    shape=detection.shape,
                    box_xyxy=detection.box_xyxy,
                )
            )
    return out


def _response_delay(raw: Sequence[Detection], smoothed: Sequence[Detection]) -> float | None:
    """How far behind the raw signal the smoothed one runs, in seconds.

    The shift that best lines the two depth signals up, searched over whole
    frames. It is the cost of the stability the filter buys, and reporting one
    without the other would be advertising rather than measuring.

    Only defined for a single continuous track. Over interleaved tracks the two
    sequences are not one signal and the best shift is meaningless - on the
    crowd clip it came out as a confident 0.0 s across 62 people.
    """
    if len({d.track_id for d in raw}) != 1:
        return None
    depths_raw = [float(d.root_translation[2]) for d in raw]
    depths_smooth = [float(d.root_translation[2]) for d in smoothed]
    if len(depths_raw) != len(depths_smooth) or len(depths_raw) < 4:
        return None
    times = [d.time_seconds for d in raw]
    step = _median([after - before for before, after in pairwise(times) if after > before])
    if step is None or step <= 0.0:
        return None

    best_lag, best_cost = 0, None
    for lag in range(min(len(depths_raw) // 2, 30)):
        pairs = zip(depths_smooth[lag:], depths_raw[: len(depths_raw) - lag], strict=True)
        cost = sum((a - b) ** 2 for a, b in pairs)
        if best_cost is None or cost < best_cost:
            best_lag, best_cost = lag, cost
    return _round(best_lag * step)


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: Sequence[float], percentile: int) -> float | None:
    """Nearest-rank, matching the benchmark: a reported value was measured."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-percentile * len(ordered) // 100))
    return ordered[min(rank, len(ordered)) - 1]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)
