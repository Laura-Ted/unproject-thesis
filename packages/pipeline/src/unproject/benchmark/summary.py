"""Turning v2 job results into the numbers a table is made of.

Reads result files and nothing else: no inference, no clock, no filesystem
beyond the results directory. Running it twice on the same inputs produces the
same bytes, which is what lets a figure be regenerated rather than kept.

Three things this reports that the v1 summary could not:

- **throughput as measured**: steady frames over the steady phase's wall clock,
  per job, rather than the reciprocal of a median;
- **the spread**: the interquartile range, beside the median, p90 and p95 that
  v1 also reported;
- **paired differences**: for two cells that timed the same source frames, the
  difference per frame on the same repetition and position, so the scene's own
  frame-to-frame variation cancels out of the comparison.

The percentiles (p25, p75, p90, p95) are nearest-rank, so each is a value that
was measured, matching v1 and the RQ3 analysis. The median is the middle value,
or the mean of the two middle ones for an even count.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from unproject.benchmark.results import FrameSample, JobResult
from unproject.benchmark.suite import (
    PREFLIGHT_PREFIX,
    PROTOCOL_MEASURED_FRAMES,
    PROTOCOL_REPETITIONS,
    RESULT_SUFFIX,
    RUN_RECORD_FILENAME,
    Pair,
)

SUMMARY_SCHEMA_VERSION = 2

SUMMARY_FILENAME = "summary.json"
SUMMARY_CSV_FILENAME = "summary.csv"
PAIRS_CSV_FILENAME = "pairs.csv"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Distribution(_Frozen):
    """One measured quantity, summarised from its sorted samples rather than a fitted model."""

    samples: int = Field(ge=0)
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    min: float | None = None
    max: float | None = None


class CellSummary(_Frozen):
    """One cell, over every job found for it."""

    name: str
    output: str
    jobs: int = Field(ge=0)
    failed_jobs: int = Field(ge=0)
    # Whether the protocol's minimums were met. Reported, never enforced by
    # topping up a short run.
    meets_protocol: bool
    # Per job, in job order. Each is a fresh process, so each is cold.
    model_load_seconds: list[float]
    session_open_seconds: list[float]
    # The median inference per job, in job order (ADR-0006): the pooled
    # distribution below is the cell's figure, and these say how far the
    # three repetitions were from one another.
    inference_median_per_job: list[float] = Field(default_factory=list)
    # Steady state only. `total` is inference plus rest bodies plus the wire
    # message - what the socket path pays per frame before the network.
    inference: Distribution
    total: Distribution
    serialize: Distribution
    warmup_inference: Distribution
    # Frames over wall clock, per job, and the median of those.
    throughput_fps: list[float]
    throughput_fps_median: float | None = None
    people: Distribution
    message_bytes: Distribution
    rest_body_bytes_total: int = Field(ge=0)
    new_tracks_total: int = Field(ge=0)
    # Max over jobs, in bytes. Each job is its own process, so a job's peak is
    # that job's.
    peak_rss_bytes: int | None = None
    rss_after_load_bytes: int | None = None
    model_footprint_bytes: int | None = None
    metal_allocated_peak_bytes: int | None = None
    metal_driver_peak_bytes: int | None = None
    cuda_allocated_peak_bytes: int | None = None
    cuda_reserved_peak_bytes: int | None = None
    source_size: str
    model_input_size: str
    # The frames every job of this cell timed. One list, or the cell is not
    # one cell.
    measured_indices: list[int]
    consistent_frames: bool
    job_indices: list[int]
    power_sources: list[str]
    cpu_speed_limits: list[int]
    machines: list[str]
    # The declared machine ids and the devices the jobs recorded (ADR-0006).
    # A cell is one machine on one device; `one_machine` says whether it was,
    # and `meets_protocol` is false otherwise.
    machine_ids: list[str] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    device_names: list[str] = Field(default_factory=list)
    one_machine: bool = True
    # The highest one-minute load average seen before or after any job, so a
    # machine that was not idle says so in the summary.
    load_average_max: float | None = None
    git_commits: list[str]
    git_dirty: bool
    sources: list[str]
    errors: list[str]


class PairSummary(_Frozen):
    """b minus a, per frame, on the same repetition and position."""

    a: str
    b: str
    question: str
    pairs: int = Field(ge=0)
    same_frames: bool
    # Of `total` seconds per frame.
    difference: Distribution
    # How often b took longer than a. 0.5 is no effect; 1.0 is always.
    fraction_b_slower: float | None = None
    # The median of b over the median of a, for a relative reading.
    median_ratio: float | None = None


class SuiteSummary(_Frozen):
    schema_version: int = Field(default=SUMMARY_SCHEMA_VERSION, ge=1)
    suite: str
    cells: list[CellSummary]
    pairs: list[PairSummary]
    dry_runs_excluded: int = Field(default=0, ge=0)
    unreadable: list[str] = Field(default_factory=list)


def read_results(directory: Path) -> tuple[list[JobResult], list[str]]:
    """Every readable job result under `directory`, and what could not be read.

    Sorted by path so the summary does not depend on directory order.
    """
    results: list[JobResult] = []
    unreadable: list[str] = []
    for path in sorted(directory.rglob(f"*{RESULT_SUFFIX}")):
        if (
            path.name in {SUMMARY_FILENAME}
            or path.name.endswith(RUN_RECORD_FILENAME)
            or path.name.startswith(PREFLIGHT_PREFIX)
        ):
            continue
        try:
            results.append(JobResult.model_validate_json(path.read_text("utf-8")))
        except (OSError, ValueError) as error:
            unreadable.append(f"{path.relative_to(directory).as_posix()}: {error}")
    return results, unreadable


def summarise(
    suite: str,
    results: Sequence[JobResult],
    *,
    pairs: Sequence[Pair] = (),
    unreadable: Sequence[str] = (),
) -> SuiteSummary:
    """Fold job results into one summary per cell, plus the declared pairs.

    Dry runs are dropped first and counted. Failed jobs stay in their cell's
    count and error list; their steady frames, if any, are not averaged in,
    because a job that stopped early did not measure what the others did.
    """
    measured = [result for result in results if not result.dry_run]
    names = sorted({result.identity.cell for result in measured})
    by_cell = {name: [r for r in measured if r.identity.cell == name] for name in names}
    return SuiteSummary(
        suite=suite,
        cells=[_summarise_cell(name, by_cell[name]) for name in names],
        pairs=[
            _summarise_pair(pair, by_cell.get(pair.a, []), by_cell.get(pair.b, []))
            for pair in pairs
        ],
        dry_runs_excluded=len(results) - len(measured),
        unreadable=list(unreadable),
    )


def _summarise_cell(name: str, results: Sequence[JobResult]) -> CellSummary:
    ordered = sorted(results, key=lambda r: r.identity.job_index)
    complete = [r for r in ordered if r.status == "ok"]
    steady = [sample for r in complete for sample in r.steady_state()]
    warmup = [sample for r in complete for sample in r.frames if sample.phase == "warmup"]
    plans = {tuple(r.inputs.measured_indices) for r in ordered}
    first = ordered[0] if ordered else None
    throughput = [
        len(r.steady_state()) / r.phases.steady_wall_seconds
        for r in complete
        if r.phases.steady_wall_seconds > 0.0
    ]
    machine_ids = sorted({r.environment.machine_id or "undeclared" for r in ordered})
    devices = sorted({r.environment.device for r in ordered})
    one_machine = len(machine_ids) <= 1 and len(devices) <= 1
    loads = [
        load
        for r in ordered
        for load in (r.conditions.load_average_before, r.conditions.load_average_after)
        if load is not None
    ]
    return CellSummary(
        name=name,
        output=first.identity.output if first else "skeleton",
        jobs=len(ordered),
        failed_jobs=sum(1 for r in ordered if r.status == "failed"),
        meets_protocol=(
            len(complete) >= PROTOCOL_REPETITIONS
            and all(len(r.steady_state()) >= PROTOCOL_MEASURED_FRAMES for r in complete)
            and len(plans) == 1
            and one_machine
        ),
        model_load_seconds=[_round(r.phases.model_load_seconds) or 0.0 for r in ordered],
        session_open_seconds=[_round(r.phases.session_open_seconds) or 0.0 for r in ordered],
        inference_median_per_job=[
            _round(_median(s.inference_seconds for s in r.steady_state())) or 0.0 for r in complete
        ],
        inference=_distribution([s.inference_seconds for s in steady]),
        total=_distribution([s.total_seconds for s in steady]),
        serialize=_distribution([s.serialize_seconds for s in steady]),
        warmup_inference=_distribution([s.inference_seconds for s in warmup]),
        throughput_fps=[_round(value) or 0.0 for value in throughput],
        throughput_fps_median=_round(_median(throughput)),
        people=_distribution([float(s.people) for s in steady]),
        message_bytes=_distribution([float(s.message_bytes) for s in steady]),
        rest_body_bytes_total=sum(s.rest_body_bytes for s in steady),
        new_tracks_total=sum(s.new_tracks for s in steady),
        peak_rss_bytes=_max_of([r.memory.peak_rss_bytes for r in complete]),
        rss_after_load_bytes=_max_of([r.memory.rss_after_load_bytes for r in complete]),
        model_footprint_bytes=_max_of(
            [
                r.memory.rss_after_load_bytes - r.memory.rss_baseline_bytes
                for r in complete
                if r.memory.rss_after_load_bytes is not None
                and r.memory.rss_baseline_bytes is not None
            ]
        ),
        metal_allocated_peak_bytes=_max_of([r.memory.metal_allocated_peak_bytes for r in complete]),
        metal_driver_peak_bytes=_max_of([r.memory.metal_driver_peak_bytes for r in complete]),
        cuda_allocated_peak_bytes=_max_of([r.memory.cuda_allocated_peak_bytes for r in complete]),
        cuda_reserved_peak_bytes=_max_of([r.memory.cuda_reserved_peak_bytes for r in complete]),
        source_size=f"{first.geometry.source_width}x{first.geometry.source_height}"
        if first
        else "",
        model_input_size=(
            f"{first.geometry.model_input_width}x{first.geometry.model_input_height}"
            if first
            else ""
        ),
        measured_indices=list(first.inputs.measured_indices) if first else [],
        consistent_frames=len(plans) <= 1,
        job_indices=[r.identity.job_index for r in ordered],
        power_sources=sorted({r.conditions.power_source for r in ordered}),
        cpu_speed_limits=sorted(
            {
                limit
                for r in ordered
                for limit in (
                    r.conditions.cpu_speed_limit_before,
                    r.conditions.cpu_speed_limit_after,
                )
                if limit is not None
            }
        ),
        machines=sorted({r.environment.machine for r in ordered}),
        machine_ids=machine_ids,
        devices=devices,
        device_names=sorted(
            {r.environment.device_name for r in ordered if r.environment.device_name}
        ),
        one_machine=one_machine,
        load_average_max=_round(max(loads)) if loads else None,
        git_commits=sorted({r.environment.git_commit or "unknown" for r in ordered}),
        git_dirty=any(r.environment.git_dirty for r in ordered),
        sources=sorted({r.inputs.ingest_hash for r in ordered}),
        errors=[f"job {r.identity.job_index}: {r.error}" for r in ordered if r.error is not None],
    )


def _summarise_pair(pair: Pair, a: Sequence[JobResult], b: Sequence[JobResult]) -> PairSummary:
    """Differences per frame between two cells, matched on repetition and position."""
    by_key_a = _steady_by_key(a)
    by_key_b = _steady_by_key(b)
    shared = sorted(set(by_key_a) & set(by_key_b))
    differences = [by_key_b[key].total_seconds - by_key_a[key].total_seconds for key in shared]
    same_frames = all(by_key_a[key].source_index == by_key_b[key].source_index for key in shared)
    median_a = _median([by_key_a[key].total_seconds for key in shared])
    median_b = _median([by_key_b[key].total_seconds for key in shared])
    return PairSummary(
        a=pair.a,
        b=pair.b,
        question=pair.question,
        pairs=len(shared),
        same_frames=same_frames,
        difference=_distribution(differences),
        fraction_b_slower=(
            _round(sum(1 for d in differences if d > 0.0) / len(differences))
            if differences
            else None
        ),
        median_ratio=(
            _round(median_b / median_a)
            if median_a is not None and median_b is not None and median_a > 0.0
            else None
        ),
    )


def _steady_by_key(results: Sequence[JobResult]) -> dict[tuple[int, int], FrameSample]:
    keyed: dict[tuple[int, int], FrameSample] = {}
    for result in results:
        if result.status != "ok":
            continue
        for sample in result.steady_state():
            keyed[(result.identity.repetition, sample.position)] = sample
    return keyed


def _distribution(values: Iterable[float]) -> Distribution:
    ordered = sorted(values)
    if not ordered:
        return Distribution(samples=0)
    return Distribution(
        samples=len(ordered),
        median=_round(_median(ordered)),
        p25=_round(_percentile(ordered, 25)),
        p75=_round(_percentile(ordered, 75)),
        p90=_round(_percentile(ordered, 90)),
        p95=_round(_percentile(ordered, 95)),
        min=_round(ordered[0]),
        max=_round(ordered[-1]),
    )


def _median(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _percentile(ordered: Sequence[float], percentile: int) -> float:
    """Nearest-rank, on an already-sorted sequence."""
    rank = max(1, -(-percentile * len(ordered) // 100))
    return ordered[min(rank, len(ordered)) - 1]


def _round(value: float | None) -> float | None:
    """Six decimals: microseconds, which is past anything this clock resolves."""
    return None if value is None else round(value, 6)


def _max_of(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def to_csv(summary: SuiteSummary) -> str:
    """One row per cell, for eyes and thesis tooling rather than for code."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "cell",
            "output",
            "machine_id",
            "device",
            "device_name",
            "jobs",
            "failed_jobs",
            "meets_protocol",
            "samples",
            "inference_median_s",
            "inference_median_per_job_s",
            "inference_p25_s",
            "inference_p75_s",
            "inference_p90_s",
            "inference_p95_s",
            "total_median_s",
            "total_p90_s",
            "total_p95_s",
            "serialize_median_s",
            "throughput_fps_median",
            "model_load_s",
            "people_median",
            "message_bytes_median",
            "peak_rss_bytes",
            "model_footprint_bytes",
            "metal_driver_peak_bytes",
            "cuda_reserved_peak_bytes",
            "source_size",
            "model_input_size",
            "power_sources",
            "load_average_max",
            "git_commit",
            "git_dirty",
        ]
    )
    for cell in summary.cells:
        writer.writerow(
            [
                cell.name,
                cell.output,
                ";".join(cell.machine_ids),
                ";".join(cell.devices),
                ";".join(cell.device_names),
                cell.jobs,
                cell.failed_jobs,
                str(cell.meets_protocol).lower(),
                cell.inference.samples,
                _cell(cell.inference.median),
                ";".join(f"{value:.6f}" for value in cell.inference_median_per_job),
                _cell(cell.inference.p25),
                _cell(cell.inference.p75),
                _cell(cell.inference.p90),
                _cell(cell.inference.p95),
                _cell(cell.total.median),
                _cell(cell.total.p90),
                _cell(cell.total.p95),
                _cell(cell.serialize.median),
                _cell(cell.throughput_fps_median),
                ";".join(f"{value:.6f}" for value in cell.model_load_seconds),
                _cell(cell.people.median),
                _cell(cell.message_bytes.median),
                _cell(cell.peak_rss_bytes),
                _cell(cell.model_footprint_bytes),
                _cell(cell.metal_driver_peak_bytes),
                _cell(cell.cuda_reserved_peak_bytes),
                cell.source_size,
                cell.model_input_size,
                ";".join(cell.power_sources),
                _cell(cell.load_average_max),
                ";".join(commit[:12] for commit in cell.git_commits),
                str(cell.git_dirty).lower(),
            ]
        )
    return buffer.getvalue()


def pairs_to_csv(summary: SuiteSummary) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "a",
            "b",
            "question",
            "pairs",
            "same_frames",
            "difference_median_s",
            "difference_p25_s",
            "difference_p75_s",
            "fraction_b_slower",
            "median_ratio",
        ]
    )
    for pair in summary.pairs:
        writer.writerow(
            [
                pair.a,
                pair.b,
                pair.question,
                pair.pairs,
                str(pair.same_frames).lower(),
                _cell(pair.difference.median),
                _cell(pair.difference.p25),
                _cell(pair.difference.p75),
                _cell(pair.fraction_b_slower),
                _cell(pair.median_ratio),
            ]
        )
    return buffer.getvalue()


def _cell(value: float | int | None) -> str:
    """Empty rather than 0 or "None": a missing measurement is not a zero."""
    return "" if value is None else str(value)


def write_summary(summary: SuiteSummary, directory: Path) -> tuple[Path, Path, Path]:
    """Write all three forms beside the raw results they were derived from."""
    directory.mkdir(parents=True, exist_ok=True)
    as_json = directory / SUMMARY_FILENAME
    as_csv = directory / SUMMARY_CSV_FILENAME
    as_pairs = directory / PAIRS_CSV_FILENAME
    as_json.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    as_csv.write_text(to_csv(summary), encoding="utf-8")
    as_pairs.write_text(pairs_to_csv(summary), encoding="utf-8")
    return as_json, as_csv, as_pairs


__all__ = [
    "PAIRS_CSV_FILENAME",
    "SUMMARY_CSV_FILENAME",
    "SUMMARY_FILENAME",
    "CellSummary",
    "Distribution",
    "PairSummary",
    "SuiteSummary",
    "pairs_to_csv",
    "read_results",
    "summarise",
    "to_csv",
    "write_summary",
]
