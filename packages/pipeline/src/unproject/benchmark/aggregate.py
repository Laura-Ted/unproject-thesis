"""Turning raw v1 observations into the numbers a table is made of.

The pilot's summary. Like the v2 one (`summary.py`), it reads result files and
nothing else, and the same inputs give the same bytes.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from unproject.benchmark.artifacts import RESULT_SUFFIX, BenchmarkResult, FrameSample

SUMMARY_SCHEMA_VERSION = 1

SUMMARY_FILENAME = "summary.json"
SUMMARY_CSV_FILENAME = "summary.csv"

# Reported alongside the median because an interactive viewer is made of its
# tail, not its average. ADR-0002 §8.2 asks for a tail percentile.
_PERCENTILES = (90, 95)


class LatencySummary(BaseModel):
    """Per-frame timing over the steady-state samples of one benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int = Field(ge=0)
    median_seconds: float | None = None
    p90_seconds: float | None = None
    p95_seconds: float | None = None
    min_seconds: float | None = None
    max_seconds: float | None = None
    # Derived from the median, not from a total: a mean rate over a run that
    # stalled once would describe neither half of it.
    median_fps: float | None = None


class BenchmarkSummary(BaseModel):
    """One benchmark, over every raw result found for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    output: str
    latency: LatencySummary
    # Repetition 0 of each run: the only one that opens a model this process has
    # not built before. Later repetitions reuse it and are much faster.
    cold_start_seconds: list[float]
    warm_start_seconds: list[float]
    peak_memory_bytes: int | None = None
    metal_driver_bytes: int | None = None
    people_median: float | None = None
    widths: list[int]
    heights: list[int]
    # Named so a summary can be traced back to what it read.
    sources: list[str]
    errors: list[str]


class Summary(BaseModel):
    """Everything the results directory had to say."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SUMMARY_SCHEMA_VERSION, ge=1)
    benchmarks: list[BenchmarkSummary]
    # Counted rather than silently dropped, so a directory full of dry runs
    # does not read as a directory with nothing in it.
    dry_runs_excluded: int = Field(default=0, ge=0)


def read_results(directory: Path) -> tuple[list[BenchmarkResult], list[str]]:
    """Every readable result under `directory`, and what could not be read.

    Sorted by path so the summary does not depend on directory order.
    """
    results: list[BenchmarkResult] = []
    unreadable: list[str] = []
    for path in sorted(directory.rglob(f"*{RESULT_SUFFIX}")):
        if path.name in {SUMMARY_FILENAME}:
            continue
        try:
            results.append(BenchmarkResult.model_validate_json(path.read_text("utf-8")))
        except (OSError, ValueError) as error:
            # Named rather than skipped: a result that stopped parsing is a
            # change worth noticing.
            unreadable.append(f"{path.name}: {error}")
    return results, unreadable


def summarise(results: Sequence[BenchmarkResult], unreadable: Sequence[str] = ()) -> Summary:
    """Fold raw results into one summary per benchmark name.

    Dry runs are dropped first. They carry the same fields as a real result and
    a timing three orders of magnitude out, so averaging the two would produce a
    number that is wrong rather than one that is missing.
    """
    measured = [result for result in results if not result.dry_run]
    names = sorted({result.name for result in measured})
    return Summary(
        benchmarks=[
            _summarise_one(name, [result for result in measured if result.name == name], unreadable)
            for name in names
        ],
        dry_runs_excluded=len(results) - len(measured),
    )


def _summarise_one(
    name: str, results: Sequence[BenchmarkResult], unreadable: Sequence[str]
) -> BenchmarkSummary:
    samples = [sample for result in results for sample in result.steady_state()]
    errors = [result.error for result in results if result.error is not None]
    repetitions = [repetition for result in results for repetition in result.repetitions]
    return BenchmarkSummary(
        name=name,
        runs=len(results),
        failed_runs=sum(1 for result in results if result.status == "failed"),
        output=results[0].config.output if results else "skeleton",
        latency=_latency(samples),
        cold_start_seconds=sorted(
            round(rep.cold_start_seconds, 6) for rep in repetitions if rep.index == 0
        ),
        warm_start_seconds=sorted(
            round(rep.cold_start_seconds, 6) for rep in repetitions if rep.index > 0
        ),
        peak_memory_bytes=max((rep.peak_memory_bytes for rep in repetitions), default=None),
        metal_driver_bytes=max(
            (rep.metal_driver_bytes for rep in repetitions if rep.metal_driver_bytes is not None),
            default=None,
        ),
        people_median=_median([float(sample.people) for sample in samples]),
        widths=sorted({sample.width for sample in samples}),
        heights=sorted({sample.height for sample in samples}),
        sources=sorted({result.inputs.ingest_hash for result in results}),
        errors=[*errors, *unreadable],
    )


def _latency(samples: Sequence[FrameSample]) -> LatencySummary:
    seconds = sorted(sample.seconds for sample in samples)
    if not seconds:
        return LatencySummary(samples=0)
    median = _median(seconds)
    return LatencySummary(
        samples=len(seconds),
        median_seconds=_round(median),
        p90_seconds=_round(_percentile(seconds, _PERCENTILES[0])),
        p95_seconds=_round(_percentile(seconds, _PERCENTILES[1])),
        min_seconds=_round(seconds[0]),
        max_seconds=_round(seconds[-1]),
        median_fps=_round(1.0 / median) if median is not None and median > 0.0 else None,
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
    """Nearest-rank, on an already-sorted sequence.

    Chosen over an interpolating definition because it always reports a value
    that was actually measured, and because two libraries disagreeing about
    interpolation must not change a thesis number.
    """
    rank = max(1, -(-percentile * len(ordered) // 100))
    return ordered[min(rank, len(ordered)) - 1]


def _round(value: float | None) -> float | None:
    """Six decimals: microseconds, which is past anything this clock resolves."""
    return None if value is None else round(value, 6)


def to_csv(summary: Summary) -> str:
    """The same numbers, for eyes and for thesis tooling rather than for code."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "name",
            "output",
            "runs",
            "failed_runs",
            "samples",
            "median_seconds",
            "p90_seconds",
            "p95_seconds",
            "median_fps",
            "people_median",
            "peak_memory_bytes",
            "metal_driver_bytes",
        ]
    )
    for benchmark in summary.benchmarks:
        latency = benchmark.latency
        writer.writerow(
            [
                benchmark.name,
                benchmark.output,
                benchmark.runs,
                benchmark.failed_runs,
                latency.samples,
                _cell(latency.median_seconds),
                _cell(latency.p90_seconds),
                _cell(latency.p95_seconds),
                _cell(latency.median_fps),
                _cell(benchmark.people_median),
                _cell(benchmark.peak_memory_bytes),
                _cell(benchmark.metal_driver_bytes),
            ]
        )
    return buffer.getvalue()


def _cell(value: float | int | None) -> str:
    """Empty rather than 0 or "None": a missing measurement is not a zero."""
    return "" if value is None else str(value)


def write_summary(summary: Summary, directory: Path) -> tuple[Path, Path]:
    """Write both forms beside the raw results they were derived from."""
    directory.mkdir(parents=True, exist_ok=True)
    as_json = directory / SUMMARY_FILENAME
    as_csv = directory / SUMMARY_CSV_FILENAME
    as_json.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    as_csv.write_text(to_csv(summary), encoding="utf-8")
    return as_json, as_csv
