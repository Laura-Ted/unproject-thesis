"""Protocol v2 for RQ1/RQ2: a suite of cells, one process per measurement.

The v1 benchmark (`config.py`, `run.py`) measured one cell per invocation,
three repetitions in one process over the first forty frames. ADR-0005 §2.3
lists what that cannot tell apart: the first repetition was cold and the others
were not, the frames were the same forty in every cell but only the start of
the clip, the timer covered `session.frame` (and, for meshes, the rest bodies)
and nothing else, and every manifest was dirty. Those results stay readable as
the pilot they were.

A suite fixes the policy once - warm-up, measured frames, repetitions, how the
frames are picked, the seed the run order is shuffled with - and lists the
cells it applies to. Every (cell, repetition) pair is a *job*, and every job
runs in a fresh interpreter, so what one leaves behind cannot reach the next.
The order is shuffled with a recorded seed and the repetitions interleaved, so
a slow drift of the machine - thermal, background work - lands on every cell
rather than on whichever ran last.

This module plans and drives; `job.py` measures; `results.py` says what is
written; `summary.py` reads it back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.benchmark.artifacts import ENV_MACHINE_ID
from unproject.benchmark.config import OutputMode
from unproject.bodies.device import ENV_DEVICE, DeviceRequest
from unproject.config import ConfigError, ExperimentConfig, load_config

logger = logging.getLogger(__name__)

SUITE_SCHEMA_VERSION = 2

# What the protocol requires of a cell before it counts as a final result.
PROTOCOL_MEASURED_FRAMES = 60
PROTOCOL_REPETITIONS = 3

# Where v2 results land, apart from the v1 pilot's `benchmark/`.
STAGE_NAME = "benchmark-v2"
RESULT_SUFFIX = ".json"
RUN_RECORD_FILENAME = "suite-run.json"
# The preflight report written beside the results (ADR-0006), which the
# summary must step over: it is not a job.
PREFLIGHT_PREFIX = "preflight-"

# The only selection policy so far. Spread rather than prefix: a clip's first
# second is one moment of it, and a cost that depends on how many people are in
# frame should be sampled across the whole of it.
type FrameSelection = Literal["spread"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Cell(_Frozen):
    """One cell of the matrix: a scene and an output mode."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    # The experiment being measured, relative to the suite file.
    experiment: Path
    output: OutputMode = "skeleton"


class Pair(_Frozen):
    """Two cells whose frames are compared one against one.

    The difference is taken per frame, on the same position of the same
    repetition, so the scene's own variation between frames cancels out of the
    comparison. `question` is prose naming what the pair isolates.
    """

    a: str
    b: str
    question: str = Field(min_length=1)


class SuiteConfig(_Frozen):
    """The measurement policy, and the cells it is applied to."""

    schema_version: int = Field(default=SUITE_SCHEMA_VERSION, ge=1)
    # Also the results directory, hence the character restriction.
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    description: str = ""
    # Which protocol document this suite implements.
    protocol: Literal["rq1rq2-v2"] = "rq1rq2-v2"

    warmup_frames: int = Field(default=5, ge=0)
    # The protocol asks for at least sixty measured frames and three
    # repetitions (`PROTOCOL_MEASURED_FRAMES`, `PROTOCOL_REPETITIONS`). Smaller
    # suites are allowed - a smoke test on the fixture clip is one - and the
    # summary marks any cell that falls short rather than the schema refusing
    # it, so a short run is reported as short, never quietly topped up.
    measured_frames: int = Field(default=60, ge=1)
    repetitions: int = Field(default=3, ge=1)
    frame_selection: FrameSelection = "spread"
    # Shuffles the job order. Recorded in every result.
    order_seed: int = 0
    # Which device every job of the suite runs the model on (ADR-0006). `auto`
    # is the estimator's own order, CUDA then Metal then CPU; a named kind is
    # insisted on, and a job that cannot get it fails rather than substitutes.
    # A CPU suite is therefore a suite file that says so, not a shell variable
    # someone remembered to set.
    device: DeviceRequest = "auto"

    cells: list[Cell] = Field(min_length=1)
    pairs: list[Pair] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != SUITE_SCHEMA_VERSION:
            msg = (
                f"benchmark suite schema_version {self.schema_version} is not supported "
                f"by this build (expected {SUITE_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_cells_and_pairs(self) -> Self:
        names = [cell.name for cell in self.cells]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"cells named more than once: {duplicates}"
            raise ValueError(msg)
        for pair in self.pairs:
            for side in (pair.a, pair.b):
                if side not in names:
                    msg = f"pair {pair.a!r}/{pair.b!r} names a cell that is not in the suite"
                    raise ValueError(msg)
            if pair.a == pair.b:
                msg = f"a pair cannot compare {pair.a!r} with itself"
                raise ValueError(msg)
        return self

    def cell(self, name: str) -> Cell:
        for cell in self.cells:
            if cell.name == name:
                return cell
        raise ConfigError(f"suite {self.name!r} has no cell {name!r}")


def load_suite_config(path: Path) -> SuiteConfig:
    """Parse and validate a suite TOML."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read benchmark suite {path}: {error}") from error
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error
    try:
        return SuiteConfig.model_validate(data)
    except ValueError as error:
        raise ConfigError(f"{path} failed validation:\n{error}") from error


def suite_hash(config: SuiteConfig) -> str:
    """Content hash of the policy and the cells; paths included, since a cell
    that measures a different file is a different cell."""
    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


def load_cell_experiment(
    config: SuiteConfig, cell: Cell, suite_dir: Path
) -> tuple[ExperimentConfig, Path]:
    """The experiment a cell measures, and the directory its own paths resolve
    against."""
    path = cell.experiment
    resolved = path if path.is_absolute() else (suite_dir / path).resolve()
    return load_config(resolved), resolved.parent


def select_frames(
    total: int, *, warmup: int, measured: int, selection: FrameSelection = "spread"
) -> tuple[list[int], list[int]]:
    """Which ingest frames a job times, and in what order.

    Warm-up takes the first `warmup` frames of the clip. The measured frames are
    spread evenly over what follows, first to last, so every repetition and
    every cell of a suite sees the same moments of the clip in the same order -
    monotone, because the tracker expects frames in the order they were shot.
    """
    if total < warmup + measured:
        raise ConfigError(
            f"the clip has {total} frame(s), but the suite asks for {warmup} warm-up "
            f"plus {measured} measured"
        )
    if selection != "spread":  # pragma: no cover - the type admits one value
        raise ConfigError(f"unknown frame selection {selection!r}")
    first, last = warmup, total - 1
    if measured == 1:
        chosen = [first]
    else:
        step = (last - first) / (measured - 1)
        chosen = [round(first + index * step) for index in range(measured)]
    if len(set(chosen)) != len(chosen) or chosen != sorted(chosen):  # pragma: no cover
        raise ConfigError("frame selection produced a non-monotone list; this is a bug")
    return list(range(warmup)), chosen


@dataclass(frozen=True, slots=True)
class Job:
    """One cell measured once. Its index is its place in the shuffled order."""

    index: int
    cell: str
    repetition: int


def plan_jobs(config: SuiteConfig) -> list[Job]:
    """Every (cell, repetition), shuffled with the suite's seed.

    Deterministic: the same suite file plans the same order, which is what
    lets the order be recorded in the results and reproduced.
    """
    pairs = [
        (cell.name, repetition) for repetition in range(config.repetitions) for cell in config.cells
    ]
    random.Random(config.order_seed).shuffle(pairs)
    return [Job(index, cell, repetition) for index, (cell, repetition) in enumerate(pairs)]


def results_dir(artifacts_root: Path, suite: str) -> Path:
    return artifacts_root / STAGE_NAME / suite


def result_path(artifacts_root: Path, suite: str, job: Job, started: datetime) -> Path:
    """Where a job writes. Named by when it started, so a repeat adds a file."""
    stamp = started.strftime("%Y%m%dT%H%M%S%f")
    return (
        results_dir(artifacts_root, suite)
        / job.cell
        / f"{stamp}-rep{job.repetition}-job{job.index:03d}{RESULT_SUFFIX}"
    )


class JobOutcome(_Frozen):
    """What the driver saw of one job's process."""

    index: int
    cell: str
    repetition: int
    started_at: datetime
    finished_at: datetime
    exit_code: int
    result: str | None = None


class SuiteRunRecord(_Frozen):
    """The driver's own record: the order, the seed, and how each job ended.

    Written after every job so a run interrupted halfway still says what ran.
    A job whose process died without writing a result is visible here as an
    exit code with no file, which is the exclusion the protocol requires to
    stay visible.
    """

    schema_version: int = SUITE_SCHEMA_VERSION
    suite: str
    suite_hash: str
    order_seed: int
    dry_run: bool
    started_at: datetime
    finished_at: datetime | None = None
    python: str
    command: list[str]
    # The declared machine and the device the suite asked for (ADR-0006);
    # absent in records written before it.
    machine_id: str | None = None
    device: str | None = None
    jobs: list[JobOutcome] = Field(default_factory=list)


def run_suite(
    config: SuiteConfig,
    *,
    suite_path: Path,
    artifacts_root: Path,
    dry_run: bool = False,
    only: Sequence[str] | None = None,
    machine_id: str | None = None,
) -> SuiteRunRecord:
    """Run every job of a suite, each in a fresh interpreter, in the planned order.

    The driver never imports torch. It resolves the plan, spawns
    `unproject benchmark-job` per job through the same interpreter it runs in,
    and records each exit. A job that fails does not stop the suite: the
    protocol's exclusion rules want the failure recorded and the rest measured.

    `machine_id` is the operator's name for this machine; it reaches every job
    through the environment and is written into every result.
    """
    jobs = plan_jobs(config)
    child_env = job_environment(config, machine_id=machine_id)
    if only is not None:
        unknown = sorted(set(only) - {cell.name for cell in config.cells})
        if unknown:
            raise ConfigError(f"suite {config.name!r} has no cell(s) {unknown}")
        jobs = [job for job in jobs if job.cell in set(only)]

    directory = results_dir(artifacts_root, config.name)
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    base_command = [sys.executable, "-m", "unproject.cli", "benchmark-job", str(suite_path)]
    record = SuiteRunRecord(
        suite=config.name,
        suite_hash=suite_hash(config),
        order_seed=config.order_seed,
        dry_run=dry_run,
        started_at=started,
        python=sys.executable,
        command=base_command,
        machine_id=machine_id,
        device=config.device,
    )
    record_path = directory / f"{started.strftime('%Y%m%dT%H%M%S%f')}-{RUN_RECORD_FILENAME}"

    for job in jobs:
        command = [
            *base_command,
            "--cell",
            job.cell,
            "--repetition",
            str(job.repetition),
            "--job-index",
            str(job.index),
            "--job-count",
            str(len(plan_jobs(config))),
            "--artifacts-root",
            str(artifacts_root),
        ]
        if dry_run:
            command.append("--dry-run")
        logger.info(
            "job %d/%d: %s repetition %d",
            job.index + 1,
            len(plan_jobs(config)),
            job.cell,
            job.repetition,
        )
        job_started = datetime.now(UTC)
        completed = subprocess.run(command, check=False, env=child_env)
        written = _newest_result(directory / job.cell, since=job_started)
        record = record.model_copy(
            update={
                "jobs": [
                    *record.jobs,
                    JobOutcome(
                        index=job.index,
                        cell=job.cell,
                        repetition=job.repetition,
                        started_at=job_started,
                        finished_at=datetime.now(UTC),
                        exit_code=completed.returncode,
                        result=(
                            None if written is None else written.relative_to(directory).as_posix()
                        ),
                    ),
                ]
            }
        )
        _write_record(record, record_path)
        if completed.returncode != 0:
            logger.error(
                "job %d (%s, repetition %d) exited %d; recorded and continuing",
                job.index,
                job.cell,
                job.repetition,
                completed.returncode,
            )

    record = record.model_copy(update={"finished_at": datetime.now(UTC)})
    _write_record(record, record_path)
    return record


def job_environment(config: SuiteConfig, *, machine_id: str | None) -> dict[str, str]:
    """The environment a job process inherits: the caller's, plus the suite's
    device and the machine id, so a job cannot run on a device the suite did
    not ask for. A variable already set to something else is a conflict."""
    env = dict(os.environ)
    if config.device != "auto":
        existing = env.get(ENV_DEVICE, "").strip().lower()
        if existing and existing != "auto" and existing != config.device:
            raise ConfigError(
                f"suite {config.name!r} asks for device {config.device!r} but the environment "
                f"sets {ENV_DEVICE}={existing!r}; unset it or change the suite"
            )
        env[ENV_DEVICE] = config.device
    if machine_id is not None:
        env[ENV_MACHINE_ID] = machine_id
    return env


def _newest_result(directory: Path, *, since: datetime) -> Path | None:
    """The result a job just wrote, if its process got that far."""
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob(f"*{RESULT_SUFFIX}"))
    for path in reversed(candidates):
        try:
            data = json.loads(path.read_text("utf-8"))
            started = datetime.fromisoformat(str(data["started_at"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if started >= since:
            return path
    return None


def _write_record(record: SuiteRunRecord, path: Path) -> None:
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


__all__ = [
    "PREFLIGHT_PREFIX",
    "PROTOCOL_MEASURED_FRAMES",
    "PROTOCOL_REPETITIONS",
    "RESULT_SUFFIX",
    "RUN_RECORD_FILENAME",
    "STAGE_NAME",
    "SUITE_SCHEMA_VERSION",
    "Cell",
    "Job",
    "JobOutcome",
    "Pair",
    "SuiteConfig",
    "SuiteRunRecord",
    "job_environment",
    "load_cell_experiment",
    "load_suite_config",
    "plan_jobs",
    "result_path",
    "results_dir",
    "run_suite",
    "select_frames",
    "suite_hash",
]
