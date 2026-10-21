"""Driving the estimator with a stopwatch on it.

Nothing here re-implements inference: it opens the same estimator the bodies
stage opens and calls the same per-frame method, so a number measured here is a
number the pipeline would actually pay. What this adds is when the clock starts
and stops, and what is written down.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from unproject.benchmark.artifacts import (
    ENV_MACHINE_ID,
    RESULT_SUFFIX,
    BenchmarkResult,
    Environment,
    FrameSample,
    Inputs,
    Repetition,
)
from unproject.benchmark.config import (
    BenchmarkConfig,
    OutputMode,
    load_measured_experiment,
)
from unproject.bodies.artifacts import EstimatorInfo
from unproject.bodies.device import (
    DeviceError,
    DeviceInfo,
    accelerator_memory,
    chip_name,
    describe_device,
    resolve_kind,
)
from unproject.bodies.multihmr import FloatArray, FrameEstimate, FrameInput
from unproject.config import BodiesConfig, ExperimentConfig, config_hash
from unproject.ingest.stage import run_ingest
from unproject.run.manifest import git_commit, git_dirty, package_versions, peak_memory_bytes

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

STAGE_NAME = "benchmark"

_HASH_PREFIX_LENGTH = 12


class BenchmarkError(Exception):
    """Raised when a measurement cannot be taken at all."""


class MeasurableSession(Protocol):
    """The slice of `EstimatorSession` a measurement drives."""

    @property
    def device(self) -> torch.device | None:
        """Where the model lives, so a clock can wait for it; None for a
        stand-in that has no device to wait for."""
        ...

    def frame(
        self, image: FrameInput, config: BodiesConfig, *, width: int, height: int
    ) -> FrameEstimate: ...

    def rest_vertices(self, shape: FloatArray) -> FloatArray: ...

    def rest_bone_poses(self, shape: FloatArray) -> FloatArray: ...


class MeasurableEstimator(Protocol):
    """The slice of `MultiHmrEstimator` a measurement drives."""

    def open(self) -> MeasurableSession: ...

    def describe(self) -> EstimatorInfo: ...


def run_benchmark(
    config: BenchmarkConfig,
    *,
    config_dir: Path,
    artifacts_root: Path,
    estimator: MeasurableEstimator | None = None,
    dry_run: bool = False,
) -> tuple[BenchmarkResult, Path]:
    """Measure one benchmark config and write the raw result beside the others.

    `estimator` exists so the orchestration can be exercised without the
    weights; leaving it out builds the real one. `dry_run` marks the result as
    measuring nothing, and is the caller's to declare: a stand-in estimator and
    a real one are the same shape from in here.
    """
    experiment, experiment_dir = load_measured_experiment(config, config_dir)
    ingested = run_ingest(experiment, config_dir=experiment_dir, artifacts_root=artifacts_root)

    wanted = config.warmup_frames + config.measured_frames
    records = ingested.artifact.frames[:wanted]
    if len(records) < wanted:
        raise BenchmarkError(
            f"{experiment.name} has {len(records)} frame(s), but this benchmark asks for "
            f"{config.warmup_frames} warm-up plus {config.measured_frames} measured"
        )
    paths = [ingested.artifact_dir / record.path for record in records]

    if estimator is None:
        from unproject.bodies.multihmr import MultiHmrEstimator

        estimator = MultiHmrEstimator(experiment.bodies, config_dir=experiment_dir)

    started_at = datetime.now(UTC)
    repetitions: list[Repetition] = []
    frames: list[FrameSample] = []
    failure: str | None = None
    try:
        for index in range(config.repetitions):
            done, sampled = _one_repetition(index, estimator, experiment.bodies, paths, config)
            repetitions.append(done)
            frames.extend(sampled)
    except Exception as error:
        # Recorded as a failed run rather than raised; see `job.measure_job`.
        failure = f"{type(error).__name__}: {error}"
        logger.exception("the benchmark stopped early")

    result = BenchmarkResult(
        name=config.name,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        status="failed" if failure is not None else "ok",
        error=failure,
        dry_run=dry_run,
        config=config,
        environment=environment(),
        inputs=Inputs(
            experiment=experiment.name,
            source_sha256=ingested.artifact.source.content_sha256,
            ingest_hash=ingested.artifact_dir.name,
            benchmark_config_hash=config_hash(config)[:_HASH_PREFIX_LENGTH],
            bodies_config_hash=config_hash(experiment.bodies)[:_HASH_PREFIX_LENGTH],
            checkpoint_sha256=estimator.describe().checkpoint_sha256,
            backbone_ref=estimator.describe().backbone_ref,
        ),
        repetitions=repetitions,
        frames=frames,
    )
    return result, write_result(result, artifacts_root)


def _one_repetition(
    index: int,
    estimator: MeasurableEstimator,
    bodies: BodiesConfig,
    paths: Sequence[Path],
    config: BenchmarkConfig,
) -> tuple[Repetition, list[FrameSample]]:
    """One pass over the frames, opened fresh so the tracker starts empty."""
    opened_at = time.perf_counter()
    session = estimator.open()
    cold_start = time.perf_counter() - opened_at

    samples: list[FrameSample] = []
    seen_tracks: set[int] = set()
    for position, path in enumerate(paths):
        width, height = _frame_size(path)
        began = time.perf_counter()
        estimate = session.frame(path, bodies, width=width, height=height)
        if config.output == "mesh":
            _build_new_bodies(session, estimate, seen_tracks)
        elapsed = time.perf_counter() - began

        samples.append(
            FrameSample(
                repetition=index,
                frame_index=position,
                warmup=position < config.warmup_frames,
                seconds=elapsed,
                people=len(estimate.people),
                width=width,
                height=height,
            )
        )

    allocated, driver = metal_memory()
    return (
        Repetition(
            index=index,
            cold_start_seconds=cold_start,
            peak_memory_bytes=peak_memory_bytes(),
            metal_allocated_bytes=allocated,
            metal_driver_bytes=driver,
        ),
        samples,
    )


def _build_new_bodies(session: MeasurableSession, estimate: FrameEstimate, seen: set[int]) -> None:
    """What mesh output costs on top of the estimate, and only that.

    Each track's rest body is built once, as the socket sends it; see
    `job._rest_bodies`.
    """
    for person in estimate.people:
        if person.track_id in seen:
            continue
        seen.add(person.track_id)
        session.rest_vertices(person.shape)
        session.rest_bone_poses(person.shape)


def _frame_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise BenchmarkError(f"cannot read {path}")
    return int(image.shape[1]), int(image.shape[0])


def environment(device: DeviceInfo | None = None) -> Environment:
    """The machine and the code, as every result records them.

    `device` is what a measurement found when it built the estimator; left
    out, it is resolved the way the estimator would resolve it, so the record
    and the run agree.
    """
    import os
    import platform

    described = _describe_device() if device is None else device
    return Environment(
        machine=chip_name(),
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        python_version=platform.python_version(),
        device="unavailable" if described is None else described.kind,
        package_versions=package_versions(),
        git_commit=git_commit(),
        git_dirty=git_dirty(),
        machine_id=os.environ.get(ENV_MACHINE_ID) or None,
        device_name=None if described is None else described.name,
        cuda_version=None if described is None else described.cuda_version,
        cudnn_version=None if described is None else described.cudnn_version,
        driver_version=None if described is None else described.driver_version,
        os_version=os_version(),
        cpu_count=os.cpu_count(),
        memory_total_bytes=memory_total_bytes(),
        torch_threads=None if described is None else described.threads,
    )


def _describe_device() -> DeviceInfo | None:
    """The device the estimator would pick right now; None without torch, or
    when the one asked for does not exist here. Never a substitute."""
    try:
        import torch
    except ImportError:
        return None
    try:
        kind = resolve_kind()
    except DeviceError as error:
        logger.warning("%s", error)
        return None
    return describe_device(torch.device(kind))


def os_version() -> str | None:
    """The OS release as its vendor names it; the kernel is in `platform`."""
    import platform

    if platform.system() == "Darwin":
        version = _run_for_text(["sw_vers", "-productVersion"])
        build = _run_for_text(["sw_vers", "-buildVersion"])
        if version is None:
            return None
        text = f"macOS {version.strip()}"
        return f"{text} ({build.strip()})" if build else text
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        return None
    return None


def memory_total_bytes() -> int | None:
    """Physical memory: `hw.memsize` on macOS, `MemTotal` on Linux."""
    import platform

    if platform.system() == "Darwin":
        text = _run_for_text(["sysctl", "-n", "hw.memsize"])
        return int(text.strip()) if text is not None and text.strip().isdigit() else None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = line.split()[1]
                return int(kib) * 1024 if kib.isdigit() else None
    except OSError:
        return None
    return None


def _run_for_text(arguments: list[str]) -> str | None:
    """Stdout of a read-only query, or None when it fails or is missing."""
    import subprocess

    try:
        completed = subprocess.run(
            arguments, capture_output=True, check=False, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def metal_memory() -> tuple[int | None, int | None]:
    """Metal's own counters, when the installed PyTorch exposes them.

    Kept under its v1 name for the v1 benchmark; the v2 job reads the device
    it actually built through `bodies.device.accelerator_memory`.
    """
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.backends.mps.is_available():
        return None, None
    return accelerator_memory(torch.device("mps"))


def results_dir(artifacts_root: Path, name: str) -> Path:
    """Where raw results for one benchmark accumulate."""
    return artifacts_root / STAGE_NAME / name


def write_result(result: BenchmarkResult, artifacts_root: Path) -> Path:
    """Write one raw observation. Never replaces an existing one.

    Named by when it started and what it measured, so repeating a benchmark
    adds evidence instead of overwriting it.
    """
    directory = results_dir(artifacts_root, result.name)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = result.started_at.strftime("%Y%m%dT%H%M%S%f")
    path = directory / f"{stamp}-{result.inputs.benchmark_config_hash}{RESULT_SUFFIX}"
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def describe_matrix(config: BenchmarkConfig, experiment: ExperimentConfig) -> str:
    """One line naming the cell of the RQ2 matrix this run occupies."""
    return f"{config.name}: {experiment.name}, {config.output}, {config.repetitions} repetition(s)"


__all__ = [
    "BenchmarkError",
    "MeasurableEstimator",
    "MeasurableSession",
    "OutputMode",
    "describe_matrix",
    "environment",
    "memory_total_bytes",
    "metal_memory",
    "os_version",
    "results_dir",
    "run_benchmark",
    "write_result",
]
