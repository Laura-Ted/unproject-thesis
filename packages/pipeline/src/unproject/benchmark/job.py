"""Measuring one job of a suite, in the process that called this.

What is timed, and what is not. The frames are decoded from the ingest
artifact into memory *before* the clock starts, so the estimator is measured on
pixels that are already there - the live path hands it arrays, and a disk read
is not its cost. Then, per frame and on a monotonic clock:

- **inference**: `session.frame` on the array - upstream's preprocessing, the
  forward pass, the tracker, and the conversion into the project's types;
- **rest bodies**: in mesh mode, the unposed body of any person seen for the
  first time, which is what the socket sends once per track;
- **serialisation**: the wire message, built by `unproject.bodies.wire`, the
  same code the socket runs, and encoded to bytes so its size is known.

Model load and the opening of a session are timed apart from the frames, once
each, because this process is fresh: there is no "warm start" to confuse with
a cold one. Ingest, the network and the browser are outside; WP3 has them.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from unproject.benchmark.config import OutputMode
from unproject.benchmark.results import (
    AcceleratorReading,
    Conditions,
    FrameSample,
    InputGeometry,
    JobIdentity,
    JobInputs,
    JobResult,
    Memory,
    Phase,
    Phases,
)
from unproject.benchmark.run import (
    BenchmarkError,
    MeasurableEstimator,
    MeasurableSession,
    environment,
)
from unproject.benchmark.suite import (
    Job,
    SuiteConfig,
    load_cell_experiment,
    result_path,
    select_frames,
    suite_hash,
)
from unproject.bodies.device import (
    ENV_DEVICE,
    DeviceInfo,
    accelerator_memory,
    accelerator_peak_memory,
    describe_device,
    synchronize,
)
from unproject.bodies.multihmr import FrameEstimate, model_input_size
from unproject.bodies.wire import frame_message
from unproject.config import BodiesConfig, config_hash
from unproject.ingest.stage import run_ingest
from unproject.run.manifest import peak_memory_bytes

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

RgbFrame = NDArray[np.uint8]

_HASH_PREFIX_LENGTH = 12


def measure_job(
    config: SuiteConfig,
    job: Job,
    *,
    suite_dir: Path,
    artifacts_root: Path,
    job_count: int,
    estimator: MeasurableEstimator | None = None,
    dry_run: bool = False,
) -> tuple[JobResult, Path]:
    """Measure one cell once and write the raw result.

    `estimator` exists so the orchestration can be exercised without the
    weights; leaving it out builds the real one, which is the point of the
    fresh process this normally runs in.
    """
    started = datetime.now(UTC)
    cell = config.cell(job.cell)
    apply_device_policy(config)
    experiment, experiment_dir = load_cell_experiment(config, cell, suite_dir)
    ingested = run_ingest(experiment, config_dir=experiment_dir, artifacts_root=artifacts_root)

    warmup_indices, measured_indices = select_frames(
        len(ingested.artifact.frames),
        warmup=config.warmup_frames,
        measured=config.measured_frames,
        selection=config.frame_selection,
    )
    plan = [*warmup_indices, *measured_indices]
    frames = _load_frames([ingested.artifact_dir / ingested.artifact.frames[i].path for i in plan])
    width, height = frames[0].shape[1], frames[0].shape[0]
    model_width, model_height, scale = model_input_size(width, height)

    conditions = read_conditions()
    rss_baseline = current_rss_bytes()

    load_started = time.perf_counter()
    if estimator is None:
        from unproject.bodies.multihmr import MultiHmrEstimator

        estimator = MultiHmrEstimator(experiment.bodies, config_dir=experiment_dir)
    # The first `open` builds the model and a tracker; the second only a
    # tracker over the loaded model, which is the cost a second session pays.
    first = estimator.open()
    device = first.device
    synchronize_device(device)
    model_load = time.perf_counter() - load_started
    open_started = time.perf_counter()
    session = estimator.open()
    synchronize_device(device)
    session_open = time.perf_counter() - open_started
    rss_after_load = current_rss_bytes()
    described: DeviceInfo | None = None if device is None else describe_device(device)
    metal_allocated_baseline, metal_driver_baseline = _accelerator_memory(device)

    samples: list[FrameSample] = []
    failure: str | None = None
    steady_started: float | None = None
    steady_finished: float | None = None
    metal_allocated_peak = metal_allocated_baseline
    metal_driver_peak = metal_driver_baseline
    seen_tracks: set[int] = set()
    try:
        for position, (source_index, image) in enumerate(zip(plan, frames, strict=True)):
            phase: Phase = "warmup" if position < len(warmup_indices) else "steady"
            if phase == "steady" and steady_started is None:
                steady_started = time.perf_counter()
            sample = _one_frame(
                session,
                experiment.bodies,
                image,
                phase=phase,
                position=position,
                source_index=source_index,
                output=cell.output,
                seen_tracks=seen_tracks,
            )
            samples.append(sample)
            if phase == "steady":
                steady_finished = time.perf_counter()
            allocated, driver = _accelerator_memory(device)
            metal_allocated_peak = _max_optional(metal_allocated_peak, allocated)
            metal_driver_peak = _max_optional(metal_driver_peak, driver)
    except Exception as error:
        # A failed job is recorded with its error, so a partial measurement is
        # never read as a complete one.
        failure = f"{type(error).__name__}: {error}"
        logger.exception("the job stopped early")

    steady_wall = (
        steady_finished - steady_started
        if steady_started is not None and steady_finished is not None
        else 0.0
    )
    cuda_allocated_peak, cuda_reserved_peak = _accelerator_peak_memory(device)
    # On CUDA the counters sampled per frame are CUDA's, and the fields named
    # for Metal would misreport them; CUDA's own peaks are recorded instead.
    if described is not None and described.kind == "cuda":
        metal_allocated_baseline = metal_driver_baseline = None
        metal_allocated_peak = metal_driver_peak = None
    # A second reading, now that the job is done. `read_conditions` always fills
    # the `_before` fields, so below they are copied into the `_after` ones.
    after = read_conditions()
    result = JobResult(
        identity=JobIdentity(
            suite=config.name,
            suite_hash=suite_hash(config),
            cell=cell.name,
            output=cell.output,
            repetition=job.repetition,
            job_index=job.index,
            job_count=job_count,
            order_seed=config.order_seed,
        ),
        started_at=started,
        finished_at=datetime.now(UTC),
        status="failed" if failure is not None else "ok",
        error=failure,
        dry_run=dry_run,
        inputs=JobInputs(
            experiment=experiment.name,
            source_sha256=ingested.artifact.source.content_sha256,
            ingest_hash=ingested.artifact_dir.name,
            bodies_config_hash=config_hash(experiment.bodies)[:_HASH_PREFIX_LENGTH],
            checkpoint_sha256=estimator.describe().checkpoint_sha256,
            backbone_ref=estimator.describe().backbone_ref,
            provenance=experiment.provenance,
            warmup_indices=warmup_indices,
            measured_indices=measured_indices,
        ),
        environment=environment(described),
        conditions=conditions.model_copy(
            update={
                "cpu_speed_limit_after": after.cpu_speed_limit_before,
                "load_average_after": after.load_average_before,
                "gpu_after": after.gpu_before,
            }
        ),
        geometry=InputGeometry(
            source_width=width,
            source_height=height,
            model_input_width=model_width,
            model_input_height=model_height,
            scale=scale,
        ),
        phases=Phases(
            model_load_seconds=model_load,
            session_open_seconds=session_open,
            steady_wall_seconds=steady_wall,
        ),
        memory=Memory(
            rss_baseline_bytes=rss_baseline,
            rss_after_load_bytes=rss_after_load,
            rss_end_bytes=current_rss_bytes(),
            peak_rss_bytes=peak_memory_bytes(),
            metal_allocated_baseline_bytes=metal_allocated_baseline,
            metal_driver_baseline_bytes=metal_driver_baseline,
            metal_allocated_peak_bytes=metal_allocated_peak,
            metal_driver_peak_bytes=metal_driver_peak,
            cuda_allocated_peak_bytes=cuda_allocated_peak,
            cuda_reserved_peak_bytes=cuda_reserved_peak,
        ),
        frames=samples,
    )
    path = result_path(artifacts_root, config.name, job, started)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result, path


def _one_frame(
    session: MeasurableSession,
    bodies: BodiesConfig,
    image: RgbFrame,
    *,
    phase: Phase,
    position: int,
    source_index: int,
    output: OutputMode,
    seen_tracks: set[int],
) -> FrameSample:
    height, width = image.shape[:2]

    device = session.device
    began = time.perf_counter()
    estimate = session.frame(image, bodies, width=width, height=height)
    # `frame` already pulls every output to the host, which waits for the
    # device; this makes the wait explicit so the clock's definition does not
    # depend on that detail of the conversion.
    synchronize_device(device)
    inferred = time.perf_counter()

    rest_bytes = 0
    new_tracks = 0
    if output == "mesh":
        rest_bytes, new_tracks = _rest_bodies(session, estimate, seen_tracks)
    rested = time.perf_counter()

    message = frame_message(
        estimate,
        time_seconds=float(position),
        size=(width, height),
        dropped=0,
        bodies=output == "mesh",
    ).model_dump_json()
    serialised = time.perf_counter()

    allocated, _ = _accelerator_memory(device)
    return FrameSample(
        phase=phase,
        position=position,
        source_index=source_index,
        inference_seconds=inferred - began,
        rest_body_seconds=rested - inferred,
        serialize_seconds=serialised - rested,
        people=len(estimate.people),
        new_tracks=new_tracks,
        message_bytes=len(message.encode("utf-8")),
        rest_body_bytes=rest_bytes,
        metal_allocated_bytes=allocated,
    )


def _rest_bodies(
    session: MeasurableSession, estimate: FrameEstimate, seen: set[int]
) -> tuple[int, int]:
    """What mesh output costs on top of the estimate, and only that.

    The socket sends a person's unposed body once, the first time it sees them,
    so this builds it once per track and counts the bytes of that message: a
    4-byte track id, the vertices and the bind bones, as float32. It counts
    every model vertex; the socket (`unproject_api.live.rest_body`) sends only
    those a triangle uses.
    """
    total = 0
    new = 0
    for person in estimate.people:
        if person.track_id in seen:
            continue
        seen.add(person.track_id)
        new += 1
        vertices = session.rest_vertices(person.shape)
        bones = session.rest_bone_poses(person.shape)
        total += 4 + vertices.astype(np.float32).nbytes + bones.astype(np.float32).nbytes
    return total, new


def _load_frames(paths: list[Path]) -> list[RgbFrame]:
    """Every frame the job will time, decoded, in order.

    BGR to RGB, as the live path does; the model was trained on RGB and getting
    this wrong raises nothing and looks like nothing.
    """
    import cv2

    frames: list[RgbFrame] = []
    size: tuple[int, int] | None = None
    for path in paths:
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is None:
            raise BenchmarkError(f"cannot read {path}")
        rgb: RgbFrame = np.ascontiguousarray(decoded[:, :, ::-1])
        if size is None:
            size = (rgb.shape[1], rgb.shape[0])
        elif size != (rgb.shape[1], rgb.shape[0]):
            raise BenchmarkError(
                f"{path} is {rgb.shape[1]}x{rgb.shape[0]} but the first frame was "
                f"{size[0]}x{size[1]}; a cell measures one geometry"
            )
        frames.append(rgb)
    return frames


def apply_device_policy(config: SuiteConfig) -> None:
    """Hold this process to the suite's device before the estimator is built.

    The driver sets the same variable for every job it spawns; setting it here
    as well means a job run by hand is held to the suite too. A variable
    already naming a different device is a conflict, not a tie to break.
    """
    if config.device == "auto":
        return
    existing = os.environ.get(ENV_DEVICE, "").strip().lower()
    if existing and existing != "auto" and existing != config.device:
        raise BenchmarkError(
            f"suite {config.name!r} asks for device {config.device!r} but {ENV_DEVICE} is "
            f"{existing!r}; unset it or change the suite"
        )
    os.environ[ENV_DEVICE] = config.device


def synchronize_device(device: torch.device | None) -> None:
    """Drain the device's queue, if the session has one to drain."""
    if device is not None:
        synchronize(device)


def _accelerator_memory(device: torch.device | None) -> tuple[int | None, int | None]:
    return (None, None) if device is None else accelerator_memory(device)


def _accelerator_peak_memory(device: torch.device | None) -> tuple[int | None, int | None]:
    return (None, None) if device is None else accelerator_peak_memory(device)


def _max_optional(current: int | None, candidate: int | None) -> int | None:
    if candidate is None:
        return current
    return candidate if current is None else max(current, candidate)


def current_rss_bytes() -> int | None:
    """Resident set size right now, from `ps`; None where that is unavailable.

    `ru_maxrss` is a high-water mark and cannot say what the process holds at
    a moment, which is what a baseline needs.
    """
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = completed.stdout.strip()
    if completed.returncode != 0 or not text.isdigit():
        return None
    return int(text) * 1024


# What is asked of `nvidia-smi`, in the order the columns come back. The
# event-reasons name is the current one; older drivers know it as
# `clocks_throttle_reasons.active`, which is tried second.
_NVIDIA_SMI_FIELDS = (
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "{reasons}",
    "pstate",
    "utilization.gpu",
)
_NVIDIA_REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")


def read_conditions() -> Conditions:
    """Power, thermal and load conditions, where the OS says.

    macOS: `pmset` for the power source, low-power mode and the thermal CPU
    limit. Linux: the CPU governor from sysfs and the card from `nvidia-smi`.
    Both: the one-minute load average. Anything the OS does not report is
    `unknown` or absent, never guessed; the raw text is kept so the parsing
    can be checked later.
    """
    raw: dict[str, str] = {}
    if platform.system() == "Darwin":
        for key, arguments in (
            ("batt", ["pmset", "-g", "batt"]),
            ("therm", ["pmset", "-g", "therm"]),
            ("settings", ["pmset", "-g"]),
        ):
            text = _run_for_text(arguments)
            if text is not None:
                raw[key] = text
    else:
        governor = _read_text(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"))
        if governor is not None:
            raw["governor"] = governor
        smi = nvidia_smi_query()
        if smi is not None:
            raw["nvidia-smi"] = smi
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = None
    return parse_conditions(raw, load_average=load)


def nvidia_smi_query() -> str | None:
    """One CSV line per card with `_NVIDIA_SMI_FIELDS`, or None without a card."""
    for reasons in _NVIDIA_REASON_FIELDS:
        fields = ",".join(field.format(reasons=reasons) for field in _NVIDIA_SMI_FIELDS)
        text = _run_for_text(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        )
        if text is not None and text.strip():
            return text
    return None


def parse_conditions(raw: dict[str, str], *, load_average: float | None = None) -> Conditions:
    """Turn the raw readings into values. Pure, so it can be tested on text."""
    battery = raw.get("batt", "")
    power: Literal["mains", "battery", "unknown"] = "unknown"
    if "AC Power" in battery:
        power = "mains"
    elif "Battery Power" in battery:
        power = "battery"

    low_power: bool | None = None
    settings = raw.get("settings", "")
    match = re.search(r"lowpowermode\s+(\d)", settings)
    if match:
        low_power = match.group(1) == "1"

    limit: int | None = None
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", raw.get("therm", ""))
    if match:
        limit = int(match.group(1))

    governor = raw.get("governor")
    return Conditions(
        power_source=power,
        low_power_mode=low_power,
        cpu_speed_limit_before=limit,
        load_average_before=load_average,
        cpu_governor=governor.strip() if governor else None,
        gpu_before=parse_nvidia_smi(raw["nvidia-smi"]) if "nvidia-smi" in raw else None,
        raw=raw,
    )


def parse_nvidia_smi(text: str) -> AcceleratorReading | None:
    """The first card's line of `nvidia_smi_query`, as numbers where they are
    numbers. A numeric field the driver reports as `[N/A]` is absent."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    cells = [cell.strip() for cell in lines[0].split(",")]
    if len(cells) != len(_NVIDIA_SMI_FIELDS):
        return None
    temperature, power_draw, sm_clock, memory_clock, reasons, pstate, utilization = cells
    return AcceleratorReading(
        temperature_c=_number(temperature),
        power_draw_w=_number(power_draw),
        sm_clock_mhz=_number(sm_clock),
        memory_clock_mhz=_number(memory_clock),
        throttle_reasons=reasons or None,
        performance_state=pstate or None,
        utilization_percent=_number(utilization),
    )


def _number(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def _run_for_text(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments, capture_output=True, check=False, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


__all__ = [
    "apply_device_policy",
    "current_rss_bytes",
    "measure_job",
    "nvidia_smi_query",
    "parse_conditions",
    "parse_nvidia_smi",
    "read_conditions",
    "synchronize_device",
]
