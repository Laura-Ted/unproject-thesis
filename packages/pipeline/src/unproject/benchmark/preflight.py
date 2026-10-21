"""What has to be true of a machine before a suite is worth an hour of it.

ADR-0006 §4. A campaign job records its conditions, but it records them after
the fact; this looks before, and refuses to start a run that the protocol
would have to report as flawed: a dirty tree, an undeclared machine, a device
other than the one the suite asks for, footage that is not the campaign's
bytes, a full disk, another process on the accelerator. What it cannot judge -
a load average, a warm card - it reports and leaves to the operator.

Read-only apart from the report it writes beside the results and, with
`fetch_weights`, the pinned checkpoint if absent. It never deletes, moves or
edits anything; on a machine that holds other people's data that is the
property that matters most.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import shutil
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from unproject.benchmark.artifacts import Environment
from unproject.benchmark.campaign import MachineRegistry
from unproject.benchmark.job import read_conditions
from unproject.benchmark.results import Conditions
from unproject.benchmark.run import environment
from unproject.benchmark.suite import (
    PREFLIGHT_PREFIX,
    RESULT_SUFFIX,
    SuiteConfig,
    load_cell_experiment,
    results_dir,
    suite_hash,
)
from unproject.bodies.device import (
    ENV_DEVICE,
    DeviceError,
    DeviceKind,
    describe_device,
    resolve_kind,
)
from unproject.config import ConfigError, ExperimentConfig
from unproject.ingest.geometry import GeometryError, output_size
from unproject.ingest.source import FrameSourceError
from unproject.ingest.stage import open_source

logger = logging.getLogger(__name__)

PREFLIGHT_SCHEMA_VERSION = 1

# Below this the ingest of one clip plus the results would not fit; below the
# second, a second clip would not.
FAIL_FREE_GB = 10.0
WARN_FREE_GB = 25.0
# A one-minute load average above this on an idle machine means something
# else is running; it is reported, not judged, because a Mac at rest sits
# near 2 with its own services.
WARN_LOAD_AVERAGE = 3.0
# A process using more than this share of one core, other than this one.
WARN_PROCESS_CPU_PERCENT = 20.0

type Status = Literal["ok", "warn", "fail"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Check(_Frozen):
    name: str
    status: Status
    detail: str


class PreflightReport(_Frozen):
    schema_version: int = Field(default=PREFLIGHT_SCHEMA_VERSION, ge=1)
    suite: str
    suite_hash: str
    machine_id: str | None
    device_requested: str
    started_at: datetime
    checks: list[Check]
    environment: Environment
    conditions: Conditions
    # True when no check failed. Warnings are the operator's to weigh.
    ok: bool


def run_preflight(
    config: SuiteConfig,
    *,
    suite_dir: Path,
    artifacts_root: Path,
    machine_id: str | None,
    registry: MachineRegistry | None,
    fetch_weights: bool = False,
) -> PreflightReport:
    """Every check, in the order a reader would want them; nothing stops early."""
    checks: list[Check] = []
    checks.extend(_check_git())
    checks.extend(_check_machine(machine_id, registry))
    kind = _check_device(config, machine_id, registry, checks)
    checks.extend(_check_estimator())
    experiments = _cell_experiments(config, suite_dir, checks)
    checks.extend(_check_weights(experiments, fetch=fetch_weights))
    checks.extend(_check_backbone(experiments))
    checks.extend(_check_inputs(config, experiments))
    checks.extend(_check_disk(artifacts_root))
    checks.extend(_check_existing_results(config, artifacts_root))
    conditions = read_conditions()
    checks.extend(_check_conditions(conditions, kind))
    checks.extend(_check_load(conditions))
    checks.extend(_check_python())
    return PreflightReport(
        suite=config.name,
        suite_hash=suite_hash(config),
        machine_id=machine_id,
        device_requested=config.device,
        started_at=datetime.now(UTC),
        checks=checks,
        environment=_environment_on(kind),
        conditions=conditions,
        ok=all(check.status != "fail" for check in checks),
    )


def write_report(report: PreflightReport, artifacts_root: Path) -> Path:
    """Beside the results the suite will write, named by when it ran."""
    directory = results_dir(artifacts_root, report.suite)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.strftime("%Y%m%dT%H%M%S%f")
    path = directory / f"{PREFLIGHT_PREFIX}{stamp}.json"
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Checks. Each returns what it found; none raises for a finding.
# --------------------------------------------------------------------------- #


def _check_git() -> Iterator[Check]:
    from unproject.run.manifest import git_commit, git_dirty

    commit = git_commit()
    if commit is None:
        yield Check(name="git", status="fail", detail="not a git checkout; a result needs a commit")
        return
    if git_dirty():
        yield Check(
            name="git",
            status="fail",
            detail=f"tree is dirty at {commit[:12]}; commit or stash, a result needs a clean tree",
        )
    else:
        yield Check(name="git", status="ok", detail=f"clean tree at {commit[:12]}")


def _check_machine(machine_id: str | None, registry: MachineRegistry | None) -> Iterator[Check]:
    if machine_id is None:
        yield Check(
            name="machine",
            status="fail",
            detail="no machine id; pass --machine <id> from benchmarks/machines.toml",
        )
        return
    if registry is None:
        yield Check(
            name="machine", status="warn", detail=f"{machine_id}: no registry to check against"
        )
        return
    machine = registry.find(machine_id)
    if machine is None:
        known = ", ".join(m.id for m in registry.machines)
        yield Check(
            name="machine", status="fail", detail=f"{machine_id!r} is not declared; known: {known}"
        )
        return
    yield Check(
        name="machine",
        status="ok",
        detail=f"{machine.id}: {machine.label} ({machine.hardware_class}, {machine.device})",
    )


def _check_device(
    config: SuiteConfig,
    machine_id: str | None,
    registry: MachineRegistry | None,
    checks: list[Check],
) -> DeviceKind | None:
    """Resolve the device the way the job will, and hold it to the registry."""
    env_value = os.environ.get(ENV_DEVICE, "").strip().lower()
    if config.device != "auto" and env_value and env_value not in {"auto", config.device}:
        checks.append(
            Check(
                name="device",
                status="fail",
                detail=f"suite asks for {config.device} but {ENV_DEVICE}={env_value}; unset it",
            )
        )
        return None
    requested = config.device if config.device != "auto" else None
    try:
        kind = resolve_kind(requested)
    except ImportError:
        checks.append(Check(name="device", status="fail", detail="torch is not installed"))
        return None
    except DeviceError as error:
        checks.append(Check(name="device", status="fail", detail=str(error)))
        return None

    described = _environment_on(kind)
    detail = f"{kind}: {described.device_name or described.machine}"
    if kind == "cuda":
        detail += f", CUDA {described.cuda_version}, driver {described.driver_version}"
    if described.torch_threads is not None:
        detail += f", {described.torch_threads} torch thread(s)"
    declared = registry.find(machine_id) if registry is not None and machine_id else None
    expected = None if declared is None else declared.device
    if config.device == "auto" and expected is not None and expected != kind:
        checks.append(
            Check(
                name="device",
                status="fail",
                detail=f"{detail}; the registry expects {expected} on {machine_id}",
            )
        )
    elif config.device == "auto" and kind == "cpu":
        checks.append(
            Check(
                name="device",
                status="warn",
                detail=f"{detail}; no accelerator found, the suite will run on the CPU",
            )
        )
    else:
        checks.append(Check(name="device", status="ok", detail=detail))
    return kind


def _environment_on(kind: DeviceKind | None) -> Environment:
    """The record as a job of this suite would write it: describing the device
    the suite will use, not the one `auto` would pick, so a CPU suite's
    report on a machine with a card says `cpu` and names the CPU."""
    if kind is None:
        return environment()
    try:
        import torch
    except ImportError:
        return environment()
    return environment(describe_device(torch.device(kind)))


def _check_estimator() -> Iterator[Check]:
    if importlib.util.find_spec("multihmr2") is None:
        yield Check(
            name="estimator",
            status="fail",
            detail="multihmr2 is not installed: uv sync --frozen --extra multihmr",
        )
    else:
        yield Check(name="estimator", status="ok", detail="multihmr2 importable")


def _cell_experiments(
    config: SuiteConfig, suite_dir: Path, checks: list[Check]
) -> list[tuple[str, ExperimentConfig, Path]]:
    """Every cell's experiment, or a failed check for the ones that do not load."""
    loaded: list[tuple[str, ExperimentConfig, Path]] = []
    for cell in config.cells:
        try:
            experiment, directory = load_cell_experiment(config, cell, suite_dir)
        except ConfigError as error:
            checks.append(Check(name=f"config:{cell.name}", status="fail", detail=str(error)))
            continue
        loaded.append((cell.name, experiment, directory))
    return loaded


def _check_weights(
    experiments: list[tuple[str, ExperimentConfig, Path]], *, fetch: bool
) -> Iterator[Check]:
    from unproject.bodies.multihmr import BodyEstimationError, checkpoint_status, ensure_checkpoint

    seen: set[tuple[str, str]] = set()
    for _, experiment, directory in experiments:
        bodies = experiment.bodies
        key = (str((directory / bodies.cache_dir).resolve()), bodies.sha256)
        if key in seen:
            continue
        seen.add(key)
        if fetch:
            try:
                path = ensure_checkpoint(bodies, directory)
            except BodyEstimationError as error:
                yield Check(name="weights", status="fail", detail=str(error))
                continue
            yield Check(
                name="weights", status="ok", detail=f"{path}: fetched or present, digest matches"
            )
            continue
        status, detail = checkpoint_status(bodies, directory)
        yield Check(name="weights", status=status, detail=detail)


def _check_backbone(experiments: list[tuple[str, ExperimentConfig, Path]]) -> Iterator[Check]:
    """torch.hub keeps the backbone source under a directory named for the ref."""
    try:
        import torch
    except ImportError:
        return
    seen: set[str] = set()
    for _, experiment, _ in experiments:
        bodies = experiment.bodies
        name = f"{bodies.backbone_repo.replace('/', '_')}_{bodies.backbone_ref}"
        if name in seen:
            continue
        seen.add(name)
        cached = Path(torch.hub.get_dir()) / name
        if cached.is_dir():
            yield Check(name="backbone", status="ok", detail=f"{name} in the torch.hub cache")
        else:
            yield Check(
                name="backbone",
                status="warn",
                detail=(
                    f"{name} not cached; the first job fetches it from GitHub and needs the network"
                ),
            )


def _check_inputs(
    config: SuiteConfig, experiments: list[tuple[str, ExperimentConfig, Path]]
) -> Iterator[Check]:
    """The footage is there, is the campaign's bytes, and is long enough."""
    needed = config.warmup_frames + config.measured_frames
    checked: dict[str, Check] = {}
    for cell_name, experiment, directory in experiments:
        key = experiment.name
        if key in checked:
            yield checked[key].model_copy(update={"name": f"input:{cell_name}"})
            continue
        check = _check_one_input(cell_name, experiment, directory, needed)
        checked[key] = check
        yield check


def _check_one_input(
    cell_name: str, experiment: ExperimentConfig, directory: Path, needed: int
) -> Check:
    name = f"input:{cell_name}"
    ingest = experiment.ingest
    path = ingest.source_path
    resolved = path if path.is_absolute() else (directory / path).resolve()
    if not resolved.exists():
        return Check(name=name, status="fail", detail=f"{resolved} does not exist")
    try:
        metadata = open_source(ingest, directory).metadata()
    except (FrameSourceError, OSError, ValueError) as error:
        return Check(name=name, status="fail", detail=f"{resolved}: cannot read: {error}")
    except Exception as error:  # the ingest stage's own error type, whatever it is
        return Check(
            name=name, status="fail", detail=f"{resolved}: {type(error).__name__}: {error}"
        )
    digest = metadata.content_sha256
    expected = experiment.provenance.expected_sha256 if experiment.provenance else None
    if expected is not None and expected != digest:
        return Check(
            name=name,
            status="fail",
            detail=(
                f"{resolved} hashes to {digest[:12]}… but the campaign expects "
                f"{expected[:12]}…; these are not the same bytes"
            ),
        )
    frames = metadata.reported_frame_count
    if frames is not None:
        selected = frames // max(1, ingest.frame_stride)
        if ingest.max_frames is not None:
            selected = min(selected, ingest.max_frames)
        if selected < needed:
            return Check(
                name=name,
                status="fail",
                detail=(
                    f"{resolved}: about {selected} frame(s) after selection, "
                    f"the suite needs {needed}"
                ),
            )
    verified = "digest matches" if expected is not None else "no expected digest declared"
    size = f"{metadata.width}x{metadata.height}"
    if experiment.ingest.geometry is not None:
        try:
            width, height = output_size(metadata.width, metadata.height, experiment.ingest.geometry)
        except GeometryError as error:
            return Check(name=name, status="fail", detail=f"{resolved.name}: {error}")
        size = f"{size} -> {width}x{height} (ingest geometry)"
    return Check(
        name=name,
        status="ok" if expected is not None else "warn",
        detail=(
            f"{resolved.name}: {size}, "
            f"{frames if frames is not None else '?'} frame(s), sha256 {digest[:12]}…, {verified}"
        ),
    )


def _check_disk(artifacts_root: Path) -> Iterator[Check]:
    probe = artifacts_root if artifacts_root.exists() else _nearest_existing(artifacts_root)
    try:
        usage = shutil.disk_usage(probe)
    except OSError as error:
        yield Check(name="disk", status="fail", detail=f"{artifacts_root}: {error}")
        return
    free_gb = usage.free / 1e9
    detail = f"{free_gb:.1f} GB free at {probe}"
    if free_gb < FAIL_FREE_GB:
        yield Check(name="disk", status="fail", detail=f"{detail}; below {FAIL_FREE_GB:.0f} GB")
    elif free_gb < WARN_FREE_GB:
        yield Check(name="disk", status="warn", detail=f"{detail}; below {WARN_FREE_GB:.0f} GB")
    else:
        yield Check(name="disk", status="ok", detail=detail)
    if artifacts_root.exists() and not os.access(artifacts_root, os.W_OK):
        yield Check(name="disk", status="fail", detail=f"{artifacts_root} is not writable")


def _nearest_existing(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _check_existing_results(config: SuiteConfig, artifacts_root: Path) -> Iterator[Check]:
    directory = results_dir(artifacts_root, config.name)
    if not directory.is_dir():
        yield Check(name="results", status="ok", detail=f"{directory} is new")
        return
    existing = [
        p
        for p in directory.rglob(f"*{RESULT_SUFFIX}")
        if not p.name.startswith(PREFLIGHT_PREFIX) and p.name != "summary.json"
    ]
    if existing:
        yield Check(
            name="results",
            status="warn",
            detail=(
                f"{directory} already holds {len(existing)} file(s); the summary pools every "
                "job it finds, so a new campaign run belongs under a fresh root"
            ),
        )
    else:
        yield Check(name="results", status="ok", detail=f"{directory} holds no results yet")


def _check_conditions(conditions: Conditions, kind: DeviceKind | None) -> Iterator[Check]:
    if platform.system() == "Darwin":
        if conditions.power_source == "battery":
            yield Check(name="power", status="fail", detail="on battery; plug the machine in")
        elif conditions.power_source == "mains":
            yield Check(name="power", status="ok", detail="on mains")
        else:
            yield Check(name="power", status="warn", detail="power source unknown to pmset")
        if conditions.low_power_mode is True:
            yield Check(name="low-power", status="fail", detail="low power mode is on; turn it off")
        elif conditions.low_power_mode is False:
            yield Check(name="low-power", status="ok", detail="low power mode off")
        else:
            yield Check(name="low-power", status="warn", detail="low power mode not reported")
        limit = conditions.cpu_speed_limit_before
        if limit is None:
            yield Check(
                name="thermal",
                status="warn",
                detail=(
                    "pmset reports no CPU speed limit on this machine; "
                    "the thermal state is unrecorded"
                ),
            )
        elif limit < 100:
            yield Check(
                name="thermal",
                status="warn",
                detail=f"CPU speed limit {limit}%; the machine is throttling",
            )
        else:
            yield Check(name="thermal", status="ok", detail="CPU speed limit 100%")
        return

    if conditions.cpu_governor is not None:
        yield Check(
            name="governor", status="ok", detail=f"cpufreq governor {conditions.cpu_governor}"
        )
    if kind == "cuda":
        gpu = conditions.gpu_before
        if gpu is None:
            yield Check(name="gpu", status="warn", detail="nvidia-smi returned nothing to record")
        else:
            detail = (
                f"{gpu.temperature_c} °C, {gpu.power_draw_w} W, SM {gpu.sm_clock_mhz} MHz, "
                f"{gpu.performance_state}, util {gpu.utilization_percent}%, "
                f"reasons {gpu.throttle_reasons}"
            )
            # Nothing of ours runs during the preflight, so utilisation is
            # someone else's, e.g. a tenant in another container that
            # `--query-compute-apps` below cannot see.
            utilisation = gpu.utilization_percent
            if utilisation is not None and utilisation > 20.0:
                yield Check(
                    name="gpu",
                    status="fail",
                    detail=f"{detail}; the card is busy for a process this container cannot see",
                )
            else:
                busy = utilisation is not None and utilisation > 5.0
                yield Check(name="gpu", status="warn" if busy else "ok", detail=detail)
        yield from _check_gpu_processes()


def _check_gpu_processes() -> Iterator[Check]:
    """Anything else on the card contaminates the clock; that is a stop."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if completed.returncode != 0:
        return
    others = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if others:
        yield Check(
            name="gpu-processes",
            status="fail",
            detail="other work on the card: " + "; ".join(others),
        )
    else:
        yield Check(name="gpu-processes", status="ok", detail="no other process on the card")


def _check_load(conditions: Conditions) -> Iterator[Check]:
    load = conditions.load_average_before
    if load is None:
        return
    hogs = _busy_processes()
    detail = f"1-minute load average {load:.2f}"
    if hogs:
        detail += "; busy: " + ", ".join(hogs)
    status: Status = "warn" if load > WARN_LOAD_AVERAGE or hogs else "ok"
    yield Check(name="load", status=status, detail=detail)


def _busy_processes() -> list[str]:
    """Processes over `WARN_PROCESS_CPU_PERCENT` of a core, other than this one."""
    try:
        completed = subprocess.run(
            ["ps", "-Ao", "pid=,pcpu=,comm="],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    busy: list[str] = []
    me = os.getpid()
    for line in completed.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid, cpu, command = parts
        try:
            if int(pid) == me or float(cpu) < WARN_PROCESS_CPU_PERCENT:
                continue
        except ValueError:
            continue
        busy.append(f"{Path(command).name} ({cpu}%)")
    return busy[:8]


def _check_python() -> Iterator[Check]:
    version = platform.python_version()
    if version.startswith("3.12."):
        yield Check(name="python", status="ok", detail=version)
    else:
        yield Check(name="python", status="warn", detail=f"{version}; the project pins 3.12")


def print_report(report: PreflightReport) -> None:
    width = max(len(check.name) for check in report.checks) if report.checks else 8
    for check in report.checks:
        marker = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}[check.status]
        print(f"{marker}  {check.name:<{width}}  {check.detail}")
    failed = sum(1 for check in report.checks if check.status == "fail")
    warned = sum(1 for check in report.checks if check.status == "warn")
    verdict = "ready" if report.ok else "not ready"
    print(
        f"{verdict}: {report.suite} on {report.machine_id or 'undeclared machine'} "
        f"({report.device_requested}); {failed} failure(s), {warned} warning(s)"
    )


__all__ = [
    "PREFLIGHT_PREFIX",
    "PREFLIGHT_SCHEMA_VERSION",
    "Check",
    "PreflightReport",
    "print_report",
    "run_preflight",
    "write_report",
]
