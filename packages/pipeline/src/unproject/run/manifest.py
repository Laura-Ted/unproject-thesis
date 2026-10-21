"""Per-stage run manifest.

The config says what to run; this records the code and environment it ran
against, plus what the stage cost.
"""

from __future__ import annotations

import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

# Recorded per run: an upstream version bump can move the metrics on its own.
_TRACKED_PACKAGES = (
    "unproject-pipeline",
    "numpy",
    "opencv-python-headless",
    "pydantic",
    # Absent unless the learned backend's extra is installed; recorded when it is,
    # because either one moves the poses.
    "torch",
    "lingbot-map",
    # Stage [B]'s estimator and the body model it drives. An upstream change to
    # either moves the bodies, so both belong in the record of what ran.
    "multihmr2",
    "anny",
)


class RunManifest(BaseModel):
    """What ran, from which commit, with which dependencies, at what cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=MANIFEST_SCHEMA_VERSION, ge=1)
    stage: str
    experiment: str
    config_hash: str
    seed: int
    status: Literal["ok", "failed"] = "ok"
    # Exception type and message when the stage raised, otherwise None.
    error: str | None = None
    git_commit: str | None = None
    # If the tree was dirty, the commit alone does not identify the code that ran.
    git_dirty: bool = False
    python_version: str
    platform: str
    package_versions: dict[str, str]
    started_at: datetime
    finished_at: datetime
    wall_clock_seconds: float = Field(ge=0.0)
    # Whole-process high-water mark, not this stage's own allocation. It is
    # what the ~12 GB per-stage budget is measured against.
    peak_memory_bytes: int = Field(ge=0)


class StageRecorder:
    """Times a stage and turns the result into a `RunManifest`.

    Records failures too, so a stage that dies halfway still leaves a trace of
    how long it ran and what stopped it. Usage::

        recorder = StageRecorder("ingest", experiment=..., config_hash=..., seed=...)
        try:
            with recorder:
                ...
        finally:
            write_manifest(recorder.finish(), directory)
    """

    def __init__(self, stage: str, *, experiment: str, config_hash: str, seed: int) -> None:
        self._stage = stage
        self._experiment = experiment
        self._config_hash = config_hash
        self._seed = seed
        self._started_at: datetime | None = None
        self._start_counter = 0.0
        self._elapsed_seconds: float | None = None
        self._finished_at: datetime | None = None
        self._failure: str | None = None

    def __enter__(self) -> Self:
        self._started_at = datetime.now(UTC)
        self._start_counter = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Runs on failure too, including KeyboardInterrupt, which is how a long
        # stage usually ends when it is going nowhere.
        self._elapsed_seconds = time.perf_counter() - self._start_counter
        self._finished_at = datetime.now(UTC)
        if exc_value is not None:
            self._failure = _describe(exc_value)

    def finish(self) -> RunManifest:
        if self._started_at is None or self._finished_at is None or self._elapsed_seconds is None:
            raise RuntimeError("StageRecorder.finish() called outside a completed `with` block")

        return RunManifest(
            stage=self._stage,
            experiment=self._experiment,
            config_hash=self._config_hash,
            seed=self._seed,
            status="failed" if self._failure is not None else "ok",
            error=self._failure,
            git_commit=git_commit(),
            git_dirty=git_dirty(),
            python_version=platform.python_version(),
            platform=f"{platform.system()} {platform.release()} {platform.machine()}",
            package_versions=package_versions(),
            started_at=self._started_at,
            finished_at=self._finished_at,
            wall_clock_seconds=self._elapsed_seconds,
            peak_memory_bytes=peak_memory_bytes(),
        )


def write_manifest(manifest: RunManifest, directory: Path) -> Path:
    """Write `manifest.json` next to the artifacts it describes."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILENAME
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def peak_memory_bytes() -> int:
    """Process peak resident set size, normalised to bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Same field, different units: macOS reports bytes, Linux kibibytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _describe(error: BaseException) -> str:
    """One line naming the exception, for the manifest."""
    message = str(error).strip()
    name = type(error).__name__
    return f"{name}: {message}" if message else name


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # Not installed: an extra left out, or a source tree with no distribution.
            continue
    return versions


def git_commit() -> str | None:
    output = _git("rev-parse", "HEAD")
    return output if output else None


def git_dirty() -> bool:
    output = _git("status", "--porcelain")
    return bool(output)


def _git(*args: str) -> str | None:
    """Run a read-only git command; returns None if git or the repo is missing."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
