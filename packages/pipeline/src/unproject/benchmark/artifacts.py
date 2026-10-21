"""The record one v1 measurement run leaves behind.

Raw observations, never rewritten, like the v2 job record (`results.py`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.benchmark.config import BenchmarkConfig

BENCHMARK_RESULT_SCHEMA_VERSION = 1

RESULT_SUFFIX = ".json"


# The operator's name for the machine a campaign job runs on (ADR-0006 §3):
# an id from `benchmarks/machines.toml`, so a result says which declared
# machine it belongs to and not only which chip it found.
ENV_MACHINE_ID = "UNPROJECT_MACHINE_ID"


class Environment(BaseModel):
    """The machine and the code. Every performance claim is about these."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    machine: str
    platform: str
    python_version: str
    # "cuda", "mps" or "cpu"; "unavailable" when none could be resolved. A
    # number without this says nothing.
    device: str
    package_versions: dict[str, str]
    git_commit: str | None = None
    # A dirty tree means the commit does not identify the code that ran.
    git_dirty: bool = False
    # Added by ADR-0006, all optional so a result written before it still
    # reads. The declared machine, the accelerator as its driver names it, and
    # the software under it.
    machine_id: str | None = None
    device_name: str | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    driver_version: str | None = None
    os_version: str | None = None
    cpu_count: int | None = Field(default=None, ge=1)
    memory_total_bytes: int | None = Field(default=None, ge=0)
    # Intra-op threads torch runs with; on the CPU this decides the number.
    torch_threads: int | None = Field(default=None, ge=1)


class Inputs(BaseModel):
    """What went in, identified well enough to run it again."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment: str
    # Content digest of the footage, not its path. See ADR-0002 §7.2.
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ingest_hash: str
    benchmark_config_hash: str
    bodies_config_hash: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backbone_ref: str


class FrameSample(BaseModel):
    """One frame through the model.

    Kept individually rather than summarised: a median hides the tail, and the
    tail is what an interactive viewer is actually made of.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repetition: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    # Warm-up samples never enter a steady-state aggregate. ADR-0002 §8.2.
    warmup: bool
    seconds: float = Field(ge=0.0)
    people: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Repetition(BaseModel):
    """One pass over the frames, with what it cost to get started."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    # Building the model and its session. Measured separately from steady state
    # and never averaged into it.
    cold_start_seconds: float = Field(ge=0.0)
    peak_memory_bytes: int = Field(ge=0)
    # Absent unless the installed PyTorch exposes the Metal counters.
    metal_allocated_bytes: int | None = Field(default=None, ge=0)
    metal_driver_bytes: int | None = Field(default=None, ge=0)


class BenchmarkResult(BaseModel):
    """One invocation of the benchmark, whatever it managed to measure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=BENCHMARK_RESULT_SCHEMA_VERSION, ge=1)
    name: str
    started_at: datetime
    finished_at: datetime
    status: Literal["ok", "failed"] = "ok"
    # Why this run is not evidence, when it is not. Exclusions stay visible.
    error: str | None = None
    # A dry run has the same shape as a real one and measures nothing. Recorded
    # here rather than inferred, so aggregation can refuse to average it with
    # real timings.
    dry_run: bool = False
    config: BenchmarkConfig
    environment: Environment
    inputs: Inputs
    repetitions: list[Repetition]
    frames: list[FrameSample]

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != BENCHMARK_RESULT_SCHEMA_VERSION:
            msg = (
                f"benchmark result schema_version {self.schema_version} is not readable "
                f"by this build (expected {BENCHMARK_RESULT_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_samples_belong_to_a_repetition(self) -> Self:
        known = {repetition.index for repetition in self.repetitions}
        unknown = sorted({sample.repetition for sample in self.frames} - known)
        if unknown:
            msg = f"frames reference repetition {unknown} that the result does not declare"
            raise ValueError(msg)
        return self

    def steady_state(self) -> list[FrameSample]:
        """The samples an aggregate may use. Warm-up is not one of them."""
        return [sample for sample in self.frames if not sample.warmup]
