"""The record one benchmark job leaves behind, under protocol v2.

A job is one cell of the matrix measured once, in a process of its own
(ADR-0005 WP2): the model is loaded in that process, the frames are timed in
that process, and the process exits. So nothing - not the weights, not the
Metal allocator, not a warmed-up cache - carries over from one repetition to
the next, and a repetition's memory figures are its own rather than a
high-water mark inherited from whatever ran before.

Raw observations, not summaries. Every frame that was timed is here with its
phases separated, and the tables are derived from these files rather than
written beside them. A result is never rewritten, so a number in the thesis can
always be traced back to the file that produced it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.benchmark.artifacts import Environment
from unproject.benchmark.config import OutputMode
from unproject.config import Provenance

JOB_RESULT_SCHEMA_VERSION = 2

# Which part of a job a frame belongs to. Warm-up is timed and kept, and never
# enters a steady-state figure; ADR-0002 §8.2.
type Phase = Literal["warmup", "steady"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AcceleratorReading(_Frozen):
    """One reading of an NVIDIA card, as `nvidia-smi` reports it.

    The Linux counterpart of macOS's thermal limit: temperature, power, the
    clocks, and the reasons the driver gives for holding a clock down, which
    is where a throttle shows. Fields are absent where the query returned
    nothing for them.
    """

    temperature_c: float | None = None
    power_draw_w: float | None = None
    sm_clock_mhz: float | None = None
    memory_clock_mhz: float | None = None
    # "0x0000000000000000" is none; anything else names a reason.
    throttle_reasons: str | None = None
    performance_state: str | None = None
    utilization_percent: float | None = None


class Conditions(_Frozen):
    """The manual conditions that move a clock without touching the code.

    Read from the OS where it exposes them, and recorded as read: the raw
    strings are kept beside the parsed values so a later reader can check the
    parsing rather than trust it. `unknown` is a value, not an omission.
    """

    power_source: Literal["mains", "battery", "unknown"] = "unknown"
    low_power_mode: bool | None = None
    # macOS reports a CPU speed limit under thermal pressure; 100 is none.
    # Before and after the job, because a throttle that arrives mid-run is
    # exactly what this is here to catch.
    cpu_speed_limit_before: int | None = Field(default=None, ge=0, le=100)
    cpu_speed_limit_after: int | None = Field(default=None, ge=0, le=100)
    # Added by ADR-0006. The one-minute load average on either OS, before and
    # after, which is what shows another process sharing the machine; the
    # CPU frequency governor and the card's own readings on Linux.
    load_average_before: float | None = Field(default=None, ge=0.0)
    load_average_after: float | None = Field(default=None, ge=0.0)
    cpu_governor: str | None = None
    gpu_before: AcceleratorReading | None = None
    gpu_after: AcceleratorReading | None = None
    raw: dict[str, str] = Field(default_factory=dict)


class InputGeometry(_Frozen):
    """The frame as given, and the tensor the model actually saw.

    Both are recorded because the cost is decided by the tensor (long edge
    resized to 768, padded to whole patches), not by the source size; see
    `unproject.bodies.multihmr.model_input_size`.
    """

    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    model_input_width: int = Field(gt=0)
    model_input_height: int = Field(gt=0)
    # Source pixels to model pixels.
    scale: float = Field(gt=0.0)


class Memory(_Frozen):
    """Process and accelerator memory at three moments, in bytes.

    Baseline is before the model is built, so the model's own footprint is
    `after_load - baseline` rather than a whole-process number that includes
    the interpreter and the frames held for timing. Peak is the process
    high-water mark at the end, which in a fresh process per job is this job's.
    Metal counters are absent, not zero, when the installed PyTorch does not
    expose them.
    """

    rss_baseline_bytes: int | None = Field(default=None, ge=0)
    rss_after_load_bytes: int | None = Field(default=None, ge=0)
    rss_end_bytes: int | None = Field(default=None, ge=0)
    peak_rss_bytes: int = Field(ge=0)
    metal_allocated_baseline_bytes: int | None = Field(default=None, ge=0)
    metal_driver_baseline_bytes: int | None = Field(default=None, ge=0)
    metal_allocated_peak_bytes: int | None = Field(default=None, ge=0)
    metal_driver_peak_bytes: int | None = Field(default=None, ge=0)
    # CUDA's own high-water marks for the process, read at the end (ADR-0006).
    # Absent on Metal and on the CPU. Reserved is what the allocator holds
    # from the driver, the figure a card runs out of.
    cuda_allocated_peak_bytes: int | None = Field(default=None, ge=0)
    cuda_reserved_peak_bytes: int | None = Field(default=None, ge=0)


class Phases(_Frozen):
    """What happened before the first frame, timed apart from the frames."""

    # Importing torch, fetching the checkpoint digest, building the model.
    model_load_seconds: float = Field(ge=0.0)
    # Opening a session: a tracker, and the first touch of the loaded model.
    session_open_seconds: float = Field(ge=0.0)
    # Wall clock around the steady phase, first frame in to last message out.
    # Throughput is frames over this, not the reciprocal of a median.
    steady_wall_seconds: float = Field(ge=0.0)


class FrameSample(_Frozen):
    """One frame through the estimator and out to the wire, timed in parts.

    `inference_seconds` is the estimator: preprocessing, the forward pass, the
    tracker, and the conversion to the project's own types, on a frame already
    in memory. `rest_body_seconds` is what mesh output adds when a person is
    seen for the first time. `serialize_seconds` is the wire message, built by
    the same code the socket uses. Ingest, network and rendering are not here;
    WP3 measures them.
    """

    phase: Phase
    # Position in this job's frame list, and the ingest frame it came from.
    position: int = Field(ge=0)
    source_index: int = Field(ge=0)
    inference_seconds: float = Field(ge=0.0)
    rest_body_seconds: float = Field(ge=0.0)
    serialize_seconds: float = Field(ge=0.0)
    people: int = Field(ge=0)
    new_tracks: int = Field(ge=0)
    message_bytes: int = Field(ge=0)
    rest_body_bytes: int = Field(ge=0)
    # The accelerator's allocated counter sampled after the frame (Metal or
    # CUDA; the name predates CUDA). On Metal the peak in `Memory` is their
    # max, because Metal keeps no high-water mark of its own.
    metal_allocated_bytes: int | None = Field(default=None, ge=0)

    @property
    def total_seconds(self) -> float:
        return self.inference_seconds + self.rest_body_seconds + self.serialize_seconds


class JobIdentity(_Frozen):
    """Which suite, cell and repetition this is, and where it sat in the order."""

    suite: str
    suite_hash: str
    cell: str
    output: OutputMode
    repetition: int = Field(ge=0)
    # Position in the shuffled run order, and the seed that produced it.
    job_index: int = Field(ge=0)
    job_count: int = Field(gt=0)
    order_seed: int


class JobInputs(_Frozen):
    """What went in, identified well enough to run it again."""

    experiment: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ingest_hash: str
    bodies_config_hash: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backbone_ref: str
    provenance: Provenance | None = None
    # The source frames that were timed, in order, and the warm-up ones before
    # them. Every cell of a suite gets the same list, and the summary checks.
    warmup_indices: list[int]
    measured_indices: list[int]


class JobResult(_Frozen):
    """One job, whatever it managed to measure."""

    schema_version: int = Field(default=JOB_RESULT_SCHEMA_VERSION, ge=1)
    identity: JobIdentity
    started_at: datetime
    finished_at: datetime
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None
    # A dry run has the same shape and measures nothing; the summary refuses
    # to average it with a real one.
    dry_run: bool = False
    inputs: JobInputs
    environment: Environment
    conditions: Conditions
    geometry: InputGeometry
    phases: Phases
    memory: Memory
    frames: list[FrameSample]

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != JOB_RESULT_SCHEMA_VERSION:
            msg = (
                f"benchmark job schema_version {self.schema_version} is not readable "
                f"by this build (expected {JOB_RESULT_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_frames_follow_the_plan(self) -> Self:
        """A frame list that disagrees with the declared indices is a job that
        measured something other than what it says."""
        planned = [*self.inputs.warmup_indices, *self.inputs.measured_indices]
        seen = [sample.source_index for sample in self.frames]
        if self.status == "ok" and seen != planned:
            msg = f"frames timed {seen} but the plan was {planned}"
            raise ValueError(msg)
        if self.status == "failed" and seen != planned[: len(seen)]:
            msg = f"a failed job timed {seen}, which is not a prefix of the plan {planned}"
            raise ValueError(msg)
        return self

    def steady_state(self) -> list[FrameSample]:
        """The samples an aggregate may use. Warm-up is not one of them."""
        return [sample for sample in self.frames if sample.phase == "steady"]
