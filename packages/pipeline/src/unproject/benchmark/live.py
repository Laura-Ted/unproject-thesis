"""What a live end-to-end measurement leaves behind, and how it is read.

ADR-0005 WP3. The live path is measured from the client, because that is where
latency is felt: a frame gets a monotone `frame_id` when it is captured, and
every message that comes back carries it, so a client can pair what it sent
with what it received on its own clock. The server attaches durations - decode,
queue wait, inference, serialisation - on its own monotonic clock. **The two
clocks are never subtracted from each other**: a browser's `performance.now()`
and a server's `perf_counter()` have different origins, and a "latency"
computed across them would be an artefact of the origins, not a measurement.

Two kinds of report share this schema. A **transport** report comes from
`unproject-live-benchmark`, a Python client that pushes JPEG frames from a
public clip over the real socket at a controlled rate; it can run unattended
and measures capture-to-response. A **browser** report comes from the viewer in
measure mode, which adds the time from response to the first animation frame
after the bodies were applied. Neither is presented as the other, and no
single end-to-end number is invented by joining them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unproject.bodies.wire import FrameTiming

LIVE_REPORT_SCHEMA_VERSION = 1

# Where reports land, beside the other benchmarks.
STAGE_NAME = "live"
REPORT_SUFFIX = ".json"

type ReportKind = Literal["transport", "browser"]
type LiveMode = Literal["skeleton", "mesh"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# The durations the server attaches to a frame message. One definition, in
# the wire module, so the report stores exactly what the socket sent.
ServerTiming = FrameTiming


class LiveSample(_Frozen):
    """One frame the client sent, and what became of it.

    Client times are milliseconds on the client's clock, relative to the moment
    the session was opened. A frame the server replaced in its latest-only
    mailbox, or that was still in flight when the session ended, has no
    `received_at_ms`; it is a row, not a gap.
    """

    frame_id: int = Field(ge=0)
    sent_at_ms: float = Field(ge=0.0)
    sent_bytes: int = Field(ge=0)
    received_at_ms: float | None = Field(default=None, ge=0.0)
    # Browser only: the first requestAnimationFrame after the bodies were
    # applied to the timeline. Not the moment those bodies were drawn - the
    # live clock lags the newest frame by design, and that lag is in `config`.
    presented_at_ms: float | None = Field(default=None, ge=0.0)
    received_bytes: int | None = Field(default=None, ge=0)
    people: int | None = Field(default=None, ge=0)
    # The server's count of frames it discarded before this one.
    dropped_before: int | None = Field(default=None, ge=0)
    server: ServerTiming | None = None

    @model_validator(mode="after")
    def _check_the_answer_follows_the_question(self) -> Self:
        if self.received_at_ms is not None and self.received_at_ms < self.sent_at_ms:
            msg = f"frame {self.frame_id} was received before it was sent"
            raise ValueError(msg)
        if self.presented_at_ms is not None and self.received_at_ms is None:
            msg = f"frame {self.frame_id} was presented without being received"
            raise ValueError(msg)
        if (
            self.presented_at_ms is not None
            and self.received_at_ms is not None
            and self.presented_at_ms < self.received_at_ms
        ):
            msg = f"frame {self.frame_id} was presented before it was received"
            raise ValueError(msg)
        return self


class LiveEnvironment(_Frozen):
    """Where the client and the server ran."""

    # Client side.
    user_agent: str | None = None
    viewport: str | None = None
    client_platform: str
    # Server side, as the server reported it at session start.
    server_machine: str | None = None
    server_device: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None


class LiveConfig(_Frozen):
    """What the client did, so the run can be repeated."""

    send_fps: float = Field(gt=0.0)
    max_width: int = Field(gt=0)
    jpeg_quality: float = Field(gt=0.0, le=1.0)
    bodies: bool
    # Browser only: how far the live clock trails the newest frame, seconds.
    live_clock_lag_seconds: float | None = Field(default=None, ge=0.0)
    # How the run was bounded: a frame count or a clip's end.
    frame_limit: int | None = Field(default=None, gt=0)


class Distribution(_Frozen):
    samples: int = Field(ge=0)
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    max: float | None = None


class LiveSummary(_Frozen):
    """Derived from the samples by `summarise_live`, and checked against them
    on read, so a stored summary cannot disagree with its own rows."""

    frames_sent: int = Field(ge=0)
    frames_answered: int = Field(ge=0)
    # The server's own count of frames it replaced in the mailbox.
    frames_dropped_by_server: int = Field(ge=0)
    # Sent, never answered, and not accounted for by the server's drops: in
    # flight at the end, lost, or dropped after the last answer - a drop is
    # only reported by the next answer, so one after the final one has no
    # messenger.
    frames_unaccounted: int = Field(ge=0)
    bytes_sent: int = Field(ge=0)
    bytes_received: int = Field(ge=0)
    duration_ms: float = Field(ge=0.0)
    # Answered frames over the session, which is the rate the viewer got.
    answered_fps: float | None = None
    # Client clock: send to receive, and send to first animation frame.
    round_trip_ms: Distribution
    capture_to_present_ms: Distribution
    receive_to_present_ms: Distribution
    # Server clock.
    decode_ms: Distribution
    queue_wait_ms: Distribution
    inference_ms: Distribution
    serialize_ms: Distribution
    people: Distribution


class LiveReport(_Frozen):
    schema_version: int = Field(default=LIVE_REPORT_SCHEMA_VERSION, ge=1)
    kind: ReportKind
    started_at: datetime
    # The clip the frames came from, and its digest where the client knows it.
    source: str
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mode: LiveMode
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    environment: LiveEnvironment
    config: LiveConfig
    samples: list[LiveSample]
    summary: LiveSummary
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_schema_version(self) -> Self:
        if self.schema_version != LIVE_REPORT_SCHEMA_VERSION:
            msg = (
                f"live report schema_version {self.schema_version} is not readable by this "
                f"build (expected {LIVE_REPORT_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_the_summary_is_the_samples(self) -> Self:
        expected = summarise_live(self.samples)
        if self.summary != expected:
            msg = "the stored summary does not match its own samples; it was edited or is stale"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_frame_ids_are_unique(self) -> Self:
        ids = [sample.frame_id for sample in self.samples]
        if len(set(ids)) != len(ids):
            msg = "a frame_id appears twice in the samples"
            raise ValueError(msg)
        return self


def summarise_live(samples: Sequence[LiveSample]) -> LiveSummary:
    """The numbers a live report is read by. Pure; the same rows give the same result."""
    answered = [s for s in samples if s.received_at_ms is not None]
    # Each answered frame carries the frames the server discarded since its
    # previous answer, so the session's total is their sum.
    dropped = sum(s.dropped_before or 0 for s in answered)
    ends = [s.sent_at_ms for s in samples] + [
        s.received_at_ms for s in answered if s.received_at_ms is not None
    ]
    duration = max(ends) if ends else 0.0
    return LiveSummary(
        frames_sent=len(samples),
        frames_answered=len(answered),
        frames_dropped_by_server=dropped,
        frames_unaccounted=max(0, len(samples) - len(answered) - dropped),
        bytes_sent=sum(s.sent_bytes for s in samples),
        bytes_received=sum(s.received_bytes or 0 for s in answered),
        duration_ms=_round(duration) or 0.0,
        answered_fps=(_round(len(answered) / (duration / 1000.0)) if duration > 0.0 else None),
        round_trip_ms=_distribution(
            s.received_at_ms - s.sent_at_ms for s in answered if s.received_at_ms is not None
        ),
        capture_to_present_ms=_distribution(
            s.presented_at_ms - s.sent_at_ms for s in answered if s.presented_at_ms is not None
        ),
        receive_to_present_ms=_distribution(
            s.presented_at_ms - s.received_at_ms
            for s in answered
            if s.presented_at_ms is not None and s.received_at_ms is not None
        ),
        decode_ms=_distribution(s.server.decode_ms for s in answered if s.server is not None),
        queue_wait_ms=_distribution(
            s.server.queue_wait_ms for s in answered if s.server is not None
        ),
        inference_ms=_distribution(s.server.inference_ms for s in answered if s.server is not None),
        serialize_ms=_distribution(
            s.server.serialize_previous_ms
            for s in answered
            if s.server is not None and s.server.serialize_previous_ms is not None
        ),
        people=_distribution(float(s.people) for s in answered if s.people is not None),
    )


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
        max=_round(ordered[-1]),
    )


def _median(ordered: Sequence[float]) -> float:
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _percentile(ordered: Sequence[float], percentile: int) -> float:
    """Nearest-rank, as every other summary here."""
    rank = max(1, -(-percentile * len(ordered) // 100))
    return ordered[min(rank, len(ordered)) - 1]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def report_path(artifacts_root: Path, report: LiveReport) -> Path:
    """Where a report is written. Named by when it started; never replaces one."""
    stamp = report.started_at.strftime("%Y%m%dT%H%M%S%f")
    return (
        artifacts_root
        / STAGE_NAME
        / f"{stamp}-{report.kind}-{report.mode}-{report.config.send_fps:g}fps{REPORT_SUFFIX}"
    )


def write_report(report: LiveReport, artifacts_root: Path) -> Path:
    path = report_path(artifacts_root, report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def read_reports(directory: Path) -> tuple[list[LiveReport], list[str]]:
    """Every readable report under `directory`, sorted by path, and what could not be read."""
    reports: list[LiveReport] = []
    unreadable: list[str] = []
    for path in sorted(directory.glob(f"*{REPORT_SUFFIX}")):
        try:
            reports.append(LiveReport.model_validate_json(path.read_text("utf-8")))
        except (OSError, ValueError) as error:
            unreadable.append(f"{path.name}: {error}")
    return reports, unreadable


__all__ = [
    "LIVE_REPORT_SCHEMA_VERSION",
    "REPORT_SUFFIX",
    "STAGE_NAME",
    "Distribution",
    "LiveConfig",
    "LiveEnvironment",
    "LiveReport",
    "LiveSample",
    "LiveSummary",
    "ServerTiming",
    "read_reports",
    "report_path",
    "summarise_live",
    "write_report",
]
