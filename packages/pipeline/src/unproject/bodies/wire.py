"""What one estimated frame looks like on the live socket.

Here rather than in the API package because two things have to serialise a
frame exactly the same way: the socket, which sends it, and the benchmark,
which measures what sending it costs (ADR-0005 WP2). One model, one builder,
and the number the benchmark reports is the number the viewer pays.

Pydantic on both ends of the process boundary, same as the artifacts. The
viewer parses these with a matching zod schema in `apps/web/src/live.ts`, so a
change here is a change there.

Joints go over as plain numbers rather than a binary frame. At the rates this
runs at - a few frames a second, a dozen people - that is on the order of
100 kB/s on localhost, and being able to read a message in the network tab is
worth more than the bytes. If it ever becomes the bottleneck the shape is
already right for a binary body.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from unproject.bodies.multihmr import FrameEstimate

# 2 adds `frame_id` and `timing` to the frame, both optional (ADR-0005 WP3).
WIRE_SCHEMA_VERSION = 2

# Enough for a rotation to a thousandth of a degree, and a third of the bytes
# a full float32 prints as. Bone transforms are most of the traffic.
WIRE_DECIMALS = 5


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LiveCamera(_Frozen):
    """The camera the positions are relative to, at source resolution."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0.0)
    fy: float = Field(gt=0.0)
    cx: float
    cy: float
    source: Literal["estimated", "assumed", "known"]


class LivePerson(_Frozen):
    track_id: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    # Flattened (bone, xyz), camera-relative OpenCV metres.
    joints: list[float]
    # Flattened (bone, 4, 4) row-major, root-relative. Empty unless the client
    # asked for bodies: it is four times the joints and most viewers draw
    # skeletons.
    bones: list[float] = Field(default_factory=list)


class FrameTiming(_Frozen):
    """What the server spent on this frame, on its own monotonic clock.

    Milliseconds, and durations only: never a timestamp, because a client's
    clock and the server's have different origins and a difference between
    them would measure the origins. `serialize_previous_ms` is the encoding of
    the previous message on this socket - a message cannot carry the time it
    took to encode itself.
    """

    decode_ms: float = Field(ge=0.0)
    queue_wait_ms: float = Field(ge=0.0)
    inference_ms: float = Field(ge=0.0)
    serialize_previous_ms: float | None = Field(default=None, ge=0.0)


class Frame(_Frozen):
    kind: Literal["frame"] = "frame"
    # Seconds along the source. For a camera this is time since the socket
    # opened, so the viewer has one timeline either way.
    time_seconds: float = Field(ge=0.0)
    camera: LiveCamera
    people: list[LivePerson]
    # How many frames the server skipped to keep up. Non-zero is the honest
    # signal that inference is behind the source, and the viewer shows it.
    dropped: int = Field(default=0, ge=0)
    # The client's own number for the frame this answers, echoed back so the
    # client can pair a response with what it sent on its own clock. A file
    # session numbers frames itself. None for a camera client that sent none.
    frame_id: int | None = Field(default=None, ge=0)
    timing: FrameTiming | None = None


def frame_message(
    estimate: FrameEstimate,
    *,
    time_seconds: float,
    size: tuple[int, int],
    dropped: int,
    bodies: bool = False,
    frame_id: int | None = None,
    timing: FrameTiming | None = None,
) -> Frame:
    """One estimate as the socket sends it."""
    focal_x, focal_y, centre_x, centre_y = estimate.intrinsics
    width, height = size
    return Frame(
        frame_id=frame_id,
        timing=timing,
        time_seconds=time_seconds,
        camera=LiveCamera(
            width=width,
            height=height,
            fx=focal_x,
            fy=focal_y,
            cx=centre_x,
            cy=centre_y,
            # The model predicts a field of view per frame; nothing is assumed.
            source="estimated",
        ),
        people=[
            LivePerson(
                track_id=person.track_id,
                confidence=person.confidence,
                joints=[float(value) for value in person.joints_3d.reshape(-1)],
                bones=(
                    [round(float(value), WIRE_DECIMALS) for value in person.bone_poses.reshape(-1)]
                    if bodies
                    else []
                ),
            )
            for person in estimate.people
        ],
        dropped=dropped,
    )


__all__ = [
    "WIRE_DECIMALS",
    "WIRE_SCHEMA_VERSION",
    "Frame",
    "FrameTiming",
    "LiveCamera",
    "LivePerson",
    "frame_message",
]
