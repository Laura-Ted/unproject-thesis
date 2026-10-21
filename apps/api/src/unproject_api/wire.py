"""What travels over the live socket.

The viewer mirrors these in `apps/web/src/live.ts` and, for `Topology`, in
`api.ts`, so a change here is a change there.

The frame message itself - `Frame`, `LiveCamera`, `LivePerson` - lives in
`unproject.bodies.wire`, shared with the benchmark (ADR-0005 WP2), and is
re-exported here. The messages that only the socket carries stay here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from unproject.bodies.artifacts import BodyModelInfo, MeshArray
from unproject.bodies.wire import (
    WIRE_SCHEMA_VERSION,
    Frame,
    FrameTiming,
    LiveCamera,
    LivePerson,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StartVideo(_Frozen):
    """Play a file already on the server and estimate as it goes."""

    kind: Literal["video"]
    # Relative to the served roots, e.g. `datasets/crowd/clip.mp4` or an
    # `uploads/<id>.mp4` handed back by the upload route.
    path: str
    # See `StartCamera.bodies`.
    bodies: bool = False
    # Every nth frame. The estimator runs at a few frames a second, so stepping
    # over frames is how a clip finishes in something like its own duration.
    stride: int = Field(default=2, ge=1)


class StartCamera(_Frozen):
    """Estimate frames the browser pushes, as fast as they can be taken."""

    kind: Literal["camera"]
    # Whether to send bone transforms and a rest body per track. Off by default:
    # it is most of the traffic and only a viewer drawing meshes needs it.
    bodies: bool = False
    # When true, every binary frame the client sends starts with a 4-byte
    # little-endian frame id that the server echoes back in `Frame.frame_id`,
    # so the client can pair responses with sends on its own clock (ADR-0005
    # WP3). Off by default: a bare JPEG stays a valid frame.
    frame_ids: bool = False


class ServerEnvironment(_Frozen):
    """Where the estimator is running, for a client writing a report."""

    machine: str
    device: str
    git_commit: str | None = None
    git_dirty: bool | None = None


class Topology(_Frozen):
    """The constant half of a body: what it is made of, and what pulls on it.

    The same for every person and every frame, so the viewer fetches it once
    over HTTP rather than being sent it down the socket.
    """

    vertex_count: int = Field(gt=0)
    face_count: int = Field(gt=0)
    influences: int = Field(gt=0)
    bone_count: int = Field(gt=0)
    # Byte layout of `/api/body/topology.bin`, in order.
    arrays: list[MeshArray]


class Ready(_Frozen):
    """Sent once the model is loaded and the source is open."""

    kind: Literal["ready"] = "ready"
    schema_version: int = WIRE_SCHEMA_VERSION
    convention: str
    body_model: BodyModelInfo
    source: str
    # None for a camera, which has no end.
    duration_seconds: float | None = None
    environment: ServerEnvironment | None = None


class Done(_Frozen):
    kind: Literal["done"] = "done"
    frames: int = Field(ge=0)


class Failed(_Frozen):
    kind: Literal["error"] = "error"
    message: str


__all__ = [
    "WIRE_SCHEMA_VERSION",
    "Done",
    "Failed",
    "Frame",
    "FrameTiming",
    "LiveCamera",
    "LivePerson",
    "Ready",
    "ServerEnvironment",
    "StartCamera",
    "StartVideo",
    "Topology",
]
