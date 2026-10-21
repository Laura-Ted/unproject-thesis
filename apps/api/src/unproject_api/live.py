"""Running the estimator while someone watches.

Inference is slower than any source worth watching - a few frames a second
against a camera's thirty - and the two sources want opposite things about that.

A **camera** has no past worth keeping: the operator wants to see now, so the
newest frame wins and the backlog is thrown away. The count of what was thrown
goes back over the socket, because a viewer that silently shows a scene from two
minutes ago is worse than one that admits it is behind.

A **file** is not live and has nothing to lose. Every frame it offers gets
estimated, in order, and the clip takes as long as it takes. Dropping here would
quietly skip content the operator asked to see.

Inference also blocks, so it runs in a worker thread. That is what lets a camera
frame actually be dropped: the event loop stays free to receive it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from unproject.bodies.artifacts import CONVENTION, MeshArray
from unproject.bodies.multihmr import EstimatorSession, FrameEstimate, MultiHmrEstimator
from unproject.bodies.wire import Frame, FrameTiming, frame_message
from unproject.config import BodiesConfig
from unproject_api.wire import Ready, ServerEnvironment, Topology

logger = logging.getLogger(__name__)

RgbFrame = NDArray[np.uint8]

_T = TypeVar("_T")

# One session holds the model thread while a second waits its turn. A third
# would be queueing behind work whose end it cannot see, so it is turned away
# with an answer instead of joining an invisible backlog.
MAX_LIVE_SESSIONS = 2


@dataclass(frozen=True, slots=True)
class Timed:
    """A frame and where it sits on its source's own timeline.

    Also what it cost to get here: how long it took to decode, and when it was
    handed to the consumer, on the server's monotonic clock, so the time it
    then waits for the model can be measured on that same clock.
    """

    time_seconds: float
    image: RgbFrame
    # The client's number for the frame, echoed back; None when it sent none.
    frame_id: int | None = None
    decode_seconds: float = 0.0
    # `time.perf_counter()` at hand-over; the wait is measured from here.
    queued_at: float = 0.0


class Estimator:
    """The loaded model, shared by every socket.

    Loading costs about twenty seconds, so it happens once. The lock is there
    because a session opening on the model thread and `topology`, which the
    HTTP routes call from a request thread, must not both build it.

    Everything that touches the model runs on **one** thread, through the
    executor below. Metal keeps a command buffer per thread and asserts if work
    arrives on another while one is open - which is what `asyncio.to_thread`
    does, since its pool hands out whichever thread is free.
    """

    def __init__(self, config: BodiesConfig, *, config_dir: Path) -> None:
        self._estimator = MultiHmrEstimator(config, config_dir=config_dir)
        self._config = config
        self._lock = threading.Lock()
        self._topology: tuple[Topology, bytes] | None = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unproject-model")
        self._live_sessions = 0

    async def run(self, work: Callable[[], _T]) -> _T:
        """Do something with the model, off the event loop and on its thread."""
        return await asyncio.get_running_loop().run_in_executor(self._worker, work)

    def claim_session(self) -> bool:
        """Take a live-session slot, or report that they are all taken.

        A plain counter rather than a lock: every caller is a request handler on
        the one event loop thread, and neither branch here awaits.
        """
        if self._live_sessions >= MAX_LIVE_SESSIONS:
            return False
        self._live_sessions += 1
        return True

    def release_session(self) -> None:
        self._live_sessions = max(0, self._live_sessions - 1)

    @property
    def live_sessions(self) -> int:
        return self._live_sessions

    def close(self) -> None:
        """Stop the model thread and drop work nobody is waiting for.

        Waits, because the point is that the process does not exit while a
        frame is still going through Metal.
        """
        self._worker.shutdown(wait=True, cancel_futures=True)

    @property
    def config(self) -> BodiesConfig:
        return self._config

    def open_session(self) -> EstimatorSession:
        """A session with its own tracker, over the shared model."""
        with self._lock:
            return self._estimator.open()

    def topology(self) -> tuple[Topology, bytes]:
        """What a body is made of, built once and shared.

        Constant for the model, so every viewer gets the same bytes and they are
        worth keeping rather than rebuilding per request.
        """
        with self._lock:
            if self._topology is None:
                session = self._estimator.open()
                self._topology = topology(session, session.body_model.bone_count)
            return self._topology


def decode_jpeg(payload: bytes) -> RgbFrame:
    """Turn a frame from the browser into what the model expects.

    OpenCV decodes to BGR and the model was trained on RGB. Getting this the
    wrong way round raises nothing and looks like nothing: the figures still
    appear, a little worse.
    """
    import cv2

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("could not decode the frame")
    return np.ascontiguousarray(decoded[:, :, ::-1])


def video_frames(path: Path, stride: int) -> Iterator[Timed]:
    """Frames from a file on disk, in order, at the requested stride.

    A file session numbers its own frames, from zero, over the frames it
    keeps: the number is what a client reads a response by, so it counts what
    was sent rather than what was decoded and skipped.
    """
    from unproject.ingest.video import VideoFileSource

    source = iter(VideoFileSource(path).frames())
    index = 0
    sent = 0
    while True:
        began = time.perf_counter()
        try:
            frame = next(source)
        except StopIteration:
            return
        decode = time.perf_counter() - began
        if index % stride == 0:
            # Same BGR to RGB conversion, same reason, as `decode_jpeg`.
            yield Timed(
                frame.timestamp_seconds,
                np.ascontiguousarray(frame.image[:, :, ::-1]),
                frame_id=sent,
                decode_seconds=decode,
            )
            sent += 1
        index += 1


# How long the decode thread may wait before it looks at `stop` again. Short
# enough that a closed tab does not leave a thread sitting on a video file.
_STOP_CHECK_SECONDS = 0.1

# Decoded frames allowed to wait for the estimator. Two is enough to keep it
# fed; more is memory spent on frames the model will not reach for seconds.
_READ_AHEAD_FRAMES = 2


def video_duration(path: Path) -> float | None:
    from unproject.ingest.video import VideoFileSource

    return VideoFileSource(path).metadata().duration_seconds


class LatestOnly:
    """A one-slot mailbox: a new frame replaces whatever was waiting.

    This is where a camera's backlog is thrown away, and `dropped` is what the
    viewer is told about it.
    """

    def __init__(self) -> None:
        self._value: Timed | None = None
        self._ready = asyncio.Event()
        self._closed = False
        self.dropped = 0

    def put(self, item: Timed) -> None:
        if self._value is not None:
            self.dropped += 1
        # Stamped here, not by the producer: the wait a frame is charged with
        # is the time it spent in this slot.
        self._value = replace(item, queued_at=time.perf_counter())
        self._ready.set()

    def close(self) -> None:
        self._closed = True
        self._ready.set()

    async def get(self) -> Timed | None:
        """The newest frame, or None once closed and drained."""
        while self._value is None:
            if self._closed:
                return None
            await self._ready.wait()
            self._ready.clear()
        item, self._value = self._value, None
        return item

    async def stream(self) -> AsyncIterator[Timed]:
        while (item := await self.get()) is not None:
            yield item


# The frame message is built by the pipeline's own serializer, so the benchmark
# measures exactly what the socket sends. The old name stays as an alias:
# `estimate_one` calls it, and docs/notes/2026-09-15-adr0005-wp2-protocol-v2.md
# cites it.
to_wire = frame_message


def ready(session: EstimatorSession, *, source: str, duration_seconds: float | None) -> Ready:
    return Ready(
        convention=CONVENTION,
        body_model=session.body_model,
        source=source,
        duration_seconds=duration_seconds,
        environment=server_environment(),
    )


def server_environment() -> ServerEnvironment:
    """The machine and the code behind the socket, for a client's report."""
    from unproject.benchmark.run import environment

    described = environment()
    return ServerEnvironment(
        machine=described.machine,
        device=described.device,
        git_commit=described.git_commit,
        git_dirty=described.git_dirty,
    )


async def estimate_one(
    estimator: Estimator,
    session: EstimatorSession,
    config: BodiesConfig,
    timed: Timed,
    *,
    dropped: int,
    bodies: bool = False,
    serialize_previous_seconds: float | None = None,
) -> tuple[Frame, FrameEstimate]:
    """One frame through the model, off the event loop.

    The estimate comes back alongside the message because a caller sending
    bodies needs the shape of anyone it has not seen before, and that is not on
    the wire.

    The message carries what the frame cost on this side - decode, the wait in
    the mailbox, inference, and the previous message's serialisation - as
    durations on this process's monotonic clock, never as timestamps.
    """
    queue_wait = max(0.0, time.perf_counter() - timed.queued_at) if timed.queued_at else 0.0
    height, width = timed.image.shape[:2]
    began = time.perf_counter()
    estimate = await estimator.run(
        lambda: session.frame(timed.image, config, width=width, height=height)
    )
    inference = time.perf_counter() - began
    frame = to_wire(
        estimate,
        time_seconds=timed.time_seconds,
        size=(width, height),
        dropped=dropped,
        bodies=bodies,
        frame_id=timed.frame_id,
        timing=FrameTiming(
            decode_ms=timed.decode_seconds * 1000.0,
            queue_wait_ms=queue_wait * 1000.0,
            inference_ms=inference * 1000.0,
            serialize_previous_ms=(
                None if serialize_previous_seconds is None else serialize_previous_seconds * 1000.0
            ),
        ),
    )
    return frame, estimate


def encode_frame(frame: Frame) -> tuple[str, float]:
    """The message text, and how long encoding it took. Reported on the next one."""
    began = time.perf_counter()
    text = frame.model_dump_json()
    return text, time.perf_counter() - began


# The binary camera message when the client numbers its frames: a 4-byte
# little-endian id, then the JPEG.
FRAME_ID_BYTES = 4


def split_frame_id(payload: bytes) -> tuple[int, bytes]:
    """Take the client's frame id off the front of a camera message."""
    if len(payload) < FRAME_ID_BYTES:
        raise ValueError("a numbered frame arrived without its id")
    (frame_id,) = struct.unpack("<I", payload[:FRAME_ID_BYTES])
    return int(frame_id), payload[FRAME_ID_BYTES:]


def topology(session: EstimatorSession, bone_count: int) -> tuple[Topology, bytes]:
    """The constant half of a body, and the bytes it describes."""
    built = session.topology()
    arrays: list[MeshArray] = []
    payload: list[bytes] = []
    offset = 0
    for name, block, dtype in (
        ("faces", built.faces, "<u2"),
        ("skin_indices", built.skin_indices, "<u2"),
        ("skin_weights", built.skin_weights, "<f4"),
    ):
        raw = np.ascontiguousarray(block).tobytes()
        arrays.append(
            MeshArray(
                name=name,
                dtype=cast(Literal["<f4", "<u2"], dtype),
                shape=tuple(int(size) for size in block.shape),
                byte_offset=offset,
                byte_length=len(raw),
            )
        )
        payload.append(raw)
        offset += len(raw)

    return (
        Topology(
            vertex_count=int(built.skin_weights.shape[0]),
            face_count=int(built.faces.shape[0]),
            influences=int(built.skin_weights.shape[1]),
            bone_count=bone_count,
            arrays=arrays,
        ),
        b"".join(payload),
    )


def rest_body(session: EstimatorSession, track: int, shape: NDArray[np.float32]) -> bytes:
    """One track's unposed body, as a binary socket message.

    A track id, then the rest vertices, then the bind bones. Binary because it
    is a fifth of a megabyte and printing that as JSON numbers would cost five
    times as much for nothing.
    """
    kept = session.topology().kept_vertices
    vertices = session.rest_vertices(shape)[kept].astype(np.float32)
    bones = session.rest_bone_poses(shape).astype(np.float32)
    return struct.pack("<I", track) + vertices.tobytes() + bones.tobytes()


async def read_video(path: Path, stride: int) -> AsyncIterator[Timed]:
    """Frames from a file, one at a time, without holding the clip in memory.

    A minute of 720p is several gigabytes of decoded pixels.

    The decode runs on a thread, and a thread cannot be cancelled, so it is
    asked to stop instead: `stop` is checked between frames and while waiting
    for room. Backpressure is the semaphore rather than the queue, because a
    thread blocked on a queue nobody drains never comes back, and keeps the
    decoder open after the viewer has closed the tab.
    """
    queue: asyncio.Queue[Timed | None] = asyncio.Queue()
    slots = threading.Semaphore(_READ_AHEAD_FRAMES)
    stop = threading.Event()
    loop = asyncio.get_running_loop()

    def hand_over(item: Timed | None) -> None:
        # The loop is gone during interpreter shutdown; the frame has nowhere to
        # go and nobody to tell.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, item)

    def read() -> None:
        try:
            for timed in video_frames(path, stride):
                while not slots.acquire(timeout=_STOP_CHECK_SECONDS):
                    if stop.is_set():
                        return
                if stop.is_set():
                    return
                # Stamped at hand-over: the wait a file frame is charged with is
                # the time it sits in the queue before the model takes it.
                hand_over(replace(timed, queued_at=time.perf_counter()))
        finally:
            hand_over(None)

    reader = asyncio.create_task(asyncio.to_thread(read))
    try:
        while (item := await queue.get()) is not None:
            yield item
            slots.release()
    finally:
        stop.set()
        # Frees a producer parked on a full buffer, so it reaches its own stop
        # check rather than waiting for a consumer that has gone.
        slots.release()
        # Awaited, not just cancelled: this is what makes a decode failure the
        # session's problem instead of a warning nobody sees.
        with contextlib.suppress(asyncio.CancelledError):
            await reader


def elapsed_since(start: float) -> float:
    """Seconds since the socket opened, which is a camera's only timeline."""
    return max(0.0, time.monotonic() - start)
