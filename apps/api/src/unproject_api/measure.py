"""The transport benchmark: a client that pushes frames over the real socket.

ADR-0005 WP3 asks for the live path to be measured apart from inference, with
a repeatable, authorised source, at more than one send rate, and without the
browser where the browser cannot be automated reliably. This is that client. It
plays a public clip into the product's own camera endpoint - the same
`/api/live` socket, the same `kind: "camera"` request, the same JPEG frames the
viewer sends - at a controlled rate, numbers every frame, and pairs each
answer with its send on one monotonic clock.

What it measures is the **transport and server** side of the live path:
capture-to-response round trip, the server's own decode, queue wait, inference
and serialisation, what was dropped by the latest-only mailbox, and the bytes
each way. What it does not measure is rendering: the browser report adds that,
separately, and the two are never joined into one number.

Frames are JPEG-encoded up front, resized to the viewer's own maximum width and
quality, so the send loop pays nothing but the send. The browser pays a canvas
encode per frame as well; that cost is the browser's and is in its report.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import platform
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unproject.benchmark.live import (
    LiveConfig,
    LiveEnvironment,
    LiveReport,
    LiveSample,
    summarise_live,
    write_report,
)
from unproject.bodies.wire import Frame, FrameTiming
from unproject_api.wire import Ready

logger = logging.getLogger("unproject_api.measure")

# The viewer's own capture settings, so the frames are the frames it sends.
DEFAULT_MAX_WIDTH = 1280
DEFAULT_JPEG_QUALITY = 0.75
# How long to wait for answers after the last frame was sent.
DEFAULT_GRACE_SECONDS = 15.0


@dataclass(slots=True)
class _Sent:
    frame_id: int
    sent_at: float
    sent_bytes: int
    received_at: float | None = None
    received_bytes: int | None = None
    people: int | None = None
    dropped_before: int | None = None
    timing: FrameTiming | None = None


@dataclass(slots=True)
class _Session:
    """Everything one measured session accumulates."""

    sent: dict[int, _Sent] = field(default_factory=dict)
    ready: Ready | None = None
    errors: list[str] = field(default_factory=list)


def encode_frames(
    path: Path, *, max_width: int, quality: float, limit: int | None
) -> tuple[list[bytes], tuple[int, int], str]:
    """Every frame of the clip as the viewer would send it: resized, JPEG.

    Returns the payloads, the sent frame size, and the clip's sha256.
    """
    import cv2

    from unproject.ingest.video import VideoFileSource

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)

    payloads: list[bytes] = []
    size: tuple[int, int] | None = None
    for index, frame in enumerate(VideoFileSource(path).frames()):
        if limit is not None and index >= limit:
            break
        bgr: NDArray[np.uint8] = frame.image
        height, width = bgr.shape[:2]
        if width > max_width:
            scale = max_width / width
            bgr = np.asarray(
                cv2.resize(bgr, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA),
                dtype=np.uint8,
            )
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), round(quality * 100)]
        )
        if not ok:
            raise RuntimeError(f"could not encode frame {index} of {path}")
        payloads.append(bytes(encoded.tobytes()))
        size = (int(bgr.shape[1]), int(bgr.shape[0]))
    if not payloads or size is None:
        raise RuntimeError(f"{path} yielded no frames")
    return payloads, size, digest.hexdigest()


async def run_session(
    url: str,
    payloads: list[bytes],
    *,
    fps: float,
    bodies: bool,
    grace_seconds: float,
) -> tuple[_Session, float]:
    """Send every payload at `fps` and collect what comes back.

    Sends are scheduled on absolute times from the session start, so a slow
    send does not push the whole schedule back. The socket is read
    concurrently, and the session ends `grace_seconds` after the last send or
    when the server closes, whichever comes first. Returns the session and
    the monotonic start it is relative to.
    """
    from websockets.asyncio.client import connect

    session = _Session()
    async with connect(url, max_size=None) as socket:
        await socket.send(json.dumps({"kind": "camera", "bodies": bodies, "frame_ids": True}))
        started = time.perf_counter()
        finished_sending = asyncio.Event()

        async def send_all() -> None:
            try:
                for frame_id, payload in enumerate(payloads):
                    due = started + frame_id / fps
                    delay = due - time.perf_counter()
                    if delay > 0.0:
                        await asyncio.sleep(delay)
                    message = struct.pack("<I", frame_id) + payload
                    sent_at = time.perf_counter()
                    await socket.send(message)
                    session.sent[frame_id] = _Sent(
                        frame_id=frame_id, sent_at=sent_at - started, sent_bytes=len(message)
                    )
            finally:
                finished_sending.set()

        async def receive_all() -> None:
            async for raw in socket:
                received_at = time.perf_counter() - started
                if isinstance(raw, bytes):
                    # A rest body. It answers no frame, and its bytes are not counted.
                    continue
                data = json.loads(raw)
                kind = data.get("kind")
                if kind == "ready":
                    session.ready = Ready.model_validate(data)
                    continue
                if kind == "error":
                    session.errors.append(str(data.get("message")))
                    return
                if kind != "frame":
                    continue
                frame = Frame.model_validate(data)
                if frame.frame_id is None or frame.frame_id not in session.sent:
                    session.errors.append(f"an answer arrived for unknown frame {frame.frame_id}")
                    continue
                record = session.sent[frame.frame_id]
                record.received_at = received_at
                record.received_bytes = len(raw.encode("utf-8"))
                record.people = len(frame.people)
                record.dropped_before = frame.dropped
                record.timing = frame.timing

        sender = asyncio.create_task(send_all())
        receiver = asyncio.create_task(receive_all())
        await finished_sending.wait()
        # Answers to the last frames are still coming; give them the grace
        # period, then stop waiting rather than hang on a server that has
        # nothing more to say.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(receiver), timeout=grace_seconds)
        await sender
        receiver.cancel()
        # The socket is closing; whatever the receiver was waiting on is gone.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await receiver
    return session, started


def build_report(
    session: _Session,
    *,
    started_at: datetime,
    source: str,
    source_sha256: str,
    frame_size: tuple[int, int],
    fps: float,
    bodies: bool,
    max_width: int,
    quality: float,
    limit: int | None,
) -> LiveReport:
    samples = [
        LiveSample(
            frame_id=record.frame_id,
            sent_at_ms=record.sent_at * 1000.0,
            sent_bytes=record.sent_bytes,
            received_at_ms=None if record.received_at is None else record.received_at * 1000.0,
            received_bytes=record.received_bytes,
            people=record.people,
            dropped_before=record.dropped_before,
            server=record.timing,
        )
        for record in sorted(session.sent.values(), key=lambda r: r.frame_id)
    ]
    environment = session.ready.environment if session.ready is not None else None
    return LiveReport(
        kind="transport",
        started_at=started_at,
        source=source,
        source_sha256=source_sha256,
        mode="mesh" if bodies else "skeleton",
        frame_width=frame_size[0],
        frame_height=frame_size[1],
        environment=LiveEnvironment(
            client_platform=f"python {platform.python_version()} on {platform.platform()}",
            server_machine=None if environment is None else environment.machine,
            server_device=None if environment is None else environment.device,
            git_commit=None if environment is None else environment.git_commit,
            git_dirty=None if environment is None else environment.git_dirty,
        ),
        config=LiveConfig(
            send_fps=fps,
            max_width=max_width,
            jpeg_quality=quality,
            bodies=bodies,
            frame_limit=limit,
        ),
        samples=samples,
        summary=summarise_live(samples),
        notes=[
            "frames were JPEG-encoded before the session started; the send loop pays only "
            "the send, unlike the browser, which encodes a canvas per frame",
            *session.errors,
        ],
    )


def wait_for_server(base_url: str, *, timeout_seconds: float) -> None:
    """Poll the run list until the server answers, or give up."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/runs", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"no server at {base_url} after {timeout_seconds:.0f} s")


def spawn_server(*, port: int, root: Path, config: Path) -> subprocess.Popen[bytes]:
    """Start the API in a subprocess, viewer build skipped, on the given port."""
    command = [
        sys.executable,
        "-m",
        "unproject_api.cli",
        "--no-build",
        "--port",
        str(port),
        "--root",
        str(root),
        "--config",
        str(config),
    ]
    return subprocess.Popen(command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unproject-live-benchmark", description=__doc__)
    parser.add_argument("clip", type=Path, help="a video file to play into the camera endpoint")
    parser.add_argument("--fps", type=float, required=True, help="send rate, frames per second")
    parser.add_argument(
        "--bodies", action="store_true", help="ask for meshes rather than skeletons"
    )
    parser.add_argument("--frames", type=int, default=None, help="send at most this many frames")
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH)
    parser.add_argument("--quality", type=float, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--grace", type=float, default=DEFAULT_GRACE_SECONDS)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="a running server")
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="start the server yourself on the port in --url, and stop it afterwards",
    )
    parser.add_argument("--root", type=Path, default=Path(), help="repository root")
    parser.add_argument("--config", type=Path, default=Path("experiments/crowd.toml"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    payloads, size, digest = encode_frames(
        args.clip, max_width=args.max_width, quality=args.quality, limit=args.frames
    )
    logger.info("%d frame(s) of %dx%d encoded from %s", len(payloads), size[0], size[1], args.clip)

    server: subprocess.Popen[bytes] | None = None
    if args.spawn:
        port = int(args.url.rsplit(":", 1)[-1])
        server = spawn_server(port=port, root=args.root, config=args.config)
    try:
        wait_for_server(args.url, timeout_seconds=120.0)
        started_at = datetime.now(UTC)
        socket_url = args.url.replace("http://", "ws://").replace("https://", "wss://")
        session, _ = asyncio.run(
            run_session(
                f"{socket_url}/api/live",
                payloads,
                fps=args.fps,
                bodies=args.bodies,
                grace_seconds=args.grace,
            )
        )
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()

    report = build_report(
        session,
        started_at=started_at,
        source=args.clip.as_posix(),
        source_sha256=digest,
        frame_size=size,
        fps=args.fps,
        bodies=args.bodies,
        max_width=args.max_width,
        quality=args.quality,
        limit=args.frames,
    )
    path = write_report(report, args.artifacts_root)
    summary = report.summary
    print(f"wrote {path}")
    print(
        f"sent {summary.frames_sent} at {args.fps:g} fps, answered {summary.frames_answered} "
        f"({summary.answered_fps or 0.0:.2f} fps), dropped by the server "
        f"{summary.frames_dropped_by_server}, unaccounted {summary.frames_unaccounted}"
    )
    rt = summary.round_trip_ms
    inf = summary.inference_ms
    if rt.median is not None and inf.median is not None:
        print(
            f"round trip median {rt.median:.0f} ms, p95 {rt.p95:.0f} ms; server inference "
            f"median {inf.median:.0f} ms, queue wait median "
            f"{summary.queue_wait_ms.median or 0.0:.0f} ms"
        )
    for note in session.errors:
        print(f"note: {note}")
    return 0 if not session.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_report", "encode_frames", "main", "run_session", "wait_for_server"]
