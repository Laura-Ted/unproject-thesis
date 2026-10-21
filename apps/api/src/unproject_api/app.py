"""Stage [6]: serves what the pipeline produced, and runs the estimator live.

One process owns the model, so the viewer is a browser tab rather than a second
copy of the inference stack. The dev server proxies here rather than reading the
artifact tree itself, which keeps one implementation of what a run is.

This process must never import pycolmap. It reaches torch through the bodies
stage, and the two abort each other; see docs/reproduce.md §6.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from unproject.benchmark.live import (
    LiveConfig,
    LiveEnvironment,
    LiveMode,
    LiveReport,
    LiveSample,
    ReportKind,
    report_path,
    summarise_live,
    write_report,
)
from unproject.bodies.artifacts import ARTIFACT_FILENAME, BodiesArtifact
from unproject.bodies.multihmr import BodyEstimationError, EstimatorSession, FrameEstimate
from unproject.config import BodiesConfig, load_config
from unproject_api import live, wire
from unproject_api.wire import Failed, StartCamera, StartVideo

logger = logging.getLogger(__name__)

# Anything larger is a mistake rather than a clip worth watching, and the file
# is written to disk before it is validated.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

# What a remote client is told. The specifics stay in the local log: a stack
# trace, a decode error or a path on this disk is not the browser's business.
SESSION_FAILED = "the live session stopped; see the server log"
SESSIONS_BUSY = "the estimator is busy with another session; try again shortly"

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Where things live. Everything is resolved once, at startup."""

    root: Path
    config_path: Path
    bodies: BodiesConfig

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    @property
    def web(self) -> Path:
        return self.root / "apps" / "web" / "dist"


class RunSummary(BaseModel):
    """A bodies run on disk, as the viewer's list shows it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    experiment: str
    hash: str
    frames: int = Field(ge=0)
    people: int = Field(ge=0)
    max_people: int = Field(ge=0)
    tracks: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    # Where the footage this was fitted to lives, so the viewer can play it
    # beside the reconstruction. None when it is no longer on disk.
    video: str | None = None


class RunDeleted(BaseModel):
    """What was removed, echoed back so the viewer can drop it from the list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str


class UploadAccepted(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    name: str


class LiveReportDraft(BaseModel):
    """A live measurement as a client submits it: rows and context, no summary.

    The summary is computed here, by the one implementation in
    `unproject.benchmark.live`, so a browser and the Python client cannot
    disagree about a percentile. ADR-0005 WP3.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ReportKind
    started_at: datetime
    source: str
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mode: LiveMode
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    environment: LiveEnvironment
    config: LiveConfig
    samples: list[LiveSample]
    notes: list[str] = Field(default_factory=list)


class LiveReportAccepted(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    frames_sent: int
    frames_answered: int


def source_video(settings: Settings, experiment: str, ingest_hash: str) -> str | None:
    """The clip a bodies run was fitted to, as a URL the viewer can play.

    Read back through the ingest artifact rather than stored again: it already
    records the source it consumed, and duplicating that would let the two
    disagree.
    """
    ingest = settings.artifacts / experiment / "ingest" / ingest_hash / "ingest.json"
    try:
        recorded = json.loads(ingest.read_text("utf-8"))["source"]["uri"]
    except (OSError, KeyError, ValueError):
        return None

    # Recorded relative to the config file that declared it.
    resolved = (settings.config_path.parent / str(recorded)).resolve()
    try:
        relative = resolved.relative_to(settings.datasets.resolve())
    except ValueError:
        return None
    return f"/datasets/{relative.as_posix()}" if resolved.is_file() else None


def collect_runs(settings: Settings) -> list[RunSummary]:
    """Every bodies run the viewer can open, newest first."""
    if not settings.artifacts.is_dir():
        return []

    found: list[tuple[float, RunSummary]] = []
    for experiment in sorted(settings.artifacts.iterdir()):
        stage = experiment / "bodies"
        if not stage.is_dir():
            continue
        for run in sorted(stage.iterdir()):
            artifact_path = run / ARTIFACT_FILENAME
            if not artifact_path.is_file():
                continue
            try:
                artifact = BodiesArtifact.model_validate_json(artifact_path.read_text("utf-8"))
            except (OSError, ValueError) as error:
                # A run this build cannot read should not take the list down
                # with it; it just does not appear.
                logger.info("skipping %s: %s", artifact_path, error)
                continue
            last = artifact.frames[-1] if artifact.frames else None
            found.append(
                (
                    artifact_path.stat().st_mtime,
                    RunSummary(
                        path=f"{experiment.name}/bodies/{run.name}",
                        experiment=experiment.name,
                        hash=run.name,
                        frames=artifact.summary.frames_total,
                        people=artifact.summary.people_total,
                        max_people=artifact.summary.max_people_in_frame,
                        tracks=artifact.summary.track_count,
                        duration_seconds=last.timestamp_seconds if last else 0.0,
                        video=source_video(settings, experiment.name, artifact.ingest_hash),
                    ),
                )
            )
    return [summary for _, summary in sorted(found, key=lambda pair: pair[0], reverse=True)]


def resolve_playable(settings: Settings, path: str) -> Path:
    """A video path from a client, resolved inside the roots we serve.

    The path is untrusted, so anything that leaves the two served directories
    is refused.
    """
    cleaned = path.lstrip("/")
    root, _, rest = cleaned.partition("/")
    bases = {"datasets": settings.datasets, "uploads": settings.uploads}
    base = bases.get(root)
    if base is None or not rest:
        raise ValueError(f"{path} is not under a served directory")

    target = (base / rest).resolve()
    if not target.is_relative_to(base.resolve()) or not target.is_file():
        raise ValueError(f"{path} is not a file this server offers")
    if target.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"{target.name} is not a video this server will decode")
    return target


def resolve_run(settings: Settings, path: str) -> Path:
    """A run directory named by a client, resolved inside the artifact tree.

    Deleting is the one destructive thing this server does, so the checks are
    deliberately narrow. Three of them, and all three have to hold:

    - it resolves inside `artifacts/`, so `../` buys nothing;
    - it is a directory;
    - it holds a `bodies.json`, so this can only ever remove a finished bodies
      run and never an arbitrary directory that happens to sit in the tree.

    ADR-0002 §10.5 forbids deleting artifacts without the user asking; these
    checks make sure the delete button can only remove what it names.
    """
    root = settings.artifacts.resolve()
    target = (settings.artifacts / path.lstrip("/")).resolve()

    if not target.is_relative_to(root) or target == root:
        raise ValueError(f"{path} is not inside the artifact tree")
    if not target.is_dir():
        raise ValueError(f"{path} is not a directory")
    if not (target / ARTIFACT_FILENAME).is_file():
        raise ValueError(f"{path} does not hold a {ARTIFACT_FILENAME}, so it is not a run")
    return target


def _mount_if_present(app: FastAPI, url: str, directory: Path) -> None:
    """Serve a directory of ours, creating it first if it is missing.

    `StaticFiles` handles range requests, which is what lets the viewer pull one
    array out of a sidecar instead of the whole file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    app.mount(url, StaticFiles(directory=directory), name=url.strip("/"))


def create_app(settings: Settings) -> FastAPI:
    estimator = live.Estimator(settings.bodies, config_dir=settings.config_path.parent)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        # On a thread, because closing waits for the frame that is in the model
        # right now and the event loop should not be the thing waiting.
        await asyncio.to_thread(estimator.close)
        logger.info("the model worker has stopped")

    app = FastAPI(title="unproject", version="0.1.0", lifespan=lifespan)
    router = APIRouter(prefix="/api")

    @router.get("/runs")
    def runs() -> list[RunSummary]:
        return collect_runs(settings)

    @router.delete("/runs/{path:path}")
    def delete_run(path: str) -> RunDeleted:
        """Remove one finished run from disk. Irreversible, and the artifact is
        the evidence behind any number drawn from it, so the viewer asks first.
        """
        try:
            target = resolve_run(settings, path)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        try:
            shutil.rmtree(target)
        except OSError as error:
            logger.exception("could not remove %s", target)
            raise HTTPException(status_code=500, detail="could not remove the run") from error

        logger.info("removed run %s at the operator's request", path)
        return RunDeleted(path=path)

    @router.post("/uploads")
    async def upload(file: Annotated[UploadFile, File()]) -> UploadAccepted:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(415, f"{suffix or 'that'} is not a video format this server reads")

        settings.uploads.mkdir(parents=True, exist_ok=True)
        target = settings.uploads / f"{uuid.uuid4().hex}{suffix}"
        written = 0
        try:
            with target.open("wb") as handle:
                while chunk := await file.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "the upload is larger than this server accepts")
                    handle.write(chunk)
        except HTTPException:
            target.unlink(missing_ok=True)
            raise
        return UploadAccepted(path=f"uploads/{target.name}", name=file.filename or target.name)

    @router.post("/live-metrics")
    def live_metrics(draft: LiveReportDraft) -> LiveReportAccepted:
        """Keep a live measurement a client took, under artifacts/live/.

        Never replaces one: a report is named by when it started, and a second
        submission of the same session is refused rather than overwritten.
        """
        report = LiveReport(
            kind=draft.kind,
            started_at=draft.started_at,
            source=draft.source,
            source_sha256=draft.source_sha256,
            mode=draft.mode,
            frame_width=draft.frame_width,
            frame_height=draft.frame_height,
            environment=draft.environment,
            config=draft.config,
            samples=draft.samples,
            summary=summarise_live(draft.samples),
            notes=draft.notes,
        )
        target = report_path(settings.artifacts, report)
        if target.exists():
            raise HTTPException(409, f"{target.name} already exists; a report is never replaced")
        written = write_report(report, settings.artifacts)
        logger.info("kept a %s live report at %s", report.kind, written)
        return LiveReportAccepted(
            path=written.relative_to(settings.root).as_posix(),
            frames_sent=report.summary.frames_sent,
            frames_answered=report.summary.frames_answered,
        )

    @router.get("/body/topology")
    def topology() -> wire.Topology:
        """What a body is made of. Constant, so the viewer fetches it once."""
        described, _ = _built_topology(estimator)
        return described

    @router.get("/body/topology.bin")
    def topology_bytes() -> Response:
        _, payload = _built_topology(estimator)
        return Response(payload, media_type="application/octet-stream")

    @router.websocket("/live")
    async def live_socket(socket: WebSocket) -> None:
        await socket.accept()
        if not estimator.claim_session():
            logger.info("refused a live session; %d already open", estimator.live_sessions)
            await socket.send_text(Failed(message=SESSIONS_BUSY).model_dump_json())
            await socket.close()
            return
        try:
            await _serve_live(socket, settings, estimator)
        except WebSocketDisconnect:
            logger.info("live socket closed by the client")
        except BodyEstimationError as error:
            # This one the client can act on: it names the extra to install and
            # says nothing about this machine. Everything else stays generic.
            logger.info("live session refused: %s", error)
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await socket.send_text(Failed(message=str(error)).model_dump_json())
        except Exception:
            # The detail goes to the log only; see SESSION_FAILED.
            logger.exception("live session failed")
            # The socket may already be gone; the log above is the record.
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await socket.send_text(Failed(message=SESSION_FAILED).model_dump_json())
        finally:
            estimator.release_session()

    app.include_router(router)
    _mount_if_present(app, "/artifacts", settings.artifacts)
    _mount_if_present(app, "/datasets", settings.datasets)
    _mount_if_present(app, "/uploads", settings.uploads)
    _mount_web(app, settings.web)
    return app


def _mount_web(app: FastAPI, directory: Path) -> None:
    """Serve the built viewer, if it has been built.

    Mounted last so every route above wins. With it, running the whole thing is
    one command; without it the API still works and `pnpm dev` proxies here.
    """
    if not (directory / "index.html").is_file():
        logger.info("no viewer build at %s; run `pnpm build`, or use `pnpm dev`", directory)
        return
    # html=True serves index.html for `/`. It is not a fallback for unknown
    # paths, and the viewer needs none: its state lives in the query string.
    app.mount("/", StaticFiles(directory=directory, html=True), name="web")
    logger.info("serving the viewer from %s", directory)


async def _serve_live(socket: WebSocket, settings: Settings, estimator: live.Estimator) -> None:
    """One live session: read the opening message, then stream estimates."""
    opening = await socket.receive_text()
    try:
        request = json.loads(opening)
        start = (
            StartCamera.model_validate(request)
            if request.get("kind") == "camera"
            else StartVideo.model_validate(request)
        )
    except (ValidationError, ValueError) as error:
        message = f"could not read the request: {error}"
        await socket.send_text(Failed(message=message).model_dump_json())
        return

    session = await estimator.run(estimator.open_session)
    config = estimator.config

    if isinstance(start, StartVideo):
        await _serve_video(socket, settings, estimator, session, config, start)
    else:
        await _serve_camera(socket, estimator, session, config, start)


async def _serve_video(
    socket: WebSocket,
    settings: Settings,
    estimator: live.Estimator,
    session: EstimatorSession,
    config: BodiesConfig,
    start: StartVideo,
) -> None:
    try:
        path = resolve_playable(settings, start.path)
    except ValueError as error:
        await socket.send_text(Failed(message=str(error)).model_dump_json())
        return

    duration = await asyncio.to_thread(live.video_duration, path)
    await socket.send_text(
        live.ready(session, source=start.path, duration_seconds=duration).model_dump_json()
    )

    count = 0
    sent_bodies: set[int] = set()
    serialize_previous: float | None = None
    # Every frame the file offers, in order: a clip is not live and has nothing
    # to gain from dropping.
    async for timed in live.read_video(path, start.stride):
        frame, estimate = await live.estimate_one(
            estimator,
            session,
            config,
            timed,
            dropped=0,
            bodies=start.bodies,
            serialize_previous_seconds=serialize_previous,
        )
        if start.bodies:
            await _send_new_bodies(socket, estimator, session, estimate, sent_bodies)
        text, serialize_previous = live.encode_frame(frame)
        await socket.send_text(text)
        count += 1
    await socket.send_text(json.dumps({"kind": "done", "frames": count}))


async def _send_new_bodies(
    socket: WebSocket,
    estimator: live.Estimator,
    session: EstimatorSession,
    estimate: FrameEstimate,
    already: set[int],
) -> None:
    """Hand over the unposed body of anyone the viewer has not seen.

    Once per track. A body is a fifth of a megabyte and does not change: shape
    belongs to a person, and the model re-guessing it every frame is noise
    rather than news.
    """
    for person in estimate.people:
        if person.track_id in already:
            continue
        already.add(person.track_id)
        # Through the estimator, not `to_thread`: this reads the body model, and
        # Metal aborts when model work arrives on a second thread.
        payload = await estimator.run(
            partial(live.rest_body, session, person.track_id, person.shape)
        )
        await socket.send_bytes(payload)


async def _serve_camera(
    socket: WebSocket,
    estimator: live.Estimator,
    session: EstimatorSession,
    config: BodiesConfig,
    start: StartCamera,
) -> None:
    opened = live.ready(session, source="camera", duration_seconds=None)
    await socket.send_text(opened.model_dump_json())

    mailbox = live.LatestOnly()
    started = time.monotonic()
    sent_bodies: set[int] = set()

    async def receive() -> None:
        try:
            while True:
                message = await socket.receive()
                payload = message.get("bytes")
                if payload is None:
                    if message.get("type") == "websocket.disconnect":
                        break
                    continue
                frame_id: int | None = None
                began = time.perf_counter()
                try:
                    if start.frame_ids:
                        frame_id, payload = live.split_frame_id(payload)
                    image = live.decode_jpeg(payload)
                except ValueError:
                    # One frame the browser mangled is not a reason to end a
                    # session; the next one along is usually fine.
                    logger.warning("dropped a camera frame that would not decode")
                    continue
                mailbox.put(
                    live.Timed(
                        live.elapsed_since(started),
                        image,
                        frame_id=frame_id,
                        decode_seconds=time.perf_counter() - began,
                    )
                )
        finally:
            mailbox.close()

    reader = asyncio.create_task(receive())
    reported = 0
    serialize_previous: float | None = None
    try:
        async for timed in mailbox.stream():
            # What is told to the client is what is counted as reported. The
            # mailbox keeps dropping while this frame is in the model (the first
            # inference includes the model load), and reading the counter again
            # afterwards would mark those drops reported without ever sending
            # them (docs/notes/2026-09-15-adr0005-live-pilot-on-m4.md).
            dropped = mailbox.dropped - reported
            frame, estimate = await live.estimate_one(
                estimator,
                session,
                config,
                timed,
                dropped=dropped,
                bodies=start.bodies,
                serialize_previous_seconds=serialize_previous,
            )
            reported += dropped
            if start.bodies:
                await _send_new_bodies(socket, estimator, session, estimate, sent_bodies)
            text, serialize_previous = live.encode_frame(frame)
            await socket.send_text(text)
    finally:
        await _stop(reader)


def _built_topology(estimator: live.Estimator) -> tuple[wire.Topology, bytes]:
    """The body's topology, or a 503 that says how to install the model.

    A reader who has only run `uv sync` meets this route before anything else,
    and a bare 500 would not tell them the one thing they need: the extra.
    """
    try:
        return estimator.topology()
    except BodyEstimationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


async def _stop(task: asyncio.Task[None]) -> None:
    """Cancel a task and wait for it to actually be over.

    Cancelling and walking away leaves the task's exception unretrieved and its
    end unobserved, so a session could be reported closed while its reader was
    still running.
    """
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def build_settings(root: Path, config_path: Path) -> Settings:
    experiment = load_config(config_path)
    return Settings(
        root=root.resolve(), config_path=config_path.resolve(), bodies=experiment.bodies
    )
