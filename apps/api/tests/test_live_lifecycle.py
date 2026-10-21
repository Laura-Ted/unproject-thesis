"""Tests for what a live session owns and what it gives back.

None of this loads the model: the estimator's session is replaced by one that
returns a fixed frame, which is enough, because what is under test is who owns
a thread, a task and a decoder, and when they are released.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from unproject.bodies.artifacts import BodyModelInfo
from unproject.bodies.multihmr import EstimatorSession, FrameEstimate
from unproject.config import BodiesConfig, ExperimentConfig
from unproject_api import live
from unproject_api.app import SESSION_FAILED, SESSIONS_BUSY, Settings, create_app


class _FakeSession:
    """An estimator session that answers without a model behind it."""

    def __init__(self) -> None:
        self.frames_seen = 0
        self.threads: set[int] = set()

    @property
    def body_model(self) -> BodyModelInfo:
        return BodyModelInfo(
            name="anny", version="0.6.0", bone_count=1, shape_count=11, bone_parents=[-1]
        )

    def frame(
        self, image: np.ndarray, config: BodiesConfig, *, width: int, height: int
    ) -> FrameEstimate:
        self.frames_seen += 1
        # Which thread the model work ran on is the whole point of the executor.
        self.threads.add(threading.get_ident())
        return FrameEstimate(intrinsics=(1000.0, 1000.0, width / 2, height / 2), people=[])


class _BlockingFirstFrame(_FakeSession):
    """A session whose first frame waits to be released, like a model loading."""

    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self._release = release

    def frame(
        self, image: np.ndarray, config: BodiesConfig, *, width: int, height: int
    ) -> FrameEstimate:
        if self.frames_seen == 0:
            self._release.wait(timeout=5.0)
        return super().frame(image, config, width=width, height=height)


def _settings(tmp_path: Path) -> Settings:
    config_path = tmp_path / "experiment.toml"
    config_path.write_text('name = "fixture"\n[ingest]\nsource_path = "clip.mp4"\n', "utf-8")
    return Settings(
        root=tmp_path,
        config_path=config_path,
        bodies=ExperimentConfig(name="fixture", ingest={"source_path": Path("clip.mp4")}).bodies,
    )


def _video(directory: Path, *, frames: int = 12, name: str = "clip.avi") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((48, 64, 3), index * 9 % 256, dtype=np.uint8))
    finally:
        writer.release()
    return path


def _jpeg(value: int = 40) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((48, 64, 3), value, dtype=np.uint8))
    assert ok
    return bytes(encoded.tobytes())


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> list[_FakeSession]:
    """Replace the model with a stand-in, and keep every session handed out."""
    opened: list[_FakeSession] = []

    def _open(self: live.Estimator) -> EstimatorSession:
        session = _FakeSession()
        opened.append(session)
        return cast(EstimatorSession, session)

    monkeypatch.setattr(live.Estimator, "open_session", _open)
    return opened


def _threads_named(prefix: str) -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name.startswith(prefix)]


class TestVideoReader:
    def test_a_whole_clip_arrives_in_order(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        """The lossless half of the contract, before breaking it on purpose."""
        _video(tmp_path / "datasets", frames=8)

        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as socket,
        ):
            socket.send_text(
                json.dumps({"kind": "video", "path": "datasets/clip.avi", "stride": 1})
            )
            assert json.loads(socket.receive_text())["kind"] == "ready"
            times = []
            while (message := json.loads(socket.receive_text()))["kind"] != "done":
                times.append(message["time_seconds"])

        assert message["frames"] == 8
        assert times == sorted(times)

    def test_leaving_early_does_not_strand_the_decode_thread(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        """The producer used to park forever on a queue nobody was draining."""
        _video(tmp_path / "datasets", frames=400)
        before = len(_threads_named("asyncio_"))

        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as socket,
        ):
            socket.send_text(json.dumps({"kind": "video", "path": "datasets/clip.avi"}))
            assert json.loads(socket.receive_text())["kind"] == "ready"
            # Take one frame of four hundred, then walk away mid-clip with the
            # read-ahead buffer full.
            socket.receive_text()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(_threads_named("asyncio_")) > before:
            time.sleep(0.05)
        assert len(_threads_named("asyncio_")) <= before

    def test_a_decode_that_dies_partway_fails_the_session(
        self,
        sessions: list[_FakeSession],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """It used to end as a short clip: the exception died with the task."""
        _video(tmp_path / "datasets", frames=8)

        def _dies_after_two(path: Path, stride: int) -> Iterator[live.Timed]:
            for index in range(2):
                yield live.Timed(index * 0.1, np.zeros((48, 64, 3), dtype=np.uint8))
            raise OSError("the decoder gave up")

        monkeypatch.setattr(live, "video_frames", _dies_after_two)

        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as socket,
        ):
            socket.send_text(
                json.dumps({"kind": "video", "path": "datasets/clip.avi", "stride": 1})
            )
            assert json.loads(socket.receive_text())["kind"] == "ready"
            messages = []
            while (message := json.loads(socket.receive_text()))["kind"] == "frame":
                messages.append(message)

        assert len(messages) == 2
        assert message["kind"] == "error"
        # The client is told the session stopped, not what the disk said.
        assert message["message"] == SESSION_FAILED
        assert "decoder gave up" not in json.dumps(message)


class TestCamera:
    def test_a_frame_that_will_not_decode_does_not_end_the_session(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as socket,
        ):
            socket.send_text(json.dumps({"kind": "camera"}))
            assert json.loads(socket.receive_text())["kind"] == "ready"

            socket.send_bytes(b"this is not a jpeg")
            socket.send_bytes(_jpeg())
            message = json.loads(socket.receive_text())

        # The good frame after the bad one still came back.
        assert message["kind"] == "frame"
        assert sessions[0].frames_seen == 1

    def test_frames_dropped_while_one_is_in_the_model_are_reported_on_the_next(
        self, sessions: list[_FakeSession], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first inference holds the model thread for as long as the model
        takes to load, and the camera keeps sending. Every frame the mailbox
        threw away meanwhile is owed to the client on the next answer; the
        transport pilot of 2026-09-15 had nine that were never accounted for."""
        release = threading.Event()
        puts: list[int] = []
        original_put = live.LatestOnly.put

        def counted_put(self: live.LatestOnly, item: live.Timed) -> None:
            original_put(self, item)
            puts.append(item.frame_id if item.frame_id is not None else -1)

        monkeypatch.setattr(live.LatestOnly, "put", counted_put)

        def _open(self: live.Estimator) -> EstimatorSession:
            session = _BlockingFirstFrame(release)
            sessions.append(session)
            return cast(EstimatorSession, session)

        monkeypatch.setattr(live.Estimator, "open_session", _open)

        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as socket,
        ):
            socket.send_text(json.dumps({"kind": "camera", "frame_ids": True}))
            assert json.loads(socket.receive_text())["kind"] == "ready"

            for frame_id in range(4):
                socket.send_bytes(frame_id.to_bytes(4, "little") + _jpeg(frame_id * 30))
            # Frames 1-3 have to be in the mailbox before frame 0 is let out of
            # the model, which is the situation under test.
            deadline = time.monotonic() + 5.0
            while len(puts) < 4 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert puts == [0, 1, 2, 3]
            release.set()

            first = json.loads(socket.receive_text())
            second = json.loads(socket.receive_text())

        assert (first["frame_id"], first["dropped"]) == (0, 0)
        # Frames 1 and 2 were replaced in the slot while 0 was in the model;
        # 3 is answered and carries them.
        assert (second["frame_id"], second["dropped"]) == (3, 2)
        assert sessions[-1].frames_seen == 2

    def test_disconnecting_mid_session_leaves_the_next_one_healthy(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        app = create_app(_settings(tmp_path))

        with TestClient(app) as client:
            with client.websocket_connect("/api/live") as socket:
                socket.send_text(json.dumps({"kind": "camera"}))
                assert json.loads(socket.receive_text())["kind"] == "ready"
                socket.send_bytes(_jpeg())
                socket.receive_text()

            with client.websocket_connect("/api/live") as socket:
                socket.send_text(json.dumps({"kind": "camera"}))
                assert json.loads(socket.receive_text())["kind"] == "ready"
                socket.send_bytes(_jpeg(value=90))
                assert json.loads(socket.receive_text())["kind"] == "frame"

        # A session apiece, and the second was not handed the first one's tracker.
        assert len(sessions) == 2
        assert sessions[0] is not sessions[1]


class TestTheModelThread:
    def test_every_frame_of_every_session_runs_on_the_one_thread(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        """Metal keeps a command buffer per thread and aborts if work strays."""
        with TestClient(create_app(_settings(tmp_path))) as client:
            for value in (20, 60):
                with client.websocket_connect("/api/live") as socket:
                    socket.send_text(json.dumps({"kind": "camera"}))
                    socket.receive_text()
                    socket.send_bytes(_jpeg(value))
                    socket.receive_text()

        used = {thread for session in sessions for thread in session.threads}
        assert len(used) == 1

    def test_a_third_session_is_turned_away_rather_than_queued(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as first,
        ):
            first.send_text(json.dumps({"kind": "camera"}))
            first.receive_text()
            with client.websocket_connect("/api/live") as second:
                second.send_text(json.dumps({"kind": "camera"}))
                second.receive_text()
                with client.websocket_connect("/api/live") as third:
                    refusal = json.loads(third.receive_text())

        assert refusal["kind"] == "error"
        assert refusal["message"] == SESSIONS_BUSY

    def test_a_slot_comes_back_when_a_session_ends(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        """Otherwise the server refuses everything after two tabs have been opened."""
        with TestClient(create_app(_settings(tmp_path))) as client:
            for _ in range(4):
                with client.websocket_connect("/api/live") as socket:
                    socket.send_text(json.dumps({"kind": "camera"}))
                    assert json.loads(socket.receive_text())["kind"] == "ready"


class TestShutdown:
    def test_leaving_the_app_stops_the_model_worker(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        app = create_app(_settings(tmp_path))

        with TestClient(app) as client, client.websocket_connect("/api/live") as socket:
            socket.send_text(json.dumps({"kind": "camera"}))
            socket.receive_text()
            socket.send_bytes(_jpeg())
            socket.receive_text()
            assert _threads_named("unproject-model")

        # The lifespan has exited by here.
        assert not _threads_named("unproject-model")


class TestStopping:
    async def _run(self) -> str:
        """A task that would outlive its owner if nobody waited for it."""
        from unproject_api.app import _stop

        started = asyncio.Event()
        ended = ""

        async def forever() -> None:
            nonlocal ended
            try:
                started.set()
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                ended = "cancelled"
                raise

        task = asyncio.create_task(forever())
        await started.wait()
        await _stop(task)
        return ended

    def test_stopping_a_task_waits_for_it_to_be_over(self) -> None:
        assert asyncio.run(self._run()) == "cancelled"


class TestWithoutTheModelExtra:
    """A reader who has only run `uv sync` meets this before anything else.

    Found by starting a clean checkout: the topology route answered a bare 500,
    and the socket answered "the live session stopped; see the server log" -
    which WP2 introduced by replacing the exception message with a stable one.
    Neither told them the one thing they needed to know.
    """

    @pytest.fixture
    def no_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unproject.bodies.multihmr import INSTALL_HINT, BodyEstimationError

        def _missing(*args: object, **kwargs: object) -> object:
            raise BodyEstimationError(INSTALL_HINT)

        monkeypatch.setattr(live.Estimator, "topology", _missing)
        monkeypatch.setattr(live.Estimator, "open_session", _missing)

    def test_the_topology_route_says_which_extra_to_install(
        self, no_extra: None, tmp_path: Path
    ) -> None:
        with TestClient(create_app(_settings(tmp_path))) as client:
            response = client.get("/api/body/topology")

        assert response.status_code == 503
        assert "uv sync --extra multihmr" in response.json()["detail"]

    def test_the_socket_says_it_too(self, no_extra: None, tmp_path: Path) -> None:
        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as socket,
        ):
            socket.send_text(json.dumps({"kind": "camera"}))
            message = json.loads(socket.receive_text())

        assert message["kind"] == "error"
        assert "uv sync --extra multihmr" in message["message"]

    def test_finished_runs_are_still_served(self, no_extra: None, tmp_path: Path) -> None:
        """Viewing what is already on disk must not need the model at all."""
        with TestClient(create_app(_settings(tmp_path))) as client:
            assert client.get("/api/runs").status_code == 200
