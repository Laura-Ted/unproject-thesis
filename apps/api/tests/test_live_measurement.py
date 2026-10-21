"""Tests for what the live path reports about itself, and for the client that
measures it. ADR-0005 WP3.

No model: the estimator session is a stand-in, and the transport client is
driven against a small WebSocket server that speaks the same protocol, so the
socket is real and the inference is not.
"""

from __future__ import annotations

import asyncio
import json
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from unproject.benchmark.live import LiveReport, read_reports
from unproject.bodies.artifacts import BodyModelInfo
from unproject.bodies.multihmr import EstimatorSession, FrameEstimate
from unproject.config import BodiesConfig, ExperimentConfig
from unproject_api import live, measure
from unproject_api.app import Settings, create_app


class _FakeSession:
    def __init__(self) -> None:
        self.frames_seen = 0

    @property
    def body_model(self) -> BodyModelInfo:
        return BodyModelInfo(
            name="anny", version="0.6.0", bone_count=1, shape_count=11, bone_parents=[-1]
        )

    def frame(
        self, image: np.ndarray, config: BodiesConfig, *, width: int, height: int
    ) -> FrameEstimate:
        self.frames_seen += 1
        return FrameEstimate(intrinsics=(1000.0, 1000.0, width / 2, height / 2), people=[])


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> list[_FakeSession]:
    opened: list[_FakeSession] = []

    def _open(self: live.Estimator) -> EstimatorSession:
        session = _FakeSession()
        opened.append(session)
        return cast(EstimatorSession, session)

    monkeypatch.setattr(live.Estimator, "open_session", _open)
    return opened


def _settings(tmp_path: Path) -> Settings:
    config_path = tmp_path / "experiment.toml"
    config_path.write_text('name = "fixture"\n[ingest]\nsource_path = "clip.mp4"\n', "utf-8")
    return Settings(
        root=tmp_path,
        config_path=config_path,
        bodies=ExperimentConfig(name="fixture", ingest={"source_path": Path("clip.mp4")}).bodies,
    )


def _jpeg(value: int = 40) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((48, 64, 3), value, dtype=np.uint8))
    assert ok
    return bytes(encoded.tobytes())


def _video(directory: Path, *, frames: int = 6) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((48, 64, 3), index * 9 % 256, dtype=np.uint8))
    finally:
        writer.release()
    return path


class TestFrameIdsAndTimings:
    def test_a_numbered_camera_frame_is_answered_with_its_number_and_timings(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as socket,
        ):
            socket.send_text(json.dumps({"kind": "camera", "frame_ids": True}))
            ready = json.loads(socket.receive_text())
            socket.send_bytes(struct.pack("<I", 41) + _jpeg())
            first = json.loads(socket.receive_text())
            socket.send_bytes(struct.pack("<I", 42) + _jpeg(value=90))
            second = json.loads(socket.receive_text())

        assert ready["schema_version"] == 2
        # The server says where it is running, for the client's report.
        assert ready["environment"]["device"] in {"mps", "cpu", "unavailable"}
        assert first["frame_id"] == 41
        assert second["frame_id"] == 42
        timing = first["timing"]
        assert timing["decode_ms"] >= 0.0
        assert timing["queue_wait_ms"] >= 0.0
        assert timing["inference_ms"] >= 0.0
        # A message cannot carry its own encoding time; the second carries the first's.
        assert timing["serialize_previous_ms"] is None
        assert second["timing"]["serialize_previous_ms"] is not None

    def test_a_bare_jpeg_stays_a_valid_frame_when_ids_were_not_asked_for(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as socket,
        ):
            socket.send_text(json.dumps({"kind": "camera"}))
            socket.receive_text()
            socket.send_bytes(_jpeg())
            message = json.loads(socket.receive_text())

        assert message["frame_id"] is None
        assert message["timing"] is not None

    def test_a_numbered_frame_too_short_for_its_id_is_dropped_not_fatal(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as socket,
        ):
            socket.send_text(json.dumps({"kind": "camera", "frame_ids": True}))
            socket.receive_text()
            socket.send_bytes(b"\x01\x02")
            socket.send_bytes(struct.pack("<I", 7) + _jpeg())
            message = json.loads(socket.receive_text())

        assert message["frame_id"] == 7
        assert sessions[0].frames_seen == 1

    def test_a_file_session_numbers_its_frames_from_zero(
        self, sessions: list[_FakeSession], tmp_path: Path
    ) -> None:
        _video(tmp_path / "datasets", frames=4)
        with (
            TestClient(create_app(_settings(tmp_path))) as client,
            client.websocket_connect("/api/live") as socket,
        ):
            socket.send_text(
                json.dumps({"kind": "video", "path": "datasets/clip.avi", "stride": 2})
            )
            socket.receive_text()
            ids = []
            while (message := json.loads(socket.receive_text()))["kind"] != "done":
                ids.append(message["frame_id"])

        # Two of four frames at stride 2, numbered by what was sent.
        assert ids == [0, 1]


class TestKeepingAReport:
    def _draft(self) -> dict[str, object]:
        return {
            "kind": "browser",
            "started_at": "2026-09-15T10:00:00Z",
            "source": "datasets/pedestrians/clip.mp4",
            "source_sha256": None,
            "mode": "skeleton",
            "frame_width": 1280,
            "frame_height": 720,
            "environment": {"client_platform": "test", "user_agent": "vitest", "viewport": "1x1"},
            "config": {
                "send_fps": 2.0,
                "max_width": 1280,
                "jpeg_quality": 0.75,
                "bodies": False,
                "live_clock_lag_seconds": 0.4,
            },
            "samples": [
                {
                    "frame_id": 0,
                    "sent_at_ms": 0.0,
                    "sent_bytes": 100,
                    "received_at_ms": 400.0,
                    "presented_at_ms": 416.0,
                    "received_bytes": 50,
                    "people": 1,
                    "dropped_before": 0,
                },
                {"frame_id": 1, "sent_at_ms": 500.0, "sent_bytes": 100},
            ],
        }

    def test_the_server_computes_the_summary_and_keeps_the_file(self, tmp_path: Path) -> None:
        with TestClient(create_app(_settings(tmp_path))) as client:
            response = client.post("/api/live-metrics", json=self._draft())

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["frames_sent"] == 2
        assert body["frames_answered"] == 1
        reports, unreadable = read_reports(tmp_path / "artifacts" / "live")
        assert unreadable == []
        [report] = reports
        assert report.kind == "browser"
        assert report.summary.round_trip_ms.median == 400.0
        assert report.summary.capture_to_present_ms.median == 416.0
        assert report.summary.frames_unaccounted == 1
        assert (
            body["path"]
            == (tmp_path / "artifacts" / "live").relative_to(tmp_path).as_posix()
            + "/"
            + Path(body["path"]).name
        )

    def test_a_report_is_never_replaced(self, tmp_path: Path) -> None:
        with TestClient(create_app(_settings(tmp_path))) as client:
            assert client.post("/api/live-metrics", json=self._draft()).status_code == 200
            again = client.post("/api/live-metrics", json=self._draft())

        assert again.status_code == 409

    def test_a_draft_with_a_summary_is_refused(self, tmp_path: Path) -> None:
        """The summary is the server's to compute; a client cannot submit one."""
        draft = {**self._draft(), "summary": {}}
        with TestClient(create_app(_settings(tmp_path))) as client:
            assert client.post("/api/live-metrics", json=draft).status_code == 422

    def test_an_impossible_sample_is_refused(self, tmp_path: Path) -> None:
        draft = self._draft()
        cast(list[dict[str, object]], draft["samples"])[0]["received_at_ms"] = -1.0
        with TestClient(create_app(_settings(tmp_path))) as client:
            assert client.post("/api/live-metrics", json=draft).status_code == 422


class TestTheTransportClient:
    def test_frames_are_encoded_as_the_viewer_sends_them(self, tmp_path: Path) -> None:
        path = _video(tmp_path, frames=5)

        payloads, size, digest = measure.encode_frames(path, max_width=32, quality=0.75, limit=3)

        assert len(payloads) == 3
        assert size == (32, 24)
        assert all(payload[:2] == b"\xff\xd8" for payload in payloads)
        assert len(digest) == 64

    def test_a_session_pairs_answers_with_sends_and_counts_what_the_server_dropped(
        self, tmp_path: Path
    ) -> None:
        """Against a small server speaking the protocol: it answers every
        other frame and reports the ones it skipped, like the latest-only
        mailbox does."""
        from websockets.asyncio.server import ServerConnection, serve

        async def scenario() -> tuple[measure._Session, float]:
            async def handler(connection: ServerConnection) -> None:
                opening = json.loads(await connection.recv())
                assert opening["frame_ids"] is True
                await connection.send(
                    json.dumps(
                        {
                            "kind": "ready",
                            "schema_version": 2,
                            "convention": "camera_relative_opencv_metres",
                            "body_model": {
                                "name": "anny",
                                "version": "0.6.0",
                                "bone_count": 1,
                                "shape_count": 11,
                                "bone_parents": [-1],
                            },
                            "source": "camera",
                            "duration_seconds": None,
                            "environment": {
                                "machine": "Fake M0",
                                "device": "cpu",
                                "git_commit": "0" * 40,
                                "git_dirty": False,
                            },
                        }
                    )
                )
                skipped = 0
                async for raw in connection:
                    assert isinstance(raw, bytes)
                    (frame_id,) = struct.unpack("<I", raw[:4])
                    if frame_id % 2 == 1:
                        skipped += 1
                        continue
                    await connection.send(
                        json.dumps(
                            {
                                "kind": "frame",
                                "frame_id": frame_id,
                                "time_seconds": frame_id / 10,
                                "camera": {
                                    "width": 64,
                                    "height": 48,
                                    "fx": 100.0,
                                    "fy": 100.0,
                                    "cx": 32.0,
                                    "cy": 24.0,
                                    "source": "estimated",
                                },
                                "people": [],
                                "dropped": skipped,
                                "timing": {
                                    "decode_ms": 1.0,
                                    "queue_wait_ms": 2.0,
                                    "inference_ms": 30.0,
                                    "serialize_previous_ms": None,
                                },
                            }
                        )
                    )
                    skipped = 0

            async with serve(handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1] if server.sockets else 0
                return await measure.run_session(
                    f"ws://127.0.0.1:{port}/api/live",
                    [_jpeg(value) for value in range(6)],
                    fps=50.0,
                    bodies=False,
                    grace_seconds=2.0,
                )

        session, _ = asyncio.run(scenario())
        report = measure.build_report(
            session,
            started_at=datetime.now(UTC),
            source="clip.avi",
            source_sha256="a" * 64,
            frame_size=(64, 48),
            fps=50.0,
            bodies=False,
            max_width=1280,
            quality=0.75,
            limit=None,
        )

        assert report.summary.frames_sent == 6
        assert report.summary.frames_answered == 3
        # Frames 1 and 3 were reported dropped by the answers to 2 and 4; frame
        # 5 was skipped after the last answer, so nothing reported it and it is
        # unaccounted rather than dropped. The books still balance.
        assert report.summary.frames_dropped_by_server == 2
        assert report.summary.frames_unaccounted == 1
        assert report.summary.inference_ms.median == 30.0
        assert report.environment.server_machine == "Fake M0"
        assert report.summary.round_trip_ms.samples == 3
        assert all(
            sample.received_at_ms is None for sample in report.samples if sample.frame_id % 2
        )
        # What was written reads back as a valid report.
        assert LiveReport.model_validate_json(report.model_dump_json()) == report
