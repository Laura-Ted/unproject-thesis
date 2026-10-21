"""Tests for the parts of the API that do not need a model.

Inference is not exercised here: its correctness is a question for the
evaluation. What is tested is what decides which file gets opened and which
frames get thrown away.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from unproject.bodies.artifacts import ARTIFACT_FILENAME, BODIES_SCHEMA_VERSION
from unproject.config import BodiesConfig
from unproject_api.app import (
    Settings,
    collect_runs,
    create_app,
    resolve_playable,
    resolve_run,
)
from unproject_api.live import LatestOnly, Timed


def _artifact_json() -> str:
    """The smallest bodies artifact that validates."""
    return json.dumps(
        {
            "schema_version": BODIES_SCHEMA_VERSION,
            "convention": "camera_relative_opencv_metres",
            "ingest_hash": "0" * 12,
            "body_model": {
                "name": "anny",
                "version": "0.6.0",
                "bone_count": 1,
                "shape_count": 11,
                "bone_parents": [-1],
            },
            "estimator": {
                "name": "multihmr2",
                "checkpoint_sha256": "f" * 64,
                "backbone_repo": "facebookresearch/dinov3",
                "backbone_ref": "0" * 40,
                "config_hash": "deadbeef",
            },
            "parameters": {"path": "bodies.bin", "person_count": 0, "arrays": []},
            "frames": [
                {
                    "frame_index": 0,
                    "source_index": 0,
                    "timestamp_seconds": 0.0,
                    "camera": {
                        "width": 1280,
                        "height": 720,
                        "fx": 1000.0,
                        "fy": 1000.0,
                        "cx": 640.0,
                        "cy": 360.0,
                        "source": "estimated",
                    },
                    "people": [],
                }
            ],
            "summary": {
                "frames_total": 1,
                "frames_with_people": 0,
                "people_total": 0,
                "max_people_in_frame": 0,
                "confidence_threshold": 0.4,
                "track_count": 0,
            },
        }
    )


def _settings(root: Path) -> Settings:
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    config = root / "experiments" / "x.toml"
    config.touch()
    return Settings(root=root, config_path=config, bodies=BodiesConfig())


def _video(settings: Settings, relative: str) -> Path:
    path = settings.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a video, but it is a file with the right suffix")
    return path


class TestResolvePlayable:
    def test_a_video_under_datasets_resolves(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        expected = _video(settings, "datasets/crowd/clip.mp4")
        assert resolve_playable(settings, "datasets/crowd/clip.mp4") == expected

    def test_an_upload_resolves(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        expected = _video(settings, "uploads/abc.mp4")
        assert resolve_playable(settings, "uploads/abc.mp4") == expected

    def test_a_leading_slash_is_tolerated(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _video(settings, "datasets/crowd/clip.mp4")
        assert resolve_playable(settings, "/datasets/crowd/clip.mp4").name == "clip.mp4"

    def test_climbing_out_of_the_served_roots_is_refused(self, tmp_path: Path) -> None:
        """The path comes from a client, so it is not to be trusted."""
        settings = _settings(tmp_path)
        outside = tmp_path / "secret.mp4"
        outside.write_bytes(b"x")
        with pytest.raises(ValueError, match="not a file this server offers"):
            resolve_playable(settings, "datasets/../secret.mp4")

    def test_an_unserved_directory_is_refused(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _video(settings, "artifacts/crowd/clip.mp4")
        with pytest.raises(ValueError, match="not under a served directory"):
            resolve_playable(settings, "artifacts/crowd/clip.mp4")

    def test_a_directory_on_its_own_is_refused(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        with pytest.raises(ValueError, match="not under a served directory"):
            resolve_playable(settings, "datasets")

    def test_something_that_is_not_a_video_is_refused(self, tmp_path: Path) -> None:
        # The decoder would fail anyway; refusing by suffix keeps it from being
        # handed arbitrary files to open.
        settings = _settings(tmp_path)
        _video(settings, "uploads/notes.txt")
        with pytest.raises(ValueError, match="not a video"):
            resolve_playable(settings, "uploads/notes.txt")

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        with pytest.raises(ValueError, match="not a file this server offers"):
            resolve_playable(settings, "datasets/nothing/here.mp4")


class TestCollectRuns:
    def test_an_empty_tree_lists_nothing(self, tmp_path: Path) -> None:
        assert collect_runs(_settings(tmp_path)) == []

    def test_an_unreadable_run_is_skipped_rather_than_fatal(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        run = settings.artifacts / "crowd" / "bodies" / "deadbeef"
        run.mkdir(parents=True)
        (run / "bodies.json").write_text("{ not json", encoding="utf-8")
        assert collect_runs(settings) == []


class TestLatestOnly:
    def test_it_hands_over_what_was_put_in(self) -> None:
        async def run() -> None:
            mailbox = LatestOnly()
            mailbox.put(Timed(1.0, None))  # type: ignore[arg-type]
            got = await mailbox.get()
            assert got is not None
            assert got.time_seconds == 1.0

        asyncio.run(run())

    def test_a_newer_frame_replaces_an_unread_one(self) -> None:
        """This is the backpressure contract: live means newest, not next."""

        async def run() -> None:
            mailbox = LatestOnly()
            for at in (1.0, 2.0, 3.0):
                mailbox.put(Timed(at, None))  # type: ignore[arg-type]
            got = await mailbox.get()
            assert got is not None
            assert got.time_seconds == 3.0
            assert mailbox.dropped == 2

        asyncio.run(run())

    def test_nothing_is_counted_as_dropped_when_it_keeps_up(self) -> None:
        async def run() -> None:
            mailbox = LatestOnly()
            for at in (1.0, 2.0):
                mailbox.put(Timed(at, None))  # type: ignore[arg-type]
                await mailbox.get()
            assert mailbox.dropped == 0

        asyncio.run(run())

    def test_closing_ends_the_stream(self) -> None:
        async def run() -> None:
            mailbox = LatestOnly()
            mailbox.close()
            assert await mailbox.get() is None

        asyncio.run(run())

    def test_a_frame_put_before_closing_is_still_delivered(self) -> None:
        # Otherwise the last frame of a clip would be lost on shutdown.
        async def run() -> None:
            mailbox = LatestOnly()
            mailbox.put(Timed(9.0, None))  # type: ignore[arg-type]
            mailbox.close()
            got = await mailbox.get()
            assert got is not None
            assert got.time_seconds == 9.0
            assert await mailbox.get() is None

        asyncio.run(run())

    def test_a_waiting_reader_wakes_when_a_frame_arrives(self) -> None:
        async def run() -> None:
            mailbox = LatestOnly()

            async def later() -> None:
                await asyncio.sleep(0.01)
                mailbox.put(Timed(4.0, None))  # type: ignore[arg-type]

            task = asyncio.create_task(later())
            got = await asyncio.wait_for(mailbox.get(), timeout=1.0)
            await task
            assert got is not None
            assert got.time_seconds == 4.0

        asyncio.run(run())


def test_the_api_loads_neither_torch_nor_pycolmap_on_import() -> None:
    """The server reaches torch, so it must never reach pycolmap.

    Each bundles its own libomp and the second to initialise aborts the process
    (see `packages/pipeline/tests/openmp_guard.py`). Torch arrives only when
    inference does; pycolmap must never arrive at all, which is why nothing here
    imports the pose stage.
    """
    probe = (
        "import sys; import unproject_api.app; "
        "print('pycolmap' in sys.modules, 'torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


class TestRoutes:
    """Behaviour through the real app, run in-process."""

    def _client(self, tmp_path: Path) -> TestClient:
        return TestClient(create_app(_settings(tmp_path)))

    def test_an_empty_tree_serves_an_empty_list(self, tmp_path: Path) -> None:
        # Serving finished runs must not depend on being able to run the model,
        # and an install with nothing computed yet is not an error.
        response = self._client(tmp_path).get("/api/runs")
        assert response.status_code == 200
        assert response.json() == []

    def test_a_run_on_disk_is_listed(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        run = settings.artifacts / "crowd" / "bodies" / "abc123"
        run.mkdir(parents=True)
        (run / "bodies.json").write_text(_artifact_json(), encoding="utf-8")

        listed = TestClient(create_app(settings)).get("/api/runs").json()
        assert [entry["hash"] for entry in listed] == ["abc123"]
        assert listed[0]["frames"] == 1
        # No ingest artifact beside it, so there is no footage to play.
        assert listed[0]["video"] is None

    def test_a_file_that_is_not_a_video_is_refused(self, tmp_path: Path) -> None:
        response = self._client(tmp_path).post(
            "/api/uploads", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 415

    def test_an_uploaded_video_comes_back_addressable(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        client = TestClient(create_app(settings))
        response = client.post(
            "/api/uploads", files={"file": ("clip.mp4", b"pretend this is h264", "video/mp4")}
        )
        assert response.status_code == 200

        body = response.json()
        assert body["name"] == "clip.mp4"
        # The path it hands back must be one the live route will accept.
        assert resolve_playable(settings, body["path"]).read_bytes() == b"pretend this is h264"


class TestDeletingARun:
    """The one destructive thing the server does, so the checks are narrow."""

    def _run(self, tmp_path: Path) -> Path:
        run = tmp_path / "artifacts" / "crowd" / "bodies" / "abc"
        run.mkdir(parents=True)
        (run / ARTIFACT_FILENAME).write_text("{}", encoding="utf-8")
        (run / "bodies.bin").write_bytes(b"\x00" * 8)
        return run

    def test_it_removes_the_run_and_everything_in_it(self, tmp_path: Path) -> None:
        run = self._run(tmp_path)

        with TestClient(create_app(_settings(tmp_path))) as client:
            response = client.delete("/api/runs/crowd/bodies/abc")

        assert response.status_code == 200
        assert response.json()["path"] == "crowd/bodies/abc"
        assert not run.exists()

    def test_climbing_out_of_the_artifact_tree_is_refused(self, tmp_path: Path) -> None:
        secret = tmp_path / "datasets" / "clip.mp4"
        secret.parent.mkdir(parents=True, exist_ok=True)
        secret.write_bytes(b"footage")

        with TestClient(create_app(_settings(tmp_path))) as client:
            response = client.delete("/api/runs/../datasets")

        # The router normalises the path before the handler sees it, so this is
        # a 404 rather than a 400. Either way nothing was removed, and the unit
        # test below covers the traversal the router does not swallow.
        assert response.status_code >= 400
        assert secret.exists()

    @pytest.mark.parametrize(
        "escape",
        ["../datasets", "crowd/../../datasets", "/etc", "crowd/bodies/../../../uploads"],
    )
    def test_resolve_run_refuses_anything_outside_the_tree(
        self, escape: str, tmp_path: Path
    ) -> None:
        """Tested directly: a raw path never reaches the handler through a router."""
        (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError):
            resolve_run(_settings(tmp_path), escape)

    def test_a_directory_without_an_artifact_is_refused(self, tmp_path: Path) -> None:
        """Only a finished run can be removed, never any directory in the tree."""
        plain = tmp_path / "artifacts" / "crowd" / "ingest" / "abc"
        plain.mkdir(parents=True)
        (plain / "frames").mkdir()

        with TestClient(create_app(_settings(tmp_path))) as client:
            response = client.delete("/api/runs/crowd/ingest/abc")

        assert response.status_code == 400
        assert plain.exists()

    def test_the_artifact_root_itself_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "artifacts"
        root.mkdir(parents=True, exist_ok=True)

        with TestClient(create_app(_settings(tmp_path))) as client:
            response = client.delete("/api/runs/")

        assert response.status_code in {400, 404}
        assert root.exists()

    def test_a_run_that_is_not_there_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

        with TestClient(create_app(_settings(tmp_path))) as client:
            response = client.delete("/api/runs/crowd/bodies/nope")

        assert response.status_code == 400

    def test_it_disappears_from_the_list(self, tmp_path: Path) -> None:
        self._run(tmp_path)

        with TestClient(create_app(_settings(tmp_path))) as client:
            client.delete("/api/runs/crowd/bodies/abc")
            assert client.get("/api/runs").json() == []
