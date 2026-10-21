"""Tests for `VideoFileSource`."""

from __future__ import annotations

import hashlib
import logging
from itertools import islice
from pathlib import Path
from typing import Any

import cv2
import pytest

from fixture_types import MakeVideo
from unproject.ingest import Frame, FrameSource, VideoDecodeError, VideoFileSource


def test_satisfies_the_frame_source_protocol(make_video: MakeVideo) -> None:
    source = VideoFileSource(make_video())

    assert isinstance(source, FrameSource)


def test_metadata_matches_the_written_video(make_video: MakeVideo) -> None:
    path = make_video(frame_count=8, fps=10.0, width=64, height=48)

    metadata = VideoFileSource(path).metadata()

    assert (metadata.width, metadata.height) == (64, 48)
    assert metadata.fps == pytest.approx(10.0)
    assert metadata.reported_frame_count == 8
    assert metadata.duration_seconds == pytest.approx(0.8)
    assert metadata.codec == "MJPG"


def test_content_hash_matches_the_file(make_video: MakeVideo) -> None:
    path = make_video()

    metadata = VideoFileSource(path).metadata()

    assert metadata.content_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_uri_is_recorded_as_declared(make_video: MakeVideo) -> None:
    path = make_video()

    metadata = VideoFileSource(path, uri="datasets/fixture/clip.avi").metadata()

    # The declared path wins over the absolute one used to open the file.
    assert metadata.uri == "datasets/fixture/clip.avi"


def test_yields_every_frame_in_order(make_video: MakeVideo) -> None:
    source = VideoFileSource(make_video(frame_count=8, fps=10.0))

    frames = list(source.frames())

    assert len(frames) == 8
    assert [frame.source_index for frame in frames] == list(range(8))
    timestamps = [frame.timestamp_seconds for frame in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(0.0)
    # Eight frames at 10 fps: the last one sits at 0.7 s, not 0.8 s.
    assert timestamps[-1] == pytest.approx(0.7, abs=1e-6)
    assert timestamps[3] == pytest.approx(0.3, abs=1e-6)


def test_frames_carry_a_bgr_uint8_image(make_video: MakeVideo) -> None:
    source = VideoFileSource(make_video(width=64, height=48))

    frame: Frame = next(iter(source.frames()))

    assert frame.image.shape == (48, 64, 3)
    assert frame.image.dtype.name == "uint8"
    assert (frame.height, frame.width) == (48, 64)


def test_is_re_iterable(make_video: MakeVideo) -> None:
    source = VideoFileSource(make_video(frame_count=5))

    first = list(source.frames())
    second = list(source.frames())

    # No decoder is held between calls, so a second pass sees the same video.
    assert len(first) == len(second) == 5


def test_stopping_early_leaves_the_source_usable(make_video: MakeVideo) -> None:
    source = VideoFileSource(make_video(frame_count=100))

    partial = list(islice(source.frames(), 3))
    # Abandoning an iterator must not strand the capture.
    full = list(source.frames())

    assert [frame.source_index for frame in partial] == [0, 1, 2]
    assert len(full) == 100


def test_missing_file_is_rejected_at_construction(tmp_path: Path) -> None:
    with pytest.raises(VideoDecodeError, match="no such video file"):
        VideoFileSource(tmp_path / "absent.avi")


def test_undecodable_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not-a-video.avi"
    path.write_bytes(b"this is not a container")

    with pytest.raises(VideoDecodeError):
        VideoFileSource(path).metadata()


# Bound before the monkeypatch below replaces the name on cv2.
_REAL_VIDEO_CAPTURE = cv2.VideoCapture


class _FlatTimestampCapture:
    """A capture reporting 0 ms for every frame, as some containers do."""

    def __init__(self, path: str) -> None:
        self._inner = _REAL_VIDEO_CAPTURE(path)

    def isOpened(self) -> bool:  # noqa: N802 - matches the OpenCV API
        return bool(self._inner.isOpened())

    def read(self) -> tuple[bool, Any]:
        return self._inner.read()

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_MSEC:
            return 0.0
        return float(self._inner.get(prop))

    def release(self) -> None:
        self._inner.release()


def test_falls_back_to_nominal_fps_when_timestamps_are_flat(
    make_video: MakeVideo,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = VideoFileSource(make_video(frame_count=5, fps=10.0))
    monkeypatch.setattr(cv2, "VideoCapture", _FlatTimestampCapture)

    with caplog.at_level(logging.WARNING):
        timestamps = [frame.timestamp_seconds for frame in source.frames()]

    assert timestamps == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    # Without the warning, a variable-rate capture would look constant-rate.
    assert "unusable frame timestamps" in caplog.text
