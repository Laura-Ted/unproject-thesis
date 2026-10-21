"""Shared fixtures.

Videos are synthesised here rather than checked in, so the suite runs on a fresh
clone with no datasets present.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from fixture_types import MakeImageDir, MakePanningVideo, MakeVideo, WriteConfig

# MJPG in an AVI container is the writer available in every OpenCV build we
# target. It is lossy, but no test asserts on exact pixel values.
_FOURCC = "MJPG"


@pytest.fixture
def make_video(tmp_path: Path) -> MakeVideo:
    """Return a factory that writes a synthetic video and yields its path."""

    def _make(
        *,
        name: str = "clip.avi",
        frame_count: int = 10,
        fps: float = 10.0,
        width: int = 64,
        height: int = 48,
    ) -> Path:
        path = tmp_path / name
        fourcc = cv2.VideoWriter.fourcc(*_FOURCC)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        assert writer.isOpened(), "OpenCV could not open a MJPG writer"
        try:
            for index in range(frame_count):
                # Distinct per frame, so consecutive frames are tellable apart.
                frame = np.full((height, width, 3), index * 7 % 256, dtype=np.uint8)
                frame[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()
        return path

    return _make


@pytest.fixture
def make_panning_video(tmp_path: Path) -> MakePanningVideo:
    """Return a factory for a textured scene panning by a fixed step per frame.

    The flat fixture above is enough for stage [1], which only counts frames.
    Keyframe selection tracks corners, and a flat frame has none.
    """

    def _make(
        *,
        name: str = "pan.avi",
        frame_count: int = 12,
        step_px: int = 4,
        fps: float = 10.0,
        width: int = 320,
        height: int = 240,
    ) -> Path:
        rng = np.random.default_rng(seed=0)
        canvas = cv2.GaussianBlur(
            rng.integers(0, 256, (height, width + step_px * frame_count + 8), dtype=np.uint8),
            (3, 3),
            0,
        )
        path = tmp_path / name
        fourcc = cv2.VideoWriter.fourcc(*_FOURCC)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        assert writer.isOpened(), "OpenCV could not open a MJPG writer"
        try:
            for index in range(frame_count):
                start = index * step_px
                window = canvas[:, start : start + width]
                writer.write(cv2.cvtColor(window, cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()
        return path

    return _make


@pytest.fixture
def make_image_dir(tmp_path: Path) -> MakeImageDir:
    """Return a factory that writes a directory of frames and yields its path."""

    def _make(
        *,
        name: str = "frames",
        filenames: Sequence[str] | None = None,
        frame_count: int = 4,
        width: int = 64,
        height: int = 48,
    ) -> Path:
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        names = (
            list(filenames)
            if filenames is not None
            else [f"{index:05d}.png" for index in range(frame_count)]
        )
        for index, filename in enumerate(names):
            image = np.full((height, width, 3), index * 7 % 256, dtype=np.uint8)
            image[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)
            assert cv2.imwrite(str(directory / filename), image)
        return directory

    return _make


@pytest.fixture
def write_config(tmp_path: Path) -> WriteConfig:
    """Return a factory that writes an experiment config next to its data."""

    def _make(body: str, *, name: str = "experiment.toml") -> Path:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        return path

    return _make
