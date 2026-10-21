"""Tests for the sparse point cloud writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unproject.pose.pointcloud import PointCloudError, read_ply, write_ply


def _cloud(count: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=0)
    points = rng.uniform(-10.0, 10.0, size=(count, 3))
    colors = rng.integers(0, 256, size=(count, 3), dtype=np.uint8)
    return points, colors


def test_round_trips(tmp_path: Path) -> None:
    points, colors = _cloud(64)

    written = write_ply(tmp_path / "points.ply", points, colors)
    read_points, read_colors = read_ply(tmp_path / "points.ply")

    assert written == 64
    assert np.allclose(read_points, points, atol=1e-5)
    assert np.array_equal(read_colors, colors)


def test_positions_are_stored_as_float32(tmp_path: Path) -> None:
    points = np.array([[1.0 / 3.0, 0.0, 0.0]])

    write_ply(tmp_path / "points.ply", points, np.zeros((1, 3), dtype=np.uint8))
    read_points, _ = read_ply(tmp_path / "points.ply")

    # Float32 is what splat viewers read, so the loss is expected and bounded.
    assert read_points[0, 0] != points[0, 0]
    assert np.isclose(read_points[0, 0], points[0, 0], atol=1e-7)


def test_an_empty_cloud_is_writable(tmp_path: Path) -> None:
    written = write_ply(tmp_path / "points.ply", np.zeros((0, 3)), np.zeros((0, 3), np.uint8))

    assert written == 0
    assert read_ply(tmp_path / "points.ply")[0].shape == (0, 3)


def test_mismatched_colours_are_rejected(tmp_path: Path) -> None:
    points, colors = _cloud(5)

    with pytest.raises(PointCloudError, match="against"):
        write_ply(tmp_path / "points.ply", points, colors[:3])


def test_a_file_without_a_header_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.ply").write_bytes(b"not a ply at all")

    with pytest.raises(PointCloudError, match="no PLY header"):
        read_ply(tmp_path / "bad.ply")


def test_a_truncated_body_is_rejected(tmp_path: Path) -> None:
    points, colors = _cloud(10)
    path = tmp_path / "points.ply"
    write_ply(path, points, colors)
    path.write_bytes(path.read_bytes()[:-7])

    # A short read would otherwise become a silently smaller cloud.
    with pytest.raises(PointCloudError, match="holds"):
        read_ply(path)


def test_an_ascii_ply_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    (tmp_path / "ascii.ply").write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n"
    )

    with pytest.raises(PointCloudError, match="binary little-endian"):
        read_ply(tmp_path / "ascii.ply")
