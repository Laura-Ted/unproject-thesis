"""Tests for the stage [3] driver.

The one thing asserted here that is not obvious: a pose artifact's location can
be worked out from a config without running anything. Stage [5] depends on that,
because it needs results from both backends and cannot load both their libraries
into one process.

In this directory because `run_pose` opens the COLMAP backend, and with it
pycolmap; see `openmp_guard.py`, the plugin that keeps this directory apart.
"""

from __future__ import annotations

from pathlib import Path

from fixture_types import MakePanningVideo, WriteConfig
from unproject.config import load_config
from unproject.pose.stage import resolve_artifact_dir, run_pose

_CONFIG = """
name = "fixture"
seed = 3

[ingest]
source_path = "pan.avi"

[preprocess]
min_disparity_px = 2.0
metric_long_edge = 160
"""


def test_the_artifact_location_can_be_worked_out_before_the_run(
    make_panning_video: MakePanningVideo, write_config: WriteConfig, tmp_path: Path
) -> None:
    make_panning_video(name="pan.avi", frame_count=14, step_px=6)
    config_path = write_config(_CONFIG)
    config = load_config(config_path)
    artifacts_root = tmp_path / "artifacts"

    predicted = resolve_artifact_dir(config, artifacts_root, config_path.parent)
    assert not predicted.exists()

    result = run_pose(config, config_dir=config_path.parent, artifacts_root=artifacts_root)
    # If these ever diverge, stage [5] silently reports that a run is missing.
    assert result.artifact_dir == predicted


def test_a_different_pose_config_resolves_somewhere_else(
    make_panning_video: MakePanningVideo, write_config: WriteConfig, tmp_path: Path
) -> None:
    make_panning_video(name="pan.avi", frame_count=14, step_px=6)
    baseline_path = write_config(_CONFIG, name="a.toml")
    changed_path = write_config(
        _CONFIG + '\n[pose.colmap]\nmatcher = "exhaustive"\n', name="b.toml"
    )
    root = tmp_path / "artifacts"

    assert resolve_artifact_dir(
        load_config(baseline_path), root, baseline_path.parent
    ) != resolve_artifact_dir(load_config(changed_path), root, changed_path.parent)


def test_an_upstream_change_moves_the_pose_artifact(
    make_panning_video: MakePanningVideo, write_config: WriteConfig, tmp_path: Path
) -> None:
    # The hash chains through preprocess, so different keyframes mean a
    # different pose run even with identical pose settings.
    make_panning_video(name="pan.avi", frame_count=14, step_px=6)
    baseline_path = write_config(_CONFIG, name="a.toml")
    changed_path = write_config(
        _CONFIG.replace("min_disparity_px = 2.0", "min_disparity_px = 9.0"), name="b.toml"
    )
    root = tmp_path / "artifacts"

    assert resolve_artifact_dir(
        load_config(baseline_path), root, baseline_path.parent
    ) != resolve_artifact_dir(load_config(changed_path), root, changed_path.parent)
