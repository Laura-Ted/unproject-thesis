"""Tests for the preprocess stage: chaining, resizing, artifact contract, reuse."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from fixture_types import MakePanningVideo
from unproject.config import ExperimentConfig, IngestConfig, PreprocessConfig
from unproject.preprocess.artifacts import (
    ARTIFACT_FILENAME,
    PREPROCESS_SCHEMA_VERSION,
    PreprocessArtifact,
)
from unproject.preprocess.stage import PreprocessError, preprocess_hash, run_preprocess
from unproject.run.manifest import MANIFEST_FILENAME, RunManifest


def _config(
    video: Path,
    *,
    frame_stride: int = 1,
    **preprocess: object,
) -> ExperimentConfig:
    defaults: dict[str, object] = {
        "min_disparity_px": 0.5,
        "blur_percentile": 0.0,
        "first_frame_window": 1,
        "held_out_every": None,
    }
    return ExperimentConfig(
        name="fixture",
        seed=7,
        ingest=IngestConfig(source_path=Path(video.name), frame_stride=frame_stride),
        preprocess=PreprocessConfig.model_validate(defaults | preprocess),
    )


def test_runs_stage_one_then_selects(make_panning_video: MakePanningVideo, tmp_path: Path) -> None:
    video = make_panning_video(frame_count=8)

    result = run_preprocess(
        _config(video), config_dir=video.parent, artifacts_root=tmp_path / "artifacts"
    )

    assert result.reused is False
    assert result.artifact.selection.keyframes_kept >= 1
    assert result.artifact.selection.frames_considered == 8
    # Both stages ran from the one call.
    assert (tmp_path / "artifacts" / "fixture" / "ingest").is_dir()
    assert (tmp_path / "artifacts" / "fixture" / "preprocess").is_dir()


def test_keyframes_are_written_and_readable(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    video = make_panning_video(frame_count=8, width=64, height=48)

    result = run_preprocess(
        _config(video), config_dir=video.parent, artifacts_root=tmp_path / "artifacts"
    )

    written = list((result.artifact_dir / "keyframes").glob("*.png"))
    assert len(written) == result.artifact.selection.keyframes_kept
    image = cv2.imread(str(result.artifact_dir / result.artifact.keyframes[0].path))
    assert image is not None
    assert image.shape == (48, 64, 3)


def test_keyframes_trace_back_to_the_source(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    video = make_panning_video(frame_count=8)

    result = run_preprocess(
        _config(video, frame_stride=2),
        config_dir=video.parent,
        artifacts_root=tmp_path / "artifacts",
    )

    indices = [record.source_index for record in result.artifact.keyframes]
    # Stage [1] kept every other frame, so only even indices are available.
    assert all(index % 2 == 0 for index in indices)
    assert indices == sorted(indices)


def test_resize_caps_the_long_edge_and_records_the_factor(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    video = make_panning_video(frame_count=4, width=320, height=240)

    result = run_preprocess(
        _config(video, max_long_edge=160),
        config_dir=video.parent,
        artifacts_root=tmp_path / "artifacts",
    )

    resize = result.artifact.resize
    assert (resize.width, resize.height) == (160, 120)
    # Intrinsics scale with this factor, so it has to reach the artifact.
    assert resize.scale == pytest.approx(0.5)


def test_resize_never_upscales(make_panning_video: MakePanningVideo, tmp_path: Path) -> None:
    video = make_panning_video(frame_count=4, width=64, height=48)

    result = run_preprocess(
        _config(video, max_long_edge=4000),
        config_dir=video.parent,
        artifacts_root=tmp_path / "artifacts",
    )

    assert result.artifact.resize.scale == 1.0
    assert (result.artifact.resize.width, result.artifact.resize.height) == (64, 48)


def test_artifact_round_trips_through_disk(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    video = make_panning_video(frame_count=6)

    result = run_preprocess(
        _config(video), config_dir=video.parent, artifacts_root=tmp_path / "artifacts"
    )

    raw = (result.artifact_dir / ARTIFACT_FILENAME).read_text("utf-8")
    assert PreprocessArtifact.model_validate_json(raw) == result.artifact
    assert json.loads(raw)["schema_version"] == PREPROCESS_SCHEMA_VERSION


def test_manifest_records_the_run(make_panning_video: MakePanningVideo, tmp_path: Path) -> None:
    video = make_panning_video(frame_count=6)

    result = run_preprocess(
        _config(video), config_dir=video.parent, artifacts_root=tmp_path / "artifacts"
    )

    manifest = RunManifest.model_validate_json(
        (result.artifact_dir / MANIFEST_FILENAME).read_text("utf-8")
    )
    assert manifest.stage == "preprocess"
    assert manifest.status == "ok"
    assert manifest.seed == 7


def test_second_run_reuses_the_artifact(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    video = make_panning_video(frame_count=6)
    artifacts = tmp_path / "artifacts"

    first = run_preprocess(_config(video), config_dir=video.parent, artifacts_root=artifacts)
    second = run_preprocess(_config(video), config_dir=video.parent, artifacts_root=artifacts)

    assert second.reused is True
    assert second.artifact == first.artifact


def test_changing_the_ingest_config_invalidates_the_keyframes(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    video = make_panning_video(frame_count=8)
    artifacts = tmp_path / "artifacts"

    every_frame = run_preprocess(_config(video), config_dir=video.parent, artifacts_root=artifacts)
    every_other = run_preprocess(
        _config(video, frame_stride=2), config_dir=video.parent, artifacts_root=artifacts
    )

    # Same stage [2] config, different frames going in, so no reuse.
    assert every_other.reused is False
    assert every_other.artifact_dir != every_frame.artifact_dir


def test_the_hash_depends_on_both_inputs() -> None:
    config = PreprocessConfig()

    assert preprocess_hash("aaa", config) != preprocess_hash("bbb", config)
    assert preprocess_hash("aaa", config) == preprocess_hash("aaa", config)
    assert preprocess_hash("aaa", config) != preprocess_hash(
        "aaa", PreprocessConfig(min_disparity_px=99.0)
    )


def test_an_impossible_threshold_is_an_error(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    video = make_panning_video(frame_count=6)

    # Only the anchor clears the threshold, and one view cannot be reconstructed.
    with pytest.raises(PreprocessError, match="cannot be reconstructed"):
        run_preprocess(
            _config(video, min_disparity_px=100000.0),
            config_dir=video.parent,
            artifacts_root=tmp_path / "artifacts",
        )


def test_a_failed_run_still_leaves_a_manifest(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    video = make_panning_video(frame_count=6)
    artifacts = tmp_path / "artifacts"

    with pytest.raises(PreprocessError):
        run_preprocess(
            _config(video, min_disparity_px=100000.0),
            config_dir=video.parent,
            artifacts_root=artifacts,
        )

    manifests = list(artifacts.rglob(f"preprocess/*/{MANIFEST_FILENAME}"))
    assert manifests
    manifest = RunManifest.model_validate_json(manifests[0].read_text("utf-8"))
    assert manifest.status == "failed"


def test_holds_out_every_nth_keyframe(make_panning_video: MakePanningVideo, tmp_path: Path) -> None:
    video = make_panning_video(frame_count=20, step_px=4)

    result = run_preprocess(
        _config(video, min_disparity_px=0.5, held_out_every=4),
        config_dir=video.parent,
        artifacts_root=tmp_path / "artifacts",
    )

    held = [record.keyframe_index for record in result.artifact.keyframes if record.held_out]
    # Every 4th by position, as the 3DGS papers evaluate.
    assert held == [index for index in range(len(result.artifact.keyframes)) if index % 4 == 0]
    assert result.artifact.selection.held_out_count == len(held)
    assert result.artifact.selection.held_out_every == 4


def test_the_split_is_deterministic(make_panning_video: MakePanningVideo, tmp_path: Path) -> None:
    video = make_panning_video(frame_count=20, step_px=4)
    config = _config(video, min_disparity_px=0.5, held_out_every=4)

    first = run_preprocess(config, config_dir=video.parent, artifacts_root=tmp_path / "a")
    second = run_preprocess(config, config_dir=video.parent, artifacts_root=tmp_path / "b")

    # The split follows from position, so no seed is involved and it cannot drift.
    assert [record.held_out for record in first.artifact.keyframes] == [
        record.held_out for record in second.artifact.keyframes
    ]


def test_no_split_marks_nothing(make_panning_video: MakePanningVideo, tmp_path: Path) -> None:
    video = make_panning_video(frame_count=12, step_px=4)

    result = run_preprocess(
        _config(video, held_out_every=None),
        config_dir=video.parent,
        artifacts_root=tmp_path / "artifacts",
    )

    assert not any(record.held_out for record in result.artifact.keyframes)
    assert result.artifact.selection.held_out_count == 0
    assert result.artifact.selection.held_out_every is None


def test_a_split_that_starves_training_is_an_error(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    video = make_panning_video(frame_count=6, step_px=4)

    # Few keyframes plus a tight split leaves too little to train on.
    with pytest.raises(PreprocessError, match="to train on"):
        run_preprocess(
            _config(video, min_disparity_px=10.0, held_out_every=2),
            config_dir=video.parent,
            artifacts_root=tmp_path / "artifacts",
        )


def test_changing_the_split_changes_the_artifact_directory(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    video = make_panning_video(frame_count=20, step_px=4)
    artifacts = tmp_path / "artifacts"

    every_four = run_preprocess(
        _config(video, min_disparity_px=0.5, held_out_every=4),
        config_dir=video.parent,
        artifacts_root=artifacts,
    )
    every_eight = run_preprocess(
        _config(video, min_disparity_px=0.5, held_out_every=8),
        config_dir=video.parent,
        artifacts_root=artifacts,
    )

    # Results under two splits are not comparable, so they cannot share a
    # directory or quietly reuse each other.
    assert every_four.artifact_dir != every_eight.artifact_dir


def test_new_source_content_moves_the_downstream_artifact(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    """The whole point of chaining: a new source must invalidate what came off it."""
    artifacts = tmp_path / "artifacts"
    video = make_panning_video(name="pan.avi", frame_count=8, step_px=6)
    config = _config(video)

    first = run_preprocess(config, config_dir=video.parent, artifacts_root=artifacts)
    video.unlink()
    make_panning_video(name="pan.avi", frame_count=8, step_px=11)
    second = run_preprocess(config, config_dir=video.parent, artifacts_root=artifacts)

    assert second.reused is False
    assert second.artifact_dir != first.artifact_dir
    assert second.artifact.ingest_hash != first.artifact.ingest_hash


def test_a_truncated_keyframe_prevents_reuse(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    video = make_panning_video(frame_count=8)
    config = _config(video)

    first = run_preprocess(config, config_dir=video.parent, artifacts_root=artifacts)
    keyframe = first.artifact_dir / first.artifact.keyframes[0].path
    with keyframe.open("rb+") as handle:
        handle.truncate(keyframe.stat().st_size // 2)

    second = run_preprocess(config, config_dir=video.parent, artifacts_root=artifacts)

    assert second.reused is False
    assert keyframe.stat().st_size == second.artifact.keyframes[0].byte_length


def test_keyframes_record_the_size_they_were_written_at(
    make_panning_video: MakePanningVideo, tmp_path: Path
) -> None:
    result = run_preprocess(
        _config(make_panning_video(frame_count=8)),
        config_dir=tmp_path,
        artifacts_root=tmp_path / "artifacts",
    )

    for record in result.artifact.keyframes:
        assert record.byte_length > 0
        assert (result.artifact_dir / record.path).stat().st_size == record.byte_length
