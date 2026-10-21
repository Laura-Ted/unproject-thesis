"""Tests for the ingest stage: selection, artifact contract, resumability."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

from fixture_types import MakeImageDir, MakeVideo
from unproject.config import ExperimentConfig, IngestConfig
from unproject.ingest.artifacts import (
    ARTIFACT_FILENAME,
    INGEST_SCHEMA_VERSION,
    IngestArtifact,
)
from unproject.ingest.stage import (
    IngestError,
    ingest_artifact_dir,
    ingest_identity,
    run_ingest,
)
from unproject.run.manifest import MANIFEST_FILENAME, RunManifest


def _config(
    video: Path,
    *,
    image_fps: float | None = None,
    frame_stride: int = 1,
    start_time_seconds: float | None = None,
    end_time_seconds: float | None = None,
    max_frames: int | None = None,
) -> ExperimentConfig:
    """An experiment with a relative source path, as a real config has."""
    return ExperimentConfig(
        name="fixture",
        seed=7,
        ingest=IngestConfig(
            source_path=Path(video.name),
            image_fps=image_fps,
            frame_stride=frame_stride,
            start_time_seconds=start_time_seconds,
            end_time_seconds=end_time_seconds,
            max_frames=max_frames,
        ),
    )


def test_writes_every_frame_by_default(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=6)
    artifacts = tmp_path / "artifacts"

    result = run_ingest(_config(video), config_dir=video.parent, artifacts_root=artifacts)

    assert result.reused is False
    assert result.artifact.selection.frames_written == 6
    assert len(list((result.artifact_dir / "frames").glob("*.png"))) == 6


def test_frames_are_readable_and_keep_their_size(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=3, width=64, height=48)

    result = run_ingest(
        _config(video), config_dir=video.parent, artifacts_root=tmp_path / "artifacts"
    )

    first = result.artifact.frames[0]
    image = cv2.imread(str(result.artifact_dir / first.path))
    assert image is not None
    assert image.shape == (48, 64, 3)


def test_stride_decimates_uniformly(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=10)

    result = run_ingest(
        _config(video, frame_stride=3),
        config_dir=video.parent,
        artifacts_root=tmp_path / "artifacts",
    )

    assert [frame.source_index for frame in result.artifact.frames] == [0, 3, 6, 9]
    assert result.artifact.selection.frames_decoded == 10


def test_time_window_clips_both_ends(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=10, fps=10.0)

    result = run_ingest(
        _config(video, start_time_seconds=0.2, end_time_seconds=0.5),
        config_dir=video.parent,
        artifacts_root=tmp_path / "artifacts",
    )

    assert [frame.source_index for frame in result.artifact.frames] == [2, 3, 4]
    # The window ends the iteration, so the tail is never decoded.
    assert result.artifact.selection.frames_decoded < 10


def test_max_frames_truncates_and_says_so(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=10)

    result = run_ingest(
        _config(video, max_frames=4),
        config_dir=video.parent,
        artifacts_root=tmp_path / "artifacts",
    )

    assert result.artifact.selection.frames_written == 4
    assert result.artifact.selection.truncated is True


def test_max_frames_beyond_the_video_is_not_truncation(
    make_video: MakeVideo, tmp_path: Path
) -> None:
    video = make_video(frame_count=4)

    result = run_ingest(
        _config(video, max_frames=99),
        config_dir=video.parent,
        artifacts_root=tmp_path / "artifacts",
    )

    assert result.artifact.selection.truncated is False


def test_empty_selection_is_an_error(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=5, fps=10.0)

    with pytest.raises(IngestError, match="no frames selected"):
        run_ingest(
            _config(video, start_time_seconds=60.0),
            config_dir=video.parent,
            artifacts_root=tmp_path / "artifacts",
        )


def test_artifact_round_trips_through_disk(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=3)

    result = run_ingest(
        _config(video), config_dir=video.parent, artifacts_root=tmp_path / "artifacts"
    )

    raw = (result.artifact_dir / ARTIFACT_FILENAME).read_text("utf-8")
    assert IngestArtifact.model_validate_json(raw) == result.artifact
    assert json.loads(raw)["schema_version"] == INGEST_SCHEMA_VERSION


def test_manifest_records_the_run(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=3)
    config = _config(video)

    result = run_ingest(config, config_dir=video.parent, artifacts_root=tmp_path / "artifacts")

    manifest = RunManifest.model_validate_json(
        (result.artifact_dir / MANIFEST_FILENAME).read_text("utf-8")
    )
    assert manifest.stage == "ingest"
    assert manifest.status == "ok"
    assert manifest.error is None
    assert manifest.experiment == "fixture"
    assert manifest.seed == 7
    assert manifest.config_hash == ingest_identity(config.ingest, video.parent)
    assert manifest.wall_clock_seconds >= 0.0
    assert manifest.peak_memory_bytes > 0
    assert "unproject-pipeline" in manifest.package_versions


def test_second_run_reuses_the_artifact(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=3)
    artifacts = tmp_path / "artifacts"

    first = run_ingest(_config(video), config_dir=video.parent, artifacts_root=artifacts)
    second = run_ingest(_config(video), config_dir=video.parent, artifacts_root=artifacts)

    assert second.reused is True
    assert second.artifact == first.artifact


def test_overwrite_forces_a_re_decode(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=3)
    artifacts = tmp_path / "artifacts"

    run_ingest(_config(video), config_dir=video.parent, artifacts_root=artifacts)
    second = run_ingest(
        _config(video), config_dir=video.parent, artifacts_root=artifacts, overwrite=True
    )

    assert second.reused is False


def test_different_ingest_config_lands_in_a_different_directory(
    make_video: MakeVideo, tmp_path: Path
) -> None:
    video = make_video(frame_count=6)
    artifacts = tmp_path / "artifacts"

    default = run_ingest(_config(video), config_dir=video.parent, artifacts_root=artifacts)
    strided = run_ingest(
        _config(video, frame_stride=2), config_dir=video.parent, artifacts_root=artifacts
    )

    # Keyed by the stage config hash, so both survive.
    assert default.artifact_dir != strided.artifact_dir
    assert default.artifact_dir.is_dir()
    assert strided.artifact_dir.is_dir()


def _failing_imwrite(*args: object, **kwargs: object) -> bool:
    """Stand-in for a disk that will not take the frame."""
    return False


def _artifact_dir_of(config: ExperimentConfig, artifacts_root: Path, config_dir: Path) -> Path:
    return ingest_artifact_dir(
        artifacts_root, config.name, ingest_identity(config.ingest, config_dir)
    )


def _manifest_of(config: ExperimentConfig, artifacts_root: Path, config_dir: Path) -> RunManifest:
    artifact_dir = _artifact_dir_of(config, artifacts_root, config_dir)
    return RunManifest.model_validate_json((artifact_dir / MANIFEST_FILENAME).read_text("utf-8"))


def test_a_failed_stage_still_writes_a_manifest(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=5, fps=10.0)
    config = _config(video, start_time_seconds=60.0)
    artifacts = tmp_path / "artifacts"

    with pytest.raises(IngestError):
        run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)

    manifest = _manifest_of(config, artifacts, video.parent)
    assert manifest.status == "failed"
    assert "no frames selected" in (manifest.error or "")
    assert manifest.wall_clock_seconds >= 0.0
    assert manifest.peak_memory_bytes > 0


def test_a_failed_stage_leaves_no_artifact(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=5, fps=10.0)
    config = _config(video, start_time_seconds=60.0)
    artifacts = tmp_path / "artifacts"

    with pytest.raises(IngestError):
        run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)

    artifact_dir = _artifact_dir_of(config, artifacts, video.parent)
    # The manifest records the attempt; the missing contract file is what keeps
    # the directory from reading as a finished run.
    assert (artifact_dir / MANIFEST_FILENAME).is_file()
    assert not (artifact_dir / ARTIFACT_FILENAME).exists()


def test_a_failed_stage_is_not_reused_by_the_next_run(
    make_video: MakeVideo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = make_video(frame_count=3)
    config = _config(video)
    artifacts = tmp_path / "artifacts"

    monkeypatch.setattr(cv2, "imwrite", _failing_imwrite)
    with pytest.raises(IngestError, match="failed to write frame"):
        run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    monkeypatch.undo()

    result = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)

    assert result.reused is False
    assert result.artifact.selection.frames_written == 3


def test_the_manifest_of_a_retried_run_reports_success(
    make_video: MakeVideo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = make_video(frame_count=3)
    config = _config(video)
    artifacts = tmp_path / "artifacts"

    monkeypatch.setattr(cv2, "imwrite", _failing_imwrite)
    with pytest.raises(IngestError):
        run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    monkeypatch.undo()
    run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)

    # The retry overwrites the failed manifest rather than accumulating one per try.
    assert _manifest_of(config, artifacts, video.parent).status == "ok"


def test_ingests_a_directory_of_images(make_image_dir: MakeImageDir, tmp_path: Path) -> None:
    directory = make_image_dir(frame_count=5)
    config = _config(directory, image_fps=10.0)

    result = run_ingest(config, config_dir=directory.parent, artifacts_root=tmp_path / "artifacts")

    assert result.artifact.source.kind == "image_directory"
    assert result.artifact.selection.frames_written == 5
    assert len(list((result.artifact_dir / "frames").glob("*.png"))) == 5


def test_selection_works_the_same_on_a_directory(
    make_image_dir: MakeImageDir, tmp_path: Path
) -> None:
    directory = make_image_dir(frame_count=10)

    result = run_ingest(
        _config(directory, image_fps=10.0, frame_stride=3),
        config_dir=directory.parent,
        artifacts_root=tmp_path / "artifacts",
    )

    # Stride and window are stage logic, not source logic, so both inputs behave
    # identically here.
    assert [frame.source_index for frame in result.artifact.frames] == [0, 3, 6, 9]


def test_a_directory_without_image_fps_is_rejected(
    make_image_dir: MakeImageDir, tmp_path: Path
) -> None:
    directory = make_image_dir(frame_count=3)

    with pytest.raises(IngestError, match="image_fps"):
        run_ingest(
            _config(directory),
            config_dir=directory.parent,
            artifacts_root=tmp_path / "artifacts",
        )


def test_the_two_sources_land_in_different_artifact_directories(
    make_video: MakeVideo, make_image_dir: MakeImageDir, tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    video = make_video(frame_count=4)
    directory = make_image_dir(frame_count=4)

    from_video = run_ingest(_config(video), config_dir=video.parent, artifacts_root=artifacts)
    from_images = run_ingest(
        _config(directory, image_fps=10.0),
        config_dir=directory.parent,
        artifacts_root=artifacts,
    )

    assert from_video.artifact_dir != from_images.artifact_dir
    assert from_video.artifact.source.kind == "video"
    assert from_images.artifact.source.kind == "image_directory"


def test_an_artifact_from_an_older_schema_is_re_run(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=3)
    config = _config(video)
    artifacts = tmp_path / "artifacts"

    first = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    stale = json.loads((first.artifact_dir / ARTIFACT_FILENAME).read_text("utf-8"))
    stale["schema_version"] = INGEST_SCHEMA_VERSION - 1
    (first.artifact_dir / ARTIFACT_FILENAME).write_text(json.dumps(stale), encoding="utf-8")

    second = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)

    # Artifacts are reproducible, so an unreadable one is regenerated rather
    # than turned into an error the user has to clear by hand.
    assert second.reused is False
    assert second.artifact.schema_version == INGEST_SCHEMA_VERSION


def test_a_corrupt_artifact_is_re_run(make_video: MakeVideo, tmp_path: Path) -> None:
    video = make_video(frame_count=3)
    config = _config(video)
    artifacts = tmp_path / "artifacts"

    first = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    (first.artifact_dir / ARTIFACT_FILENAME).write_text("{ not json", encoding="utf-8")

    assert run_ingest(config, config_dir=video.parent, artifacts_root=artifacts).reused is False


def test_a_different_video_at_the_same_path_is_a_different_run(
    make_video: MakeVideo, tmp_path: Path
) -> None:
    """A path is not an input identity: the bytes behind it are."""
    artifacts = tmp_path / "artifacts"
    video = make_video(name="clip.avi", frame_count=4)
    config = _config(video)

    first = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    video.unlink()
    make_video(name="clip.avi", frame_count=7)
    second = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)

    assert second.reused is False
    assert second.artifact_dir != first.artifact_dir
    # The first run is still addressable, not overwritten by the second.
    assert first.artifact_dir.is_dir()
    assert second.artifact.selection.frames_written == 7


def test_replacing_one_image_in_a_directory_is_a_different_run(
    make_image_dir: MakeImageDir, tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    directory = make_image_dir(frame_count=4)
    config = _config(directory, image_fps=10.0)

    first = run_ingest(config, config_dir=directory.parent, artifacts_root=artifacts)
    replacement = np.full((48, 64, 3), 200, dtype=np.uint8)
    assert cv2.imwrite(str(directory / "00002.png"), replacement)
    second = run_ingest(config, config_dir=directory.parent, artifacts_root=artifacts)

    assert second.reused is False
    assert second.artifact_dir != first.artifact_dir


def test_the_same_content_and_config_reuses_the_existing_output(
    make_video: MakeVideo, tmp_path: Path
) -> None:
    """The other half of the contract: identity that never matches is useless."""
    artifacts = tmp_path / "artifacts"
    video = make_video(frame_count=4)
    config = _config(video)

    first = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    written_at = (first.artifact_dir / ARTIFACT_FILENAME).stat().st_mtime_ns
    second = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)

    assert second.reused is True
    assert second.artifact_dir == first.artifact_dir
    assert (second.artifact_dir / ARTIFACT_FILENAME).stat().st_mtime_ns == written_at


def test_a_missing_frame_prevents_reuse(make_video: MakeVideo, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    video = make_video(frame_count=3)
    config = _config(video)

    first = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    (first.artifact_dir / first.artifact.frames[1].path).unlink()

    second = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)

    # The descriptor still reads, which is exactly why it cannot be trusted alone.
    assert second.reused is False
    assert (second.artifact_dir / second.artifact.frames[1].path).is_file()


def test_a_truncated_frame_prevents_reuse(make_video: MakeVideo, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    video = make_video(frame_count=3)
    config = _config(video)

    first = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    frame = first.artifact_dir / first.artifact.frames[0].path
    with frame.open("rb+") as handle:
        handle.truncate(frame.stat().st_size // 2)

    second = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)

    assert second.reused is False
    assert (
        second.artifact_dir / second.artifact.frames[0].path
    ).stat().st_size == second.artifact.frames[0].byte_length


def test_frames_record_the_size_they_were_written_at(make_video: MakeVideo, tmp_path: Path) -> None:
    result = run_ingest(
        _config(make_video(frame_count=3)),
        config_dir=tmp_path,
        artifacts_root=tmp_path / "artifacts",
    )

    for record in result.artifact.frames:
        assert record.byte_length > 0
        assert (result.artifact_dir / record.path).stat().st_size == record.byte_length


def test_a_failed_run_does_not_destroy_the_artifact_already_there(
    make_video: MakeVideo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--overwrite on a run that then dies must not cost the good artifact."""
    artifacts = tmp_path / "artifacts"
    video = make_video(frame_count=3)
    config = _config(video)

    first = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    manifest_before = (first.artifact_dir / MANIFEST_FILENAME).read_text("utf-8")

    monkeypatch.setattr(cv2, "imwrite", _failing_imwrite)
    with pytest.raises(IngestError, match="failed to write frame"):
        run_ingest(config, config_dir=video.parent, artifacts_root=artifacts, overwrite=True)
    monkeypatch.undo()

    # Still complete, still reusable, and still describing the run that made it.
    retried = run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    assert retried.reused is True
    assert retried.artifact == first.artifact
    assert (first.artifact_dir / MANIFEST_FILENAME).read_text("utf-8") == manifest_before


def test_a_run_that_dies_midway_leaves_nothing_that_reads_as_complete(
    make_video: MakeVideo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    video = make_video(frame_count=6)
    config = _config(video)
    written: list[str] = []

    # Through a permissive alias: the real signature is overloaded, and the
    # stand-in only needs to count calls and pass them on.
    real_imwrite = cast(Callable[..., bool], cv2.imwrite)

    def _dies_after_two(*args: object, **kwargs: object) -> bool:
        if len(written) >= 2:
            raise OSError("the disk is full")
        written.append(str(args[0]))
        return real_imwrite(*args, **kwargs)

    monkeypatch.setattr(cv2, "imwrite", _dies_after_two)
    with pytest.raises(OSError, match="the disk is full"):
        run_ingest(config, config_dir=video.parent, artifacts_root=artifacts)
    monkeypatch.undo()

    artifact_dir = _artifact_dir_of(config, artifacts, video.parent)
    # The manifest is the record of the attempt; without a descriptor beside it
    # the directory cannot be mistaken for a finished run.
    assert (artifact_dir / MANIFEST_FILENAME).is_file()
    assert not (artifact_dir / ARTIFACT_FILENAME).exists()
    assert run_ingest(config, config_dir=video.parent, artifacts_root=artifacts).reused is False


def test_no_staging_directory_survives_a_finished_run(
    make_video: MakeVideo, tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    video = make_video(frame_count=3)

    result = run_ingest(_config(video), config_dir=video.parent, artifacts_root=artifacts)

    assert [entry.name for entry in result.artifact_dir.parent.iterdir()] == [
        result.artifact_dir.name
    ]
