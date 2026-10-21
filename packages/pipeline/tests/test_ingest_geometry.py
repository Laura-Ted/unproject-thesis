"""The ingest geometry: exact, deterministic, and recorded in the artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fixture_types import MakeImageDir
from unproject.config import ExperimentConfig, FrameGeometry, IngestConfig, config_hash
from unproject.ingest.artifacts import ARTIFACT_FILENAME, IngestArtifact
from unproject.ingest.geometry import (
    GeometryError,
    apply_geometry,
    downscale,
    letterbox_square,
    output_size,
)
from unproject.ingest.stage import IngestError, ingest_identity, run_ingest


class TestDownscale:
    def test_averages_each_block_exactly(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[0:2, 0:2] = (10, 20, 30)
        image[0, 2] = 255  # one bright pixel in an otherwise black block
        out = downscale(image, 2)

        assert out.shape == (2, 3, 3)
        assert tuple(out[0, 0]) == (10, 20, 30)
        # (255 + 0 + 0 + 0 + 2) // 4 = 64: rounded half up, in integers.
        assert tuple(out[0, 1]) == (64, 64, 64)
        assert tuple(out[1, 2]) == (0, 0, 0)

    def test_is_the_same_on_every_call_and_leaves_the_input_alone(self) -> None:
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
        before = image.copy()

        assert np.array_equal(downscale(image, 4), downscale(image, 4))
        assert np.array_equal(image, before)

    def test_refuses_a_size_that_does_not_divide(self) -> None:
        with pytest.raises(GeometryError, match="does not divide"):
            downscale(np.zeros((9, 16, 3), dtype=np.uint8), 2)

    def test_factor_one_is_the_identity(self) -> None:
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        assert downscale(image, 1) is image


class TestLetterbox:
    def test_pads_a_wide_frame_top_and_bottom_centred(self) -> None:
        image = np.full((4, 10, 3), 200, dtype=np.uint8)
        out = letterbox_square(image)

        assert out.shape == (10, 10, 3)
        assert out[:3].max() == 0 and out[7:].max() == 0
        assert np.array_equal(out[3:7], image)

    def test_an_odd_difference_puts_the_extra_row_at_the_bottom(self) -> None:
        out = letterbox_square(np.full((3, 6, 3), 1, dtype=np.uint8))
        assert out.shape == (6, 6, 3)
        assert out[0].max() == 0 and out[1:4].min() == 1 and out[4:].max() == 0

    def test_a_square_is_unchanged(self) -> None:
        image = np.full((5, 5, 3), 7, dtype=np.uint8)
        assert np.array_equal(letterbox_square(image), image)


class TestApplyAndSize:
    def test_downscale_then_letterbox_matches_the_matrix_suite(self) -> None:
        image = np.full((1080, 1920, 3), 90, dtype=np.uint8)
        geometry = FrameGeometry(downscale=2, letterbox_square=True)

        out = apply_geometry(image, geometry)

        assert out.shape == (960, 960, 3)
        assert output_size(1920, 1080, geometry) == (960, 960)
        assert out[210:750].min() == 90 and out[:210].max() == 0 and out[750:].max() == 0

    def test_output_size_reports_every_geometry(self) -> None:
        assert output_size(1920, 1080, None) == (1920, 1080)
        assert output_size(1920, 1080, FrameGeometry(downscale=2)) == (960, 540)
        assert output_size(1920, 1080, FrameGeometry(letterbox_square=True)) == (1920, 1920)
        with pytest.raises(GeometryError):
            output_size(1919, 1080, FrameGeometry(downscale=2))

    def test_the_identity_geometry_is_refused_by_the_config(self) -> None:
        with pytest.raises(ValueError, match="does nothing"):
            FrameGeometry()


def _experiment(directory: Path, geometry: FrameGeometry | None) -> ExperimentConfig:
    return ExperimentConfig(
        name="fixture",
        ingest=IngestConfig(source_path=Path(directory.name), image_fps=10.0, geometry=geometry),
    )


class TestInTheStage:
    def test_frames_are_written_at_the_geometry_and_the_artifact_says_so(
        self, make_image_dir: MakeImageDir, tmp_path: Path
    ) -> None:
        directory = make_image_dir(frame_count=3, width=64, height=48)
        artifacts = tmp_path / "artifacts"
        geometry = FrameGeometry(downscale=2, letterbox_square=True)

        result = run_ingest(
            _experiment(directory, geometry), config_dir=directory.parent, artifacts_root=artifacts
        )

        artifact = IngestArtifact.model_validate_json(
            (result.artifact_dir / ARTIFACT_FILENAME).read_text("utf-8")
        )
        assert artifact.geometry == geometry
        assert artifact.frame_size() == (32, 32)
        assert (artifact.source.width, artifact.source.height) == (64, 48)
        for frame in artifact.frames:
            written = cv2.imread(str(result.artifact_dir / frame.path))
            assert written is not None
            assert written.shape == (32, 32, 3)

    def test_without_a_geometry_the_frame_size_is_the_sources(
        self, make_image_dir: MakeImageDir, tmp_path: Path
    ) -> None:
        directory = make_image_dir(frame_count=2, width=64, height=48)
        result = run_ingest(
            _experiment(directory, None), config_dir=directory.parent, artifacts_root=tmp_path / "a"
        )
        assert result.artifact.geometry is None
        assert result.artifact.frame_size() == (64, 48)

    def test_a_geometry_is_a_different_ingest_identity(
        self, make_image_dir: MakeImageDir, tmp_path: Path
    ) -> None:
        directory = make_image_dir(frame_count=2, width=64, height=48)
        plain = _experiment(directory, None).ingest
        halved = _experiment(directory, FrameGeometry(downscale=2)).ingest
        boxed = _experiment(directory, FrameGeometry(letterbox_square=True)).ingest

        identities = {
            ingest_identity(config, directory.parent) for config in (plain, halved, boxed)
        }
        assert len(identities) == 3

    def test_an_unset_geometry_does_not_change_an_old_configs_hash(self) -> None:
        # Configs written before the field existed keep the artifacts they made:
        # the hash is over the dump without the field, exactly as it was.
        config = IngestConfig(source_path=Path("clip.mp4"))
        as_before = hashlib.sha256(
            config.model_dump_json(exclude={"geometry"}).encode("utf-8")
        ).hexdigest()

        assert config_hash(config) == as_before
        assert (
            config_hash(
                IngestConfig(source_path=Path("clip.mp4"), geometry=FrameGeometry(downscale=2))
            )
            != as_before
        )

    def test_a_size_that_does_not_divide_is_an_ingest_error(
        self, make_image_dir: MakeImageDir, tmp_path: Path
    ) -> None:
        directory = make_image_dir(frame_count=2, width=63, height=48)
        with pytest.raises(IngestError, match="does not divide"):
            run_ingest(
                _experiment(directory, FrameGeometry(downscale=2)),
                config_dir=directory.parent,
                artifacts_root=tmp_path / "a",
            )

    def test_an_artifact_written_before_the_field_existed_still_reads(
        self, make_image_dir: MakeImageDir, tmp_path: Path
    ) -> None:
        directory = make_image_dir(frame_count=1, width=64, height=48)
        result = run_ingest(
            _experiment(directory, None), config_dir=directory.parent, artifacts_root=tmp_path / "a"
        )
        text = (result.artifact_dir / ARTIFACT_FILENAME).read_text("utf-8")
        data = {
            key: value
            for key, value in json.loads(text).items()
            if key not in ("geometry", "frame_width", "frame_height")
        }
        old = IngestArtifact.model_validate(data)
        assert old.frame_size() == (64, 48)
