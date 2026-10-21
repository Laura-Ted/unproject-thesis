"""Tests for the body-scale hypothesis test and the letterbox transform.
ADR-0005 WP5.

The fit is exercised on data built to a known answer - a residual that *is*
the scale ratio, one that is unrelated, one in between - so the reading rule
is checked against cases where the right reading is not in doubt. The
transform is checked pixel by pixel and annotation by annotation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from test_ground_truth_report import JOINTS, _config, _evaluation_dir
from unproject.config import ExperimentConfig
from unproject.ground_truth.config import load_ground_truth_config
from unproject.ground_truth.dataset import (
    CANONICAL_CONVENTION,
    Camera,
    DatasetRecord,
    GroundTruthError,
    GroundTruthPerson,
)
from unproject.ground_truth.hypotheses import (
    FIGURE_FILENAME,
    HYPOTHESES_FILENAME,
    Hypotheses,
    _rank,
    analyse,
    fit_line,
    predicted_relative_error,
    reading_of,
    write_hypotheses,
)
from unproject.ground_truth.metrics import PersonErrors
from unproject.ground_truth.mupots import JOINT_NAMES, SequenceIntrinsics
from unproject.ground_truth.observations import Observations, collect
from unproject.ground_truth.sequences import sequence_experiment
from unproject.ground_truth.transform import (
    letterbox_padding,
    letterbox_record,
    letterbox_sequence,
    release_derived_frames,
    write_letterboxed_frames,
)


def _scaled_observations(
    ratios: list[float],
    relative_errors: list[float],
    *,
    depth_m: float = 4.0,
    sequences: tuple[str, ...] = ("TS9",),
    focal_ratio: float = 1.0,
) -> Observations:
    """People whose scale ratio and relative depth error are exactly as given."""
    people = []
    frames = []
    for index, (ratio, relative) in enumerate(zip(ratios, relative_errors, strict=True)):
        errors = PersonErrors(
            absolute_mm=np.full(len(JOINTS), 100.0),
            root_relative_mm=np.full(len(JOINTS), 40.0),
            valid=np.ones(len(JOINTS), dtype=np.bool_),
            root_error_xyz_mm=np.array([0.0, 20.0, relative * depth_m * 1000.0]),
            depth_m=depth_m,
            scale_proxy_predicted_m=1.7 * ratio,
            scale_proxy_annotated_m=1.7,
        )
        sequence = index % len(sequences)
        people.append((sequence, index, 0, index, errors))
        frames.append((sequence, index, 1, 1, 1))
    return collect(
        JOINTS, sequences, people, frames, [(1100.0, 1100.0 * focal_ratio)] * len(sequences)
    )


class TestTheFit:
    def test_a_residual_that_is_the_scale_ratio_reads_as_supported(self) -> None:
        rng = np.random.default_rng(1)
        ratios = list(1.0 + rng.normal(0.0, 0.05, 400))
        # y = r - 1 plus a little noise: exactly the H1 prediction.
        errors = [
            r - 1.0 + float(noise)
            for r, noise in zip(ratios, rng.normal(0, 0.005, 400), strict=True)
        ]
        observations = _scaled_observations(ratios, errors)

        p, y = predicted_relative_error(observations)
        fit = fit_line(p, y, rounds=200, seed=1)

        assert fit.slope == pytest.approx(1.0, abs=0.05)
        assert fit.intercept == pytest.approx(0.0, abs=0.005)
        assert fit.r_squared is not None and fit.r_squared > 0.95
        assert fit.slope_ci_low is not None and fit.slope_ci_low < 1.0 < (fit.slope_ci_high or 0)
        assert reading_of(fit) == "supported"

    def test_an_unrelated_residual_reads_as_unsupported(self) -> None:
        rng = np.random.default_rng(2)
        ratios = list(1.0 + rng.normal(0.0, 0.05, 400))
        errors = list(-0.05 + rng.normal(0.0, 0.03, 400))
        observations = _scaled_observations(ratios, errors)

        p, y = predicted_relative_error(observations)
        fit = fit_line(p, y, rounds=200, seed=1)

        assert fit.r_squared is not None and fit.r_squared < 0.05
        assert reading_of(fit) == "unsupported"
        # The effect size is still reported: the residual is -5 %.
        assert fit.y_median == pytest.approx(-0.05, abs=0.01)

    def test_a_residual_that_half_tracks_the_ratio_reads_as_partial(self) -> None:
        rng = np.random.default_rng(3)
        ratios = list(1.0 + rng.normal(0.0, 0.05, 600))
        errors = [
            0.5 * (r - 1.0) + float(noise)
            for r, noise in zip(ratios, rng.normal(0, 0.02, 600), strict=True)
        ]
        observations = _scaled_observations(ratios, errors)

        p, y = predicted_relative_error(observations)
        fit = fit_line(p, y, rounds=200, seed=1)

        assert fit.slope == pytest.approx(0.5, abs=0.1)
        assert reading_of(fit) == "partial"

    def test_the_focal_ratio_enters_the_prediction_under_the_model_camera(self) -> None:
        """A 10 % longer estimated focal length predicts 10 % further, before
        any body-scale effect."""
        observations = _scaled_observations([1.0, 1.0, 1.0], [0.1, 0.1, 0.1], focal_ratio=1.1)

        p, _ = predicted_relative_error(observations)

        assert p == pytest.approx([0.1, 0.1, 0.1])

    def test_too_few_people_or_no_spread_is_not_testable(self) -> None:
        few = fit_line(np.array([0.1, 0.2]), np.array([0.1, 0.2]), rounds=10, seed=1)
        flat = fit_line(np.zeros(10), np.linspace(0, 1, 10), rounds=10, seed=1)

        assert few.slope is None and reading_of(few) == "not_testable"
        assert flat.slope is None and reading_of(flat) == "not_testable"

    def test_a_missing_proxy_is_left_out_of_the_fit_not_zeroed(self) -> None:
        observations = _scaled_observations([1.0, 1.1, 0.9, 1.05], [0.0, 0.1, -0.1, 0.05])
        broken = replace(
            observations,
            scale_proxy_annotated_m=np.array([1.7, np.nan, 1.7, 1.7]),
        )

        p, y = predicted_relative_error(broken)
        fit = fit_line(p, y, rounds=10, seed=1)

        assert fit.people == 3

    def test_a_schema_1_evaluation_is_refused_with_the_way_out(self) -> None:
        observations = replace(_scaled_observations([1.0] * 3, [0.0] * 3), has_scale_proxy=False)

        with pytest.raises(GroundTruthError, match="Rescore"):
            predicted_relative_error(observations)

    def test_ranks_share_their_mean_on_ties(self) -> None:
        assert list(_rank(np.array([3.0, 1.0, 3.0, 2.0]))) == [3.5, 1.0, 3.5, 2.0]

    def test_the_bootstrap_is_seeded(self) -> None:
        rng = np.random.default_rng(4)
        p = rng.normal(0, 0.1, 50)
        y = p + rng.normal(0, 0.02, 50)

        assert fit_line(p, y, rounds=100, seed=7) == fit_line(p, y, rounds=100, seed=7)
        assert fit_line(p, y, rounds=100, seed=7) != fit_line(p, y, rounds=100, seed=8)


class TestTheAnalysisOnAnEvaluation:
    def test_it_writes_the_three_files_deterministically_and_labels_the_post_hoc(
        self, tmp_path: Path
    ) -> None:
        rng = np.random.default_rng(5)
        ratios = list(1.0 + rng.normal(0.0, 0.05, 80))
        errors = [r - 1.0 for r in ratios]
        observations = _scaled_observations(ratios, errors, sequences=("TS1", "TS9"))
        directory = _evaluation_dir(
            tmp_path, "one", observations, config=_config(camera="annotation_derived")
        )

        first = analyse(directory)
        written = write_hypotheses(first, directory, tmp_path / "out1")
        write_hypotheses(analyse(directory), directory, tmp_path / "out2")

        assert first.reading == "supported"
        assert first.kind == "confirmatory"
        assert first.shape_split.kind == "post_hoc"
        assert first.vertical_offset.kind == "post_hoc"
        assert {row.shape for row in first.per_sequence} == {"square", "wide"}
        assert [p.name for p in written] == [
            HYPOTHESES_FILENAME,
            "hypotheses-per-sequence.csv",
            FIGURE_FILENAME,
        ]
        for name in (HYPOTHESES_FILENAME, "hypotheses-per-sequence.csv", FIGURE_FILENAME):
            assert (tmp_path / "out1" / name).read_bytes() == (
                tmp_path / "out2" / name
            ).read_bytes()
        assert (
            Hypotheses.model_validate_json(
                (tmp_path / "out1" / HYPOTHESES_FILENAME).read_text("utf-8")
            )
            == first
        )

    def test_the_reading_rule_is_written_into_the_file(self, tmp_path: Path) -> None:
        observations = _scaled_observations([1.0, 1.1, 0.9, 1.05, 0.95], [0.0] * 5)
        directory = _evaluation_dir(tmp_path, "one", observations)

        hypotheses = analyse(directory)

        assert "declared before the data" in hypotheses.reading_rule
        assert "0.7" in hypotheses.reading_rule
        assert hypotheses.bootstrap_seed == 20260915


def _record(width: int, height: int, sequence: str = "TS9") -> DatasetRecord:
    joints = np.zeros((len(JOINT_NAMES), 3))
    joints[:, 2] = 3.0
    two = np.stack([np.full(len(JOINT_NAMES), 100.0), np.full(len(JOINT_NAMES), 200.0)], axis=1)
    return DatasetRecord(
        dataset_name="mupots-3d",
        release_identity="fixture",
        sample_id=f"{sequence}/000000",
        sequence_id=sequence,
        frame_index=0,
        image_path=Path("img_000000.jpg"),
        image_width=width,
        image_height=height,
        people=(
            GroundTruthPerson(
                ground_truth_person_id="slot0",
                joints_3d=joints,
                joint_names=JOINT_NAMES,
                joint_valid=np.ones(len(JOINT_NAMES), dtype=np.bool_),
                joints_2d=two,
            ),
        ),
        camera=Camera(fx=1100.0, fy=1100.0, cx=960.0, cy=528.0),
        convention=CANONICAL_CONVENTION,
    )


class TestTheLetterbox:
    def test_the_padding_is_symmetric_and_the_odd_row_goes_below(self) -> None:
        assert letterbox_padding(1920, 1080) == (420, 420)
        assert letterbox_padding(11, 8) == (1, 2)
        assert letterbox_padding(8, 8) == (0, 0)

    def test_a_tall_frame_cannot_be_letterboxed_to_a_square(self) -> None:
        with pytest.raises(GroundTruthError, match="taller"):
            letterbox_padding(1080, 1920)

    def test_the_2d_joints_and_the_principal_point_move_down_and_the_3d_does_not(self) -> None:
        record = _record(1920, 1080)

        moved = letterbox_record(record, 420)

        person = moved.people[0]
        assert person.joints_2d is not None
        assert person.joints_2d[0].tolist() == [100.0, 620.0]
        assert np.array_equal(person.joints_3d, record.people[0].joints_3d)
        assert (moved.image_width, moved.image_height) == (1920, 1920)
        assert moved.camera is not None and moved.camera.cy == 948.0 and moved.camera.cx == 960.0

    def test_frames_are_padded_with_black_and_reused_when_complete(self, tmp_path: Path) -> None:
        source = tmp_path / "TS9"
        source.mkdir()
        frame = np.full((6, 10, 3), 200, dtype=np.uint8)
        assert cv2.imwrite(str(source / "img_000000.jpg"), frame)
        record = replace(_record(10, 6), image_path=source / "img_000000.jpg")
        target = tmp_path / "derived"

        write_letterboxed_frames([record], target, top=2, bottom=2)
        stamp = (target / "letterbox.txt").stat().st_mtime_ns
        write_letterboxed_frames([record], target, top=2, bottom=2)

        padded = cv2.imread(str(target / "img_000000.png"), cv2.IMREAD_COLOR)
        assert padded is not None
        assert padded.shape == (10, 10, 3)
        assert padded[:2].max() == 0 and padded[-2:].max() == 0
        assert padded[2:8].min() >= 195
        assert (target / "letterbox.txt").stat().st_mtime_ns == stamp

    def test_a_square_sequence_is_returned_untouched(self, tmp_path: Path) -> None:
        record = replace(_record(2048, 2048, "TS1"), image_path=tmp_path / "TS1" / "img_000000.jpg")
        intrinsics = SequenceIntrinsics(
            camera=Camera(1500.0, 1500.0, 1024.0, 1024.0), residual_px=0.5, samples=10
        )

        frames_dir, records, moved, padding = letterbox_sequence(
            [record], intrinsics, artifacts_root=tmp_path, dataset="mupots-3d", sequence_id="TS1"
        )

        assert padding == 0
        assert records == [record]
        assert moved == intrinsics
        assert frames_dir == tmp_path / "TS1"
        assert not (tmp_path / "_derived").exists()

    def test_a_wide_sequence_is_padded_moved_and_releasable(self, tmp_path: Path) -> None:
        source = tmp_path / "TS9"
        source.mkdir()
        assert cv2.imwrite(str(source / "img_000000.jpg"), np.full((6, 10, 3), 100, dtype=np.uint8))
        record = replace(_record(10, 6), image_path=source / "img_000000.jpg")
        intrinsics = SequenceIntrinsics(
            camera=Camera(9.0, 9.0, 5.0, 3.0), residual_px=0.1, samples=17
        )

        frames_dir, records, moved, padding = letterbox_sequence(
            [record], intrinsics, artifacts_root=tmp_path, dataset="mupots-3d", sequence_id="TS9"
        )

        assert padding == 2
        assert frames_dir == tmp_path / "_derived" / "mupots-3d" / "letterbox-square" / "TS9"
        assert records[0].image_path == frames_dir / "img_000000.png"
        assert records[0].image_height == 10
        assert moved.camera.cy == 5.0
        assert release_derived_frames(tmp_path, "mupots-3d", "TS9") > 0
        assert not frames_dir.exists()
        assert release_derived_frames(tmp_path, "mupots-3d", "TS9") == 0

    def test_the_transform_is_part_of_the_config_and_the_artifact_name(self) -> None:
        base = ExperimentConfig.model_validate(
            {"name": "mupots", "ingest": {"source_path": "x", "image_fps": 30.0}}
        )

        built = sequence_experiment(
            base, sequence_id="TS9", frames_dir=Path("/d"), transform="letterbox_square"
        )

        assert built.name == "mupots-letterbox-square-TS9"
        config = load_ground_truth_config(Path("experiments/ground-truth/mupots-3d.toml"))
        assert config.frame_transform == "none"
        transformed = config.model_copy(update={"frame_transform": "letterbox_square"})
        from unproject.ground_truth.config import identity_hash

        assert identity_hash(
            config, inventory_hash="a" * 64, mapping_hash="b" * 64
        ) != identity_hash(transformed, inventory_hash="a" * 64, mapping_hash="b" * 64)
