"""Tests for the derived summary, its regeneration, the final-run gate and the
comparison of two evaluations. ADR-0005 WP4.

Everything here runs on hand-built observations. What has to hold: the summary
is a pure view of the observations, regenerating the derived files reproduces
the run byte for byte, a final run refuses what it must, and a comparison
tells "same numbers, new labels" from "different numbers".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from unproject.ground_truth import evaluate as evaluate_module
from unproject.ground_truth.artifacts import (
    OBSERVATIONS_FILENAME,
    BodyArtifactProvenance,
    DatasetIdentity,
    Environment,
    EvaluationDescriptor,
    MatchingIdentity,
    describe_file,
    read_descriptor,
    staged_evaluation,
    write_descriptor,
)
from unproject.ground_truth.compare import (
    _walk,
    compare_evaluations,
    compare_numbers,
    describe,
    write_paired,
)
from unproject.ground_truth.config import GroundTruthConfig, MetricThresholds
from unproject.ground_truth.dataset import GroundTruthError
from unproject.ground_truth.evaluate import (
    check_final_bodies,
    check_final_preconditions,
    regenerate_derived,
    write_derived,
)
from unproject.ground_truth.metrics import GROSS_DEPTH_ERROR_MM, PersonErrors, robust_depth
from unproject.ground_truth.observations import Observations, collect
from unproject.ground_truth.report import SUMMARY_FILENAME, EvaluationSummary, build_summary

JOINTS = ("pelvis", "left_knee", "right_knee")
_CHECKPOINT = "f" * 64


def _errors(depth_error_mm: float, depth_m: float = 3.0, absolute: float = 100.0) -> PersonErrors:
    return PersonErrors(
        absolute_mm=np.full(len(JOINTS), absolute),
        root_relative_mm=np.full(len(JOINTS), 40.0),
        valid=np.ones(len(JOINTS), dtype=np.bool_),
        root_error_xyz_mm=np.array([5.0, -3.0, depth_error_mm]),
        depth_m=depth_m,
        scale_proxy_predicted_m=1.6,
        scale_proxy_annotated_m=1.7,
    )


def _observations(depth_errors: list[float], sequences: tuple[str, ...] = ("TS9",)) -> Observations:
    people = [
        (index % len(sequences), index, 0, index, _errors(error))
        for index, error in enumerate(depth_errors)
    ]
    frames = [
        (sequence, index, 1, 1, 1)
        for index, sequence in enumerate(
            index % len(sequences) for index in range(len(depth_errors))
        )
    ]
    return collect(JOINTS, sequences, people, frames, [(1100.0, 1300.0)] * len(sequences))


def _config(**overrides: object) -> GroundTruthConfig:
    defaults: dict[str, object] = {
        "name": "x",
        "dataset": "mupots-3d",
        "dataset_root": "/nowhere",
        "release": "r",
        "experiment": "e.toml",
        "mapping": "m.toml",
    }
    return GroundTruthConfig.model_validate(defaults | overrides)


def _matching() -> MatchingIdentity:
    return MatchingIdentity(
        matcher="mupots_greedy_2d_40px", parameter="radius_px_per_axis", value=40.0
    )


def _evaluation_dir(
    tmp_path: Path,
    name: str,
    observations: Observations,
    *,
    config: GroundTruthConfig | None = None,
) -> Path:
    """A complete evaluation directory, as `evaluate` would leave it."""
    config = config or _config()
    target = tmp_path / name
    with staged_evaluation(target) as staging:
        observations.save(staging / OBSERVATIONS_FILENAME)
        references = [describe_file(staging / OBSERVATIONS_FILENAME, relative_to=staging)]
        references.extend(
            write_derived(
                observations,
                config,
                staging,
                evaluation_id=name * 8,
                matching=_matching(),
                final=False,
            )
        )
        started = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
        write_descriptor(
            EvaluationDescriptor(
                evaluation_id=name * 8,
                started_at=started,
                finished_at=started,
                config=config,
                config_hash="a" * 64,
                mapping_identifier="m-v1",
                mapping_sha256="b" * 64,
                dataset=DatasetIdentity(
                    name="mupots-3d",
                    release="r",
                    split="test",
                    inventory_sha256="c" * 64,
                    file_count=1,
                    byte_count=1,
                ),
                environment=Environment(
                    machine="arm64",
                    platform="Darwin",
                    python_version="3.12",
                    device="mps",
                    package_versions={},
                    git_commit="d" * 40,
                    git_dirty=False,
                ),
                metric_protocol="unproject_mupots_14j_v1",
                matching=_matching(),
                camera_source=config.camera,
                checkpoint_sha256=_CHECKPOINT,
                coverage=observations.coverage,
                references=references,
            ),
            staging,
        )
    return target


class TestRobustDepth:
    def test_median_and_quartiles_are_values_that_were_measured(self) -> None:
        depths = np.full(10, 3.0)
        errors = np.zeros((10, 3))
        errors[:, 2] = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 4000]

        robust = robust_depth(depths, errors)

        assert robust.people == 10
        assert robust.median_mm == -10.0
        assert robust.p25_mm == -30.0
        assert robust.p75_mm == 30.0
        assert robust.iqr_mm == 60.0
        assert robust.p90_mm == 40.0
        # The mean keeps the outlier; the median does not notice it.
        assert robust.mean_mm == pytest.approx(395.0)

    def test_outliers_are_counted_under_both_definitions(self) -> None:
        depths = np.full(6, 4.0)
        errors = np.zeros((6, 3))
        errors[:, 2] = [0, 10, -10, 20, -20, GROSS_DEPTH_ERROR_MM + 1]

        robust = robust_depth(depths, errors)

        assert robust.gross_outliers == 1
        assert robust.mad_outliers == 1

    def test_the_relative_median_is_error_over_distance(self) -> None:
        depths = np.array([2.0, 4.0, 8.0])
        errors = np.zeros((3, 3))
        errors[:, 2] = [-100.0, -200.0, -400.0]

        assert robust_depth(depths, errors).relative_median == pytest.approx(-0.05)

    def test_nobody_gives_nan_and_zero_outliers(self) -> None:
        robust = robust_depth(np.zeros(0), np.zeros((0, 3)))

        assert robust.people == 0
        assert np.isnan(robust.median_mm)
        assert robust.gross_outliers == 0

    def test_a_shape_mismatch_is_refused(self) -> None:
        with pytest.raises(GroundTruthError, match="against"):
            robust_depth(np.zeros(2), np.zeros((3, 3)))


class TestTheSummary:
    def _summary(self, observations: Observations, **overrides: object) -> EvaluationSummary:
        return build_summary(
            observations,
            MetricThresholds(depth_bins_metres=[2.5, 3.5]),
            evaluation_id="abc",
            metric_protocol="unproject_mupots_14j_v1",
            matching=_matching(),
            camera_source="model_estimated",
            final=False,
        )

    def test_it_names_what_it_is(self) -> None:
        summary = self._summary(_observations([10.0, 20.0]))

        assert summary.metric_protocol == "unproject_mupots_14j_v1"
        assert summary.matching.value == 40.0
        assert summary.camera_source == "model_estimated"
        assert summary.final is False
        assert summary.coverage.matched == 2

    def test_the_depth_distribution_travels_beside_the_mean(self) -> None:
        summary = self._summary(_observations([-100.0, -50.0, 0.0, 50.0, 3000.0]))

        assert summary.root_bias_xyz_mm[2] == pytest.approx(580.0)
        assert summary.root_bias_median_xyz_mm[2] == 0.0
        assert summary.depth.median_mm == 0.0
        assert summary.depth.gross_outliers == 1
        # The one bin everyone lands in carries its own distribution.
        [bin_] = [b for b in summary.depth_bins if b.people]
        assert bin_.distribution.people == 5

    def test_nan_becomes_null_never_a_string(self) -> None:
        summary = self._summary(_observations([]))

        text = summary.model_dump_json()
        assert "NaN" not in text
        assert summary.recall is None
        assert summary.depth.median_mm is None
        # And it reads back.
        assert EvaluationSummary.model_validate_json(text) == summary

    def test_per_sequence_rows_carry_the_focal_lengths_and_the_median(self) -> None:
        summary = self._summary(_observations([10.0, 30.0, 20.0, 40.0], sequences=("TS1", "TS2")))

        rows = {row.sequence: row for row in summary.per_sequence}
        assert rows["TS1"].matched == 2
        assert rows["TS1"].root_bias_z_mm == pytest.approx(15.0)
        assert rows["TS1"].root_bias_z_median_mm == pytest.approx(15.0)
        assert rows["TS2"].focal_ratio_estimated_over_derived == pytest.approx(1300 / 1100)

    def test_it_is_deterministic(self) -> None:
        observations = _observations([1.0, 2.0, 3.0])

        assert (
            self._summary(observations).model_dump_json()
            == self._summary(observations).model_dump_json()
        )


class TestRegeneration:
    def test_the_derived_files_reproduce_the_run_byte_for_byte(self, tmp_path: Path) -> None:
        directory = _evaluation_dir(tmp_path, "one", _observations([10.0, -20.0, 30.0]))
        before = {
            path.name: path.read_bytes()
            for path in directory.rglob("*")
            if path.is_file() and path.name != "evaluation.json"
        }

        descriptor, changed = regenerate_derived(directory)

        assert changed == []
        after = {
            path.name: path.read_bytes()
            for path in directory.rglob("*")
            if path.is_file() and path.name != "evaluation.json"
        }
        assert after == before
        # The descriptor still vouches for every file it names.
        assert read_descriptor(directory).references == descriptor.references
        assert any(reference.path == SUMMARY_FILENAME for reference in descriptor.references)

    def test_a_changed_view_is_named(self, tmp_path: Path) -> None:
        """If a derived file cannot be reproduced, say which one; the observations
        are the record and the file is a stale view."""
        directory = _evaluation_dir(tmp_path, "one", _observations([10.0, -20.0]))
        (directory / SUMMARY_FILENAME).write_text("{}", encoding="utf-8")
        stale = describe_file(directory / SUMMARY_FILENAME, relative_to=directory)
        # Keep the descriptor consistent with the stale file so it still reads.
        original = EvaluationDescriptor.model_validate_json(
            (directory / "evaluation.json").read_text("utf-8")
        )
        write_descriptor(
            original.model_copy(
                update={
                    "references": [
                        stale if r.path == SUMMARY_FILENAME else r for r in original.references
                    ]
                }
            ),
            directory,
        )

        _, changed = regenerate_derived(directory)

        assert changed == [SUMMARY_FILENAME]
        assert json.loads((directory / SUMMARY_FILENAME).read_text("utf-8"))["evaluation_id"]


class TestTheFinalGate:
    def test_a_dirty_tree_is_refused_before_anything_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(evaluate_module, "git_commit", lambda: "e" * 40)
        monkeypatch.setattr(evaluate_module, "git_dirty", lambda: True)

        with pytest.raises(GroundTruthError, match="dirty"):
            check_final_preconditions(None)

    def test_no_checkout_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(evaluate_module, "git_commit", lambda: None)

        with pytest.raises(GroundTruthError, match="not a checkout"):
            check_final_preconditions(None)

    def test_a_subset_is_not_the_benchmark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(evaluate_module, "git_commit", lambda: "e" * 40)
        monkeypatch.setattr(evaluate_module, "git_dirty", lambda: False)

        with pytest.raises(GroundTruthError, match="subset"):
            check_final_preconditions(["TS9"])
        check_final_preconditions(None)

    def test_bodies_from_a_dirty_or_unknown_tree_are_refused(self) -> None:
        clean = BodyArtifactProvenance(
            experiment="mupots-TS1", artifact_hash="abc", git_commit="e" * 40, git_dirty=False
        )
        dirty = clean.model_copy(update={"git_dirty": True})
        unknown = clean.model_copy(update={"git_dirty": None, "git_commit": None})

        check_final_bodies(clean)
        with pytest.raises(GroundTruthError, match="dirty tree"):
            check_final_bodies(dirty)
        with pytest.raises(GroundTruthError, match="unknown"):
            check_final_bodies(unknown)

    def test_evaluate_refuses_a_dirty_tree_before_touching_the_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The dataset root does not exist; a dirty tree must be reported first."""
        monkeypatch.setattr(evaluate_module, "git_commit", lambda: "e" * 40)
        monkeypatch.setattr(evaluate_module, "git_dirty", lambda: True)

        with pytest.raises(GroundTruthError, match="dirty"):
            evaluate_module.evaluate(
                _config(), config_dir=tmp_path, artifacts_root=tmp_path, final=True
            )


class TestComparing:
    def test_the_same_numbers_under_new_labels_are_identical(self) -> None:
        a = _observations([10.0, -20.0, 30.0])
        b = _observations([10.0, -20.0, 30.0])

        comparison = compare_numbers(a, b)

        assert comparison.identical_numbers
        assert comparison.same_labels
        assert comparison.array_differences == ()

    def test_a_moved_number_is_named_with_its_size(self) -> None:
        a = _observations([10.0, -20.0, 30.0])
        b = _observations([10.0, -20.0, 30.5])

        comparison = compare_numbers(a, b)

        assert not comparison.identical_numbers
        [difference] = comparison.array_differences
        assert difference.key == "root_error_xyz_mm"
        assert difference.max_abs_difference == pytest.approx(0.5)

    def test_a_schema_1_file_against_a_schema_2_file_reports_the_added_arrays(
        self, tmp_path: Path
    ) -> None:
        current = _observations([10.0, -20.0])
        path = tmp_path / "v1.npz"
        current.save(path)
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        for key in ("schema_version", "scale_proxy_predicted_m", "scale_proxy_annotated_m"):
            del arrays[key]
        arrays["sequence_fx_released"] = arrays.pop("sequence_fx_annotation_derived")
        arrays["sequence_fx_estimated"] = arrays.pop("sequence_fx_model_estimated")
        np.savez(path, **arrays)
        old = Observations.load(path)

        comparison = compare_numbers(old, current)

        assert comparison.identical_numbers
        assert comparison.only_in_b == ("scale_proxy_predicted_m", "scale_proxy_annotated_m")

    def test_descriptor_differences_are_listed_by_field(self, tmp_path: Path) -> None:
        observations = _observations([10.0, -20.0])
        a = _evaluation_dir(tmp_path, "a", observations)
        b = _evaluation_dir(
            tmp_path, "b", observations, config=_config(camera="annotation_derived")
        )

        comparison = compare_evaluations(a, b)

        assert comparison.identical_numbers
        assert any(
            line.startswith("camera_source: 'model_estimated' -> 'annotation_derived'")
            for line in comparison.descriptor_differences
        )
        assert any(line.startswith("config.camera:") for line in comparison.descriptor_differences)
        text = describe(comparison)
        assert "identical" in text

    def test_the_paired_table_and_figure_are_written_deterministically(
        self, tmp_path: Path
    ) -> None:
        a = _evaluation_dir(
            tmp_path, "a", _observations([100.0, 200.0, -50.0, 0.0], ("TS1", "TS2"))
        )
        b = _evaluation_dir(
            tmp_path, "b", _observations([-30.0, -40.0, -60.0, -20.0], ("TS1", "TS2"))
        )

        first = write_paired(a, b, tmp_path / "out1", label_a="guessed", label_b="told")
        second = write_paired(a, b, tmp_path / "out2", label_a="guessed", label_b="told")

        assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]
        rows = (tmp_path / "out1" / "paired-per-sequence.csv").read_text("utf-8").splitlines()
        assert rows[0].startswith("sequence,matched_guessed,matched_told,root_bias_z_mm_guessed")
        assert rows[1].split(",")[0] == "TS1"
        # TS1 in a: 100 and -50 -> mean 25; in b: -30 and -60 -> -45.
        assert rows[1].split(",")[3:6] == ["25.0", "-45.0", "-70.0"]

    def test_different_skeletons_are_not_comparable(self, tmp_path: Path) -> None:
        a = _evaluation_dir(tmp_path, "a", _observations([1.0]))
        other = collect(
            ("pelvis", "neck"),
            ("TS9",),
            [
                (
                    0,
                    0,
                    0,
                    0,
                    PersonErrors(
                        np.zeros(2), np.zeros(2), np.ones(2, dtype=np.bool_), np.zeros(3), 3.0
                    ),
                )
            ],
            [(0, 0, 1, 1, 1)],
            [(1.0, 1.0)],
        )
        b = _evaluation_dir(tmp_path, "b", other)

        with pytest.raises(GroundTruthError, match="not comparable"):
            write_paired(a, b, tmp_path / "out", label_a="a", label_b="b")


def test_compare_descriptors_flattens_nested_fields() -> None:
    lines: list[str] = []
    _walk({"x": {"y": 1, "z": 2}, "w": 3}, {"x": {"y": 1, "z": 3}, "v": 4}, "", lines)

    assert lines == ["v: (absent) -> 4", "w: 3 -> (absent)", "x.z: 2 -> 3"]
