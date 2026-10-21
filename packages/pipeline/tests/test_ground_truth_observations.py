"""Tests for the raw record and the tables derived from it.

The tables are views. What has to be true is that they can be regenerated from
the file byte for byte, that nothing is lost on the way in, and that a frame
where nothing was detected still leaves a row saying so.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from unproject.ground_truth.config import MetricThresholds
from unproject.ground_truth.dataset import GroundTruthError
from unproject.ground_truth.metrics import PersonErrors
from unproject.ground_truth.observations import (
    OBSERVATIONS_SCHEMA_VERSION,
    Observations,
    collect,
    summarise,
    write_per_joint,
    write_per_sequence,
)

JOINTS = ("pelvis", "left_knee", "right_knee")


def _errors(absolute: float, relative: float = 0.0, depth: float = 3.0) -> PersonErrors:
    return PersonErrors(
        absolute_mm=np.full(len(JOINTS), absolute),
        root_relative_mm=np.full(len(JOINTS), relative),
        valid=np.ones(len(JOINTS), dtype=np.bool_),
        root_error_xyz_mm=np.array([0.0, 0.0, absolute]),
        depth_m=depth,
    )


def _observations(
    people: list[tuple[int, int, int, int, PersonErrors]] | None = None,
    frames: list[tuple[int, int, int, int, int]] | None = None,
    sequences: tuple[str, ...] = ("TS9",),
) -> Observations:
    if people is None:
        people = [(0, 0, 0, 7, _errors(100.0))]
    if frames is None:
        frames = [(0, 0, 1, 1, len(people))]
    return collect(JOINTS, sequences, people, frames, [(1100.0, 1300.0)] * len(sequences))


class TestTheBooksBalance:
    def test_a_person_row_lost_between_the_two_tallies_is_caught(self) -> None:
        """The frame rows say how many were matched; the person rows are those
        matches. If they disagree, a score was computed for somebody who is not
        in the coverage, or the reverse."""
        with pytest.raises(GroundTruthError, match="lost between the two"):
            collect(
                JOINTS, ("TS9",), [(0, 0, 0, 7, _errors(100.0))], [(0, 0, 1, 1, 0)], [(1.0, 1.0)]
            )

    def test_coverage_counts_every_frame_including_the_empty_ones(self) -> None:
        """A frame where nothing was detected is a recall failure, not an absent
        row. ADR-0004 §6.1: missing detections cannot disappear from reports."""
        observations = _observations(
            people=[(0, 0, 0, 7, _errors(100.0))],
            frames=[(0, 0, 1, 1, 1), (0, 1, 2, 0, 0)],
        )

        coverage = observations.coverage

        assert coverage.samples == 2
        assert coverage.ground_truth_people == 3
        assert coverage.matched == 1
        assert coverage.false_negatives == 2

    def test_unmatched_predictions_are_counted_not_discarded(self) -> None:
        observations = _observations(frames=[(0, 0, 1, 4, 1)])

        assert observations.coverage.false_positives == 3

    def test_recall_travels_with_the_accuracy(self) -> None:
        """A model that found one easy person out of ten must not read as
        accurate; §6.1 makes that impossible by pairing the two."""
        observations = _observations(
            people=[(0, 0, 0, 7, _errors(1.0))],
            frames=[(0, 0, 10, 1, 1)],
        )

        summary = summarise(observations, MetricThresholds())

        assert summary.mpjpe_mm.micro == pytest.approx(1.0)
        assert summary.recall == pytest.approx(0.1)


class TestRoundTrip:
    def test_what_went_in_comes_back_out(self, tmp_path: Path) -> None:
        original = _observations(
            people=[(0, 0, 0, 7, _errors(100.0)), (0, 1, 1, 8, _errors(250.0, 30.0, 5.0))],
            frames=[(0, 0, 1, 1, 1), (0, 1, 1, 2, 1)],
        )
        path = tmp_path / "observations.npz"

        original.save(path)
        restored = Observations.load(path)

        assert restored.joint_names == original.joint_names
        assert restored.sequence_names == original.sequence_names
        assert restored.absolute_mm == pytest.approx(original.absolute_mm)
        assert restored.depth_m == pytest.approx(original.depth_m)
        assert restored.coverage == original.coverage

    def test_a_file_that_is_not_observations_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "other.npz"
        np.savez(path, something=np.zeros(3))

        with pytest.raises(GroundTruthError, match="not an observations file"):
            Observations.load(path)

    def test_no_object_arrays_are_written(self, tmp_path: Path) -> None:
        """An NPZ with object dtype is a pickle with a different extension, and
        loading one executes whatever it holds. ADR-0004 §7.2 forbids it."""
        path = tmp_path / "observations.npz"
        _observations().save(path)

        with np.load(path, allow_pickle=False) as data:
            assert all(data[key].dtype != object for key in data.files)

    def test_the_file_says_which_schema_it_is(self, tmp_path: Path) -> None:
        path = tmp_path / "observations.npz"
        _observations().save(path)

        with np.load(path, allow_pickle=False) as data:
            assert int(data["schema_version"]) == OBSERVATIONS_SCHEMA_VERSION
            assert "sequence_fx_annotation_derived" in data.files
            assert "sequence_fx_released" not in data.files

    def test_the_scale_proxies_round_trip(self, tmp_path: Path) -> None:
        errors = _errors(100.0)
        with_proxy = PersonErrors(
            absolute_mm=errors.absolute_mm,
            root_relative_mm=errors.root_relative_mm,
            valid=errors.valid,
            root_error_xyz_mm=errors.root_error_xyz_mm,
            depth_m=errors.depth_m,
            scale_proxy_predicted_m=1.55,
            scale_proxy_annotated_m=1.70,
        )
        path = tmp_path / "observations.npz"
        _observations(people=[(0, 0, 0, 7, with_proxy)]).save(path)

        restored = Observations.load(path)

        assert restored.has_scale_proxy
        assert restored.scale_proxy_predicted_m == pytest.approx([1.55])
        assert restored.scale_proxy_annotated_m == pytest.approx([1.70])

    def test_a_schema_1_file_loads_under_the_new_names(self, tmp_path: Path) -> None:
        """The two thesis evaluations were written as v1: the same arrays under
        `sequence_fx_released` and `sequence_fx_estimated`, no schema marker and
        no scale proxy. Labels change, numbers do not."""
        current = _observations(
            people=[(0, 0, 0, 7, _errors(100.0)), (0, 1, 1, 8, _errors(250.0, 30.0, 5.0))],
            frames=[(0, 0, 1, 1, 1), (0, 1, 1, 2, 1)],
        )
        path = tmp_path / "observations.npz"
        current.save(path)
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        del arrays["schema_version"]
        del arrays["scale_proxy_predicted_m"]
        del arrays["scale_proxy_annotated_m"]
        arrays["sequence_fx_released"] = arrays.pop("sequence_fx_annotation_derived")
        arrays["sequence_fx_estimated"] = arrays.pop("sequence_fx_model_estimated")
        old = tmp_path / "v1.npz"
        np.savez(old, **arrays)

        restored = Observations.load(old)

        assert restored.absolute_mm == pytest.approx(current.absolute_mm)
        assert restored.sequence_fx_annotation_derived == pytest.approx([1100.0])
        assert restored.sequence_fx_model_estimated == pytest.approx([1300.0])
        assert not restored.has_scale_proxy
        assert np.isnan(restored.scale_proxy_predicted_m).all()
        assert restored.coverage == current.coverage

    def test_a_future_schema_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "observations.npz"
        _observations().save(path)
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["schema_version"] = np.array(OBSERVATIONS_SCHEMA_VERSION + 1, dtype=np.int32)
        np.savez(path, **arrays)

        with pytest.raises(GroundTruthError, match="schema"):
            Observations.load(path)


class TestNarrowing:
    def test_one_sequence_carries_only_its_own_rows(self) -> None:
        observations = _observations(
            people=[(0, 0, 0, 7, _errors(100.0)), (1, 0, 0, 2, _errors(400.0))],
            frames=[(0, 0, 1, 1, 1), (1, 0, 1, 1, 1)],
            sequences=("TS9", "TS10"),
        )

        first = observations.for_sequence("TS9")

        assert first.coverage.matched == 1
        assert first.absolute_mm[0] == pytest.approx(np.full(len(JOINTS), 100.0))

    def test_an_unknown_sequence_is_refused(self) -> None:
        with pytest.raises(GroundTruthError, match="TS99"):
            _observations().for_sequence("TS99")


class TestTables:
    def test_they_regenerate_byte_for_byte(self, tmp_path: Path) -> None:
        """§7.2: figures and CSVs are derived views. If they did not regenerate
        identically, the file would not be the source of truth - the run would."""
        observations = _observations(
            people=[(0, 0, 0, 7, _errors(123.456)), (0, 1, 0, 7, _errors(98.7654, 12.3))],
            frames=[(0, 0, 1, 1, 1), (0, 1, 1, 1, 1)],
        )
        thresholds = MetricThresholds()
        observations.save(tmp_path / "o.npz")

        first, second = tmp_path / "a.csv", tmp_path / "b.csv"
        write_per_sequence(observations, thresholds, first)
        write_per_sequence(Observations.load(tmp_path / "o.npz"), thresholds, second)

        assert first.read_bytes() == second.read_bytes()

    def test_every_row_has_exactly_as_many_cells_as_the_header(self, tmp_path: Path) -> None:
        """A row longer than its header is read by name with the tail dropped,
        which is how three columns went missing without anything failing."""
        observations = _observations(
            people=[(0, 0, 0, 7, _errors(100.0)), (1, 0, 0, 2, _errors(300.0))],
            frames=[(0, 0, 1, 1, 1), (1, 0, 1, 1, 1)],
            sequences=("TS9", "TS10"),
        )

        for writer in (write_per_sequence, write_per_joint):
            rows = _rows(observations, writer, tmp_path)
            assert {len(row) for row in rows} == {len(rows[0])}, writer.__name__

    def test_the_sequence_table_carries_the_focal_lengths(self, tmp_path: Path) -> None:
        """The mechanism column. Blank on the total row: a mean focal length
        over different cameras is not a camera."""
        observations = _observations(
            people=[(0, 0, 0, 7, _errors(100.0)), (1, 0, 0, 2, _errors(300.0))],
            frames=[(0, 0, 1, 1, 1), (1, 0, 1, 1, 1)],
            sequences=("TS9", "TS10"),
        )

        rows = _rows(observations, write_per_sequence, tmp_path)

        assert rows[0][-3:] == [
            "fx_annotation_derived",
            "fx_model_estimated",
            "focal_ratio_estimated_over_derived",
        ]
        assert rows[1][-3:] == ["1100.0", "1300.0", f"{1300.0 / 1100.0:.4f}"]
        assert rows[-1][-3:] == ["", "", ""]

    def test_the_sequence_table_carries_a_total_row(self, tmp_path: Path) -> None:
        observations = _observations(
            people=[(0, 0, 0, 7, _errors(100.0)), (1, 0, 0, 2, _errors(300.0))],
            frames=[(0, 0, 1, 1, 1), (1, 0, 1, 1, 1)],
            sequences=("TS9", "TS10"),
        )
        rows = _rows(observations, write_per_sequence, tmp_path)

        assert [row[0] for row in rows[1:]] == ["TS9", "TS10", "ALL"]
        assert rows[-1][4] == "2"

    def test_a_joint_nobody_annotated_leaves_a_blank_not_a_zero(self, tmp_path: Path) -> None:
        """Zero is a perfect score for that joint. Blank is the truth."""
        errors = _errors(100.0)
        errors.valid[1] = False
        errors.absolute_mm[1] = np.nan
        errors.root_relative_mm[1] = np.nan

        rows = _rows(_observations(people=[(0, 0, 0, 7, errors)]), write_per_joint, tmp_path)

        assert rows[2][0] == "left_knee"
        assert rows[2][1] == "0"
        assert rows[2][2] == ""


type TableWriter = Callable[[Observations, MetricThresholds, Path], None]


def _rows(observations: Observations, writer: TableWriter, directory: Path) -> list[list[str]]:
    path = directory / "table.csv"
    writer(observations, MetricThresholds(), path)
    with path.open(encoding="utf-8") as handle:
        return list(csv.reader(handle))
