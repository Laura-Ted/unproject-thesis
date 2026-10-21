"""Tests for the RQ3 metrics, on scenes whose answers are known in advance.

Every case here is synthetic: a body placed exactly where the test says, moving
exactly as fast as the test says. That is the only way to tell a metric that is
right from one that merely produces a number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fixture_types import WriteConfig
from unproject.bodies.artifacts import BodiesArtifact
from unproject.rq3.analyse import analyse, positions_csv, write_summary
from unproject.rq3.annotations import (
    Annotation,
    AnnotationError,
    Marker,
    MotionInterval,
    Scene,
    StationaryInterval,
    load_annotation,
)
from unproject.rq3.figures import (
    JITTER_FIGURE,
    POSITION_FIGURE,
    VELOCITY_FIGURE,
    write_figures,
)
from unproject.rq3.metrics import (
    MAX_STEP_SECONDS,
    axis_motion,
    position_errors,
    reprojection_consistency,
    shape_variation,
    smoothing_effect,
    track_fragmentation,
    track_summaries,
)
from unproject.rq3.reader import Detection, Run
from unproject.rq3.smoothing import DEFAULT_SMOOTHING, TrackSmoother

_FIXTURES = Path(__file__).parent / "fixtures"


def _detection(
    time: float,
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 8.0,
    track: int = 0,
    shape: list[float] | None = None,
    box: tuple[float, float, float, float] = (0.0, 0.0, 1280.0, 720.0),
    frame_index: int = 0,
) -> Detection:
    return Detection(
        time_seconds=time,
        frame_index=frame_index,
        track_id=track,
        confidence=0.9,
        root_translation=np.array([x, y, z], dtype=np.float32),
        shape=np.array(shape if shape is not None else [0.0] * 4, dtype=np.float32),
        box_xyxy=box,
    )


def _annotation(**overrides: object) -> Annotation:
    defaults: dict[str, object] = {
        "name": "corridor",
        "scene": Scene(participants=1),
        "markers": [
            Marker(id="A", x_metres=0.0, z_metres=5.0, uncertainty_metres=0.05),
            Marker(id="B", x_metres=2.0, z_metres=9.0, uncertainty_metres=0.05),
        ],
    }
    return Annotation.model_validate(defaults | overrides)


class TestTheAnnotationFormat:
    def test_an_interval_naming_an_unknown_marker_is_refused(self) -> None:
        """A typo would drop the interval and report the rest as complete."""
        with pytest.raises(ValueError, match=r"marker\(s\) \['C'\]"):
            _annotation(stationary=[StationaryInterval(marker="C", start_seconds=0, end_seconds=1)])

    def test_an_interval_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not after"):
            StationaryInterval(marker="A", start_seconds=5.0, end_seconds=2.0)

    def test_a_walk_from_a_marker_to_itself_is_refused(self) -> None:
        with pytest.raises(ValueError, match="to itself"):
            MotionInterval(from_marker="A", to_marker="A", start_seconds=0.0, end_seconds=1.0)

    def test_two_markers_cannot_share_an_id(self) -> None:
        with pytest.raises(ValueError, match="share an id"):
            _annotation(
                markers=[
                    Marker(id="A", x_metres=0.0, z_metres=5.0, uncertainty_metres=0.05),
                    Marker(id="A", x_metres=1.0, z_metres=6.0, uncertainty_metres=0.05),
                ]
            )

    def test_the_convention_has_to_be_the_one_the_artifact_uses(self) -> None:
        # Reading metres in one frame against metres in another is the kind of
        # mistake that produces a plausible number.
        with pytest.raises(ValueError, match="camera_relative_opencv_metres"):
            _annotation(convention="world_metres")

    def test_every_marker_carries_its_own_uncertainty(self) -> None:
        with pytest.raises(ValueError):
            Marker(id="A", x_metres=0.0, z_metres=5.0, uncertainty_metres=0.0)

    def test_it_reads_a_whole_file(self, write_config: WriteConfig) -> None:
        path = write_config(
            'name = "corridor"\n'
            '[scene]\nparticipants = 1\nannotated_by = "lt"\n'
            '[[markers]]\nid = "A"\nx_metres = 0.0\nz_metres = 5.0\n'
            "uncertainty_metres = 0.05\n"
            '[[stationary]]\nmarker = "A"\nstart_seconds = 1.0\nend_seconds = 4.0\n',
            name="corridor.toml",
        )

        annotation = load_annotation(path)

        assert annotation.marker("A").z_metres == 5.0
        assert annotation.stationary[0].boundary_uncertainty_seconds == 0.1

    def test_a_broken_file_is_reported_with_its_path(self, write_config: WriteConfig) -> None:
        path = write_config("name = 'x'\nmarkers = []\nnope = 1\n", name="bad.toml")
        with pytest.raises(AnnotationError, match=r"bad\.toml"):
            load_annotation(path)


class TestPositionError:
    def test_a_body_placed_exactly_right_has_no_error(self) -> None:
        run = Run(
            artifact=_artifact(),
            detections=[_detection(time, x=0.0, z=5.0) for time in (1.0, 1.5, 2.0)],
        )
        annotation = _annotation(
            stationary=[StationaryInterval(marker="A", start_seconds=0.5, end_seconds=2.5)]
        )

        [error] = position_errors(run, annotation)

        assert error.samples == 3
        assert error.lateral_abs_median_metres == 0.0
        assert error.depth_abs_median_metres == 0.0
        assert error.depth_error_exceeds_uncertainty is False

    def test_depth_and_lateral_error_are_reported_apart(self) -> None:
        """The whole question is which axis is wrong, so one number will not do."""
        run = Run(
            artifact=_artifact(),
            # 10 cm right of the marker, 1.5 m too far away.
            detections=[_detection(time, x=0.1, z=6.5) for time in (1.0, 1.5, 2.0)],
        )
        annotation = _annotation(
            stationary=[StationaryInterval(marker="A", start_seconds=0.5, end_seconds=2.5)]
        )

        [error] = position_errors(run, annotation)

        assert error.lateral_abs_median_metres == pytest.approx(0.1, abs=1e-6)
        assert error.depth_abs_median_metres == pytest.approx(1.5, abs=1e-6)
        assert error.depth_bias_metres == pytest.approx(1.5, abs=1e-6)
        assert error.depth_error_exceeds_uncertainty is True

    def test_an_error_inside_the_ruler_is_marked_as_such(self) -> None:
        run = Run(
            artifact=_artifact(),
            detections=[_detection(time, z=5.02) for time in (1.0, 1.5)],
        )
        annotation = _annotation(
            stationary=[StationaryInterval(marker="A", start_seconds=0.5, end_seconds=2.5)]
        )

        [error] = position_errors(run, annotation)

        # 2 cm against a marker known to 5 cm is not a measurement of anything.
        assert error.depth_error_exceeds_uncertainty is False

    def test_an_interval_with_no_detections_reports_none_rather_than_zero(self) -> None:
        """Zero error and no data must not look the same."""
        run = Run(artifact=_artifact(), detections=[])
        annotation = _annotation(
            stationary=[StationaryInterval(marker="A", start_seconds=0.5, end_seconds=2.5)]
        )

        [error] = position_errors(run, annotation)

        assert error.samples == 0
        assert error.depth_abs_median_metres is None
        assert error.lateral_abs_median_metres is None


class TestAxisMotion:
    def test_a_known_velocity_comes_back(self) -> None:
        # 0.5 m of depth every 0.25 s is 2 m/s, and nothing sideways.
        detections = [_detection(index * 0.25, z=8.0 + index * 0.5) for index in range(6)]

        motion = axis_motion(detections, label="walk", track_id=0)

        assert motion.steps == 5
        assert motion.depth_median_metres_per_second == pytest.approx(2.0, abs=1e-6)
        assert motion.lateral_median_metres_per_second == 0.0

    def test_a_gap_is_skipped_rather_than_averaged_across(self) -> None:
        """Two seconds of nobody is not a person moving slowly."""
        detections = [
            _detection(0.0, z=8.0),
            _detection(0.25, z=8.5),
            _detection(0.25 + MAX_STEP_SECONDS + 0.1, z=20.0),
        ]

        motion = axis_motion(detections, label="walk", track_id=0)

        assert motion.steps == 1
        assert motion.gaps_skipped == 1
        assert motion.depth_median_metres_per_second == pytest.approx(2.0, abs=1e-6)

    def test_steps_between_different_tracks_are_not_steps(self) -> None:
        detections = [_detection(0.0, z=8.0, track=0), _detection(0.25, z=20.0, track=1)]

        motion = axis_motion(detections, label="two people", track_id=None)

        assert motion.steps == 0

    def test_two_people_in_every_frame_still_produce_steps(self) -> None:
        """Found on the crowd clip: 2832 detections, and zero steps.

        Walking the list in time order and skipping pairs whose track changed
        looks equivalent to grouping by track and is not - with more than one
        person per frame, consecutive entries are different people and every
        single pair is skipped.
        """
        detections = [
            _detection(index * 0.25, z=8.0 + index * 0.5, track=track)
            for index in range(5)
            for track in (0, 1)
        ]

        motion = axis_motion(detections, label="two people", track_id=None)

        # Four steps each, not zero.
        assert motion.steps == 8
        assert motion.depth_median_metres_per_second == pytest.approx(2.0, abs=1e-6)

    def test_jitter_is_also_reported_against_range(self) -> None:
        # 0.4 m of jitter at 8.4 m is about 5% of range, which is the form the error takes.
        detections = [_detection(0.0, z=8.0), _detection(0.25, z=8.4)]

        motion = axis_motion(detections, label="still", track_id=0)

        assert motion.depth_jitter_fraction_of_range == pytest.approx(0.4 / 8.4, abs=1e-4)


class TestShapeAndTracks:
    def test_a_body_whose_shape_never_moves_has_no_variation(self) -> None:
        run = Run(
            artifact=_artifact(),
            detections=[_detection(t, shape=[1.0, 2.0]) for t in (0.0, 0.5, 1.0)],
        )

        [variation] = shape_variation(run)

        assert variation.max_parameter_std == 0.0

    def test_a_shape_that_wanders_is_reported(self) -> None:
        run = Run(
            artifact=_artifact(),
            detections=[
                _detection(0.0, shape=[1.0, 2.0]),
                _detection(0.5, shape=[1.0, 4.0]),
                _detection(1.0, shape=[1.0, 6.0]),
            ],
        )

        [variation] = shape_variation(run)

        # The steady parameter must not dilute the wandering one.
        assert variation.max_parameter_std == pytest.approx(2.0, abs=1e-6)

    def test_one_person_becoming_three_tracks_is_visible(self) -> None:
        run = Run(
            artifact=_artifact(),
            detections=[
                _detection(0.0, track=0),
                _detection(0.5, track=1),
                _detection(1.0, track=2),
            ],
        )

        fragmentation = track_fragmentation(run, participants=1)

        assert fragmentation.tracks == 3
        assert fragmentation.tracks_per_participant == 3.0

    def test_a_hole_inside_a_track_is_reported(self) -> None:
        run = Run(
            artifact=_artifact(),
            detections=[_detection(0.0), _detection(0.25), _detection(4.0)],
        )

        fragmentation = track_fragmentation(run, participants=1)

        assert fragmentation.longest_gap_seconds == pytest.approx(3.75, abs=1e-6)


class TestReprojection:
    def test_a_root_that_lands_on_its_own_box_is_consistent(self) -> None:
        # fx * x/z + cx with x=0 puts it at cx, which is inside a centred box.
        run = Run(
            artifact=_artifact(),
            detections=[_detection(0.0, x=0.0, y=0.0, z=8.0, box=(600.0, 320.0, 680.0, 400.0))],
        )

        consistency = reprojection_consistency(run)

        assert consistency.clear_inside == 1
        assert consistency.clear_inside_fraction == 1.0

    def test_a_root_that_lands_elsewhere_is_measured_in_pixels(self) -> None:
        run = Run(
            artifact=_artifact(),
            detections=[_detection(0.0, x=0.0, z=8.0, box=(200.0, 10.0, 210.0, 20.0))],
        )

        consistency = reprojection_consistency(run)

        assert consistency.clear_inside == 0
        assert consistency.median_outside_px is not None
        assert consistency.median_outside_px > 0.0

    def test_a_body_cut_off_by_the_frame_is_counted_separately(self) -> None:
        """A person walking out of the bottom of the shot has their pelvis below it.

        Counting those as failures put the crowd clip at 58% when every body
        fully in frame was at 100%.
        """
        run = Run(
            artifact=_artifact(),
            detections=[
                # Box clamped to the bottom of a 720-high frame: legs cut off.
                _detection(0.0, x=0.0, y=8.0, z=8.0, box=(600.0, 400.0, 700.0, 720.0)),
                # Box clear of every edge, and consistent.
                _detection(0.5, x=0.0, y=0.0, z=8.0, box=(600.0, 320.0, 680.0, 400.0)),
            ],
        )

        consistency = reprojection_consistency(run)

        assert consistency.clear_samples == 1
        assert consistency.clear_inside_fraction == 1.0
        assert consistency.clipped_samples == 1
        assert consistency.clipped_inside_fraction == 0.0

    def test_a_body_behind_the_camera_is_not_projected(self) -> None:
        """Negative depth is a result, not a division to attempt."""
        run = Run(artifact=_artifact(), detections=[_detection(0.0, z=-2.0)])

        consistency = reprojection_consistency(run)

        assert consistency.samples == 1
        assert consistency.clear_inside + consistency.clipped_inside == 0


class TestSmoothing:
    def test_it_damps_a_body_that_is_standing_still(self) -> None:
        # A metre of depth noise on someone who has not moved.
        detections = [
            _detection(index * 0.4, z=8.0 + (0.5 if index % 2 else -0.5)) for index in range(20)
        ]

        effect = smoothing_effect(detections, label="still", track_id=0)

        raw = effect.raw.depth_median_metres_per_second
        smoothed = effect.smoothed.depth_median_metres_per_second
        assert raw is not None and smoothed is not None
        assert smoothed < raw

    def test_the_raw_numbers_survive_beside_the_smoothed_ones(self) -> None:
        """The raw jitter is the finding; a report that only smoothed is useless."""
        detections = [
            _detection(index * 0.4, z=8.0 + (0.5 if index % 2 else -0.5)) for index in range(20)
        ]

        effect = smoothing_effect(detections, label="still", track_id=0)

        assert "raw" in effect.raw.label
        assert "smoothed" in effect.smoothed.label
        assert effect.raw.steps == effect.smoothed.steps

    def test_the_delay_the_filter_costs_is_reported(self) -> None:
        # A body walking steadily away: the filter has to lag it, and the lag is
        # the price of the stability above.
        detections = [_detection(index * 0.4, z=8.0 + index * 0.6) for index in range(20)]

        effect = smoothing_effect(detections, label="walk", track_id=0)

        assert effect.response_delay_seconds is not None
        assert effect.response_delay_seconds > 0.0


class TestTheSharedFilterVector:
    """The Python filter and the viewer's must not drift apart.

    Both suites read `fixtures/one_euro.json`. If either implementation changes,
    both fail, which is the point: the alternative is one number quietly moving.
    """

    def test_the_port_reproduces_the_recorded_vector(self) -> None:
        fixture = json.loads((_FIXTURES / "one_euro.json").read_text("utf-8"))
        smoother = TrackSmoother(DEFAULT_SMOOTHING)

        produced = []
        for sample in fixture["samples"]:
            [(_, root)] = smoother.frame(
                sample["time"], [(0, (sample["x"], sample["y"], sample["z"]))]
            )
            produced.append(list(root))

        for got, want in zip(produced, fixture["expected"], strict=True):
            assert got == pytest.approx(want, abs=1e-6)

    def test_the_options_in_the_fixture_are_the_viewer_defaults(self) -> None:
        fixture = json.loads((_FIXTURES / "one_euro.json").read_text("utf-8"))
        options = fixture["options"]

        assert options["depth"]["cutoffHz"] == DEFAULT_SMOOTHING.depth.cutoff_hz
        assert options["lateral"]["cutoffHz"] == DEFAULT_SMOOTHING.lateral.cutoff_hz
        assert options["referenceRangeMetres"] == DEFAULT_SMOOTHING.reference_range_metres


class TestTheSummary:
    def test_it_is_deterministic(self) -> None:
        run, annotation = _scene()

        first = analyse(run, annotation).model_dump_json(indent=2)
        second = analyse(run, annotation).model_dump_json(indent=2)

        assert first == second

    def test_it_names_its_units_and_frame(self) -> None:
        run, annotation = _scene()

        summary = analyse(run, annotation)

        assert summary.convention == "camera_relative_opencv_metres"
        assert "metres" in summary.units

    def test_the_csv_leaves_a_missing_measurement_empty(self) -> None:
        run = Run(artifact=_artifact(), detections=[])
        annotation = _annotation(
            stationary=[StationaryInterval(marker="A", start_seconds=0.0, end_seconds=1.0)]
        )

        rows = positions_csv(analyse(run, annotation)).splitlines()

        header, row = rows[0].split(","), rows[1].split(",")
        assert row[header.index("depth_abs_median_metres")] == ""
        assert row[header.index("marker_uncertainty_metres")] == "0.05"

    def test_it_writes_both_forms(self, tmp_path: Path) -> None:
        run, annotation = _scene()

        as_json, as_csv = write_summary(analyse(run, annotation), tmp_path)

        assert as_json.is_file()
        assert as_csv.read_text("utf-8").startswith("marker,track_id,")


def _artifact() -> BodiesArtifact:
    """A minimal bodies artifact: only what the metrics read off it."""
    return BodiesArtifact.model_validate(
        {
            "convention": "camera_relative_opencv_metres",
            "ingest_hash": "0" * 12,
            "body_model": {
                "name": "anny",
                "version": "0.6.0",
                "bone_count": 1,
                "shape_count": 11,
                "bone_parents": [-1],
            },
            "estimator": {
                "name": "multihmr2",
                "checkpoint_sha256": "f" * 64,
                "backbone_repo": "facebookresearch/dinov3",
                "backbone_ref": "0" * 40,
                "config_hash": "deadbeef",
            },
            "parameters": {"path": "bodies.bin", "person_count": 0, "arrays": []},
            "frames": [
                {
                    "frame_index": 0,
                    "source_index": 0,
                    "timestamp_seconds": 0.0,
                    "camera": {
                        "width": 1280,
                        "height": 720,
                        "fx": 1000.0,
                        "fy": 1000.0,
                        "cx": 640.0,
                        "cy": 360.0,
                        "source": "estimated",
                    },
                    "people": [],
                }
            ],
            "summary": {
                "frames_total": 1,
                "frames_with_people": 0,
                "people_total": 0,
                "max_people_in_frame": 0,
                "confidence_threshold": 0.4,
                "track_count": 0,
            },
        }
    )


def _scene() -> tuple[Run, Annotation]:
    """One participant standing at A, then walking to B."""
    still = [_detection(1.0 + index * 0.25, x=0.0, z=5.0) for index in range(8)]
    walk = [
        _detection(4.0 + index * 0.25, x=index * 0.2, z=5.0 + index * 0.4) for index in range(10)
    ]
    annotation = _annotation(
        stationary=[StationaryInterval(marker="A", start_seconds=1.0, end_seconds=3.0)],
        motion=[MotionInterval(from_marker="A", to_marker="B", start_seconds=4.0, end_seconds=6.5)],
    )
    return Run(artifact=_artifact(), detections=still + walk), annotation


class TestFigures:
    """A figure is regenerated from the summary, so it has to be reproducible."""

    def test_two_runs_over_one_summary_produce_identical_bytes(self, tmp_path: Path) -> None:
        # SVG carries a timestamp and a random id salt unless both are pinned,
        # and a picture whose bytes move on every run cannot be trusted in a
        # thesis repository.
        run, annotation = _scene()
        summary = analyse(run, annotation)

        first = write_figures(summary, tmp_path / "one")
        second = write_figures(summary, tmp_path / "two")

        assert [path.name for path in first] == [path.name for path in second]
        for a, b in zip(first, second, strict=True):
            assert a.read_bytes() == b.read_bytes()

    def test_it_draws_the_three_the_argument_needs(self, tmp_path: Path) -> None:
        run, annotation = _scene()

        drawn = {path.name for path in write_figures(analyse(run, annotation), tmp_path)}

        assert drawn == {POSITION_FIGURE, VELOCITY_FIGURE, JITTER_FIGURE}

    def test_every_figure_names_its_unit_and_frame(self, tmp_path: Path) -> None:
        """A plot of metres that does not say which metres is not a result."""
        run, annotation = _scene()

        for path in write_figures(analyse(run, annotation), tmp_path):
            text = path.read_text("utf-8")
            assert "camera_relative_opencv_metres" in text
            assert "metres" in text or "%" in text

    def test_a_summary_with_nothing_in_it_draws_nothing(self, tmp_path: Path) -> None:
        """An empty axis reads as a measurement of zero."""
        empty = analyse(Run(artifact=_artifact(), detections=[]), _annotation())

        assert write_figures(empty, tmp_path) == []
        assert not list(tmp_path.glob("*.svg"))

    def test_an_interval_with_no_detections_is_left_out_of_the_position_figure(
        self, tmp_path: Path
    ) -> None:
        run, annotation = _scene()
        annotation = _annotation(
            stationary=[
                *annotation.stationary,
                # Nothing was found here; it must not appear as a zero bar.
                StationaryInterval(marker="B", start_seconds=90.0, end_seconds=95.0),
            ],
            motion=annotation.motion,
        )

        write_figures(analyse(run, annotation), tmp_path)

        text = (tmp_path / POSITION_FIGURE).read_text("utf-8")
        assert "n=8" in text
        assert "n=0" not in text


class TestWholeRunMotion:
    """The half of RQ3 that needs no ground truth has to work without it.

    Real footage found this: an uncontrolled scene has no annotated intervals,
    and every motion metric was reachable only through one, so a run with no
    marks produced no jitter figure at all.
    """

    def test_motion_is_measured_without_any_annotated_interval(self) -> None:
        detections = [_detection(index * 0.25, z=8.0 + index * 0.5) for index in range(6)]
        run = Run(artifact=_artifact(), detections=detections)

        summary = analyse(run, _annotation())

        assert summary.stationary_motion == []
        assert summary.overall_motion.steps == 5
        assert summary.overall_motion.depth_median_metres_per_second == pytest.approx(2.0)

    def test_it_does_not_step_between_two_different_people(self) -> None:
        run = Run(
            artifact=_artifact(),
            detections=[_detection(0.0, z=8.0, track=0), _detection(0.25, z=30.0, track=1)],
        )

        summary = analyse(run, _annotation())

        assert summary.overall_motion.steps == 0

    def test_the_jitter_figure_draws_it_when_there_is_nothing_else(self, tmp_path: Path) -> None:
        detections = [
            _detection(index * 0.25, z=8.0 + (0.4 if index % 2 else 0.0)) for index in range(8)
        ]
        summary = analyse(Run(artifact=_artifact(), detections=detections), _annotation())

        drawn = {path.name for path in write_figures(summary, tmp_path)}

        assert JITTER_FIGURE in drawn
        assert "whole run" in (tmp_path / JITTER_FIGURE).read_text("utf-8")


class TestResponseDelayIsNotAlwaysDefined:
    def test_it_is_not_reported_across_interleaved_tracks(self) -> None:
        """On the crowd clip this returned a confident 0.0 s over 62 people.

        Two interleaved tracks are not one signal, so the shift that best
        aligns them says nothing about lag.
        """
        detections = [
            _detection(index * 0.25, z=8.0 + index * 0.5, track=track)
            for index in range(8)
            for track in (0, 1)
        ]

        effect = smoothing_effect(detections, label="two people", track_id=None)

        assert effect.response_delay_seconds is None
        # The motion metrics either side of it are still measured.
        assert effect.raw.steps > 0
        assert effect.smoothed.steps > 0


class TestPerTrackSummaries:
    """One entry per track, for inspecting them one at a time. ADR-0003 §3.2."""

    def test_every_track_gets_an_entry_in_id_order(self) -> None:
        run = Run(
            artifact=_artifact(),
            detections=[
                _detection(0.0, track=2),
                _detection(0.1, track=0),
                _detection(0.2, track=2),
                _detection(0.3, track=1),
            ],
        )

        summaries = track_summaries(run)

        assert [s.track_id for s in summaries] == [0, 1, 2]
        assert [s.detections for s in summaries] == [1, 1, 2]

    def test_it_records_when_a_track_starts_and_ends(self) -> None:
        run = Run(
            artifact=_artifact(),
            detections=[_detection(1.5), _detection(2.0), _detection(4.25)],
        )

        [summary] = track_summaries(run)

        assert summary.first_seconds == 1.5
        assert summary.last_seconds == 4.25

    def test_the_longest_hole_inside_a_track_is_reported(self) -> None:
        """This is what the operator sees as a body vanishing and coming back."""
        run = Run(
            artifact=_artifact(),
            detections=[_detection(0.0), _detection(0.25), _detection(3.0)],
        )

        [summary] = track_summaries(run)

        assert summary.longest_gap_seconds == pytest.approx(2.75, abs=1e-6)

    def test_a_single_detection_track_has_no_gap(self) -> None:
        run = Run(artifact=_artifact(), detections=[_detection(1.0)])

        [summary] = track_summaries(run)

        assert summary.longest_gap_seconds is None
        assert summary.detections == 1

    def test_it_records_how_far_away_the_track_was(self) -> None:
        # Depth error scales with range, so a per-track figure without the
        # range it was measured at cannot be compared to another track's.
        run = Run(
            artifact=_artifact(),
            detections=[_detection(0.0, z=10.0), _detection(0.1, z=20.0)],
        )

        [summary] = track_summaries(run)

        assert summary.median_range_metres == pytest.approx(15.0, abs=1e-6)

    def test_the_motion_it_reports_is_that_track_only(self) -> None:
        """Two tracks in the same frames must not be measured as one moving body."""
        run = Run(
            artifact=_artifact(),
            detections=[
                _detection(0.0, z=8.0, track=0),
                _detection(0.0, z=30.0, track=1),
                _detection(0.25, z=8.5, track=0),
                _detection(0.25, z=30.5, track=1),
            ],
        )

        first, second = track_summaries(run)

        assert first.motion.steps == 1
        assert second.motion.steps == 1
        assert first.motion.depth_median_metres_per_second == pytest.approx(2.0, abs=1e-6)

    def test_the_numbers_match_the_ones_the_rest_of_rq3_reports(self) -> None:
        """A figure in the viewer and a figure in the table cannot disagree."""
        run, annotation = _scene()
        summary = analyse(run, annotation)

        by_id = {s.track_id: s for s in summary.per_track}
        for variation in summary.shapes:
            assert by_id[variation.track_id].shape == variation

    def test_it_is_in_the_summary(self) -> None:
        run, annotation = _scene()

        summary = analyse(run, annotation)

        assert [s.track_id for s in summary.per_track] == run.track_ids
