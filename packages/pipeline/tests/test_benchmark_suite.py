"""Tests for protocol v2: the suite, the job, and the arithmetic on top.

The model never runs here. What is under test is which frames a job times and
in what order, that every job of a suite times the same ones, that the run
order is the seed's and not the file's, that a job's phases stay apart, and
that a summary computes what the protocol says from what a job wrote.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fixture_types import MakeVideo, WriteConfig
from unproject.benchmark.artifacts import Environment
from unproject.benchmark.config import OutputMode
from unproject.benchmark.dryrun import DRY_RUN_CHECKPOINT, DryRunEstimator, DryRunSession
from unproject.benchmark.job import measure_job, parse_conditions
from unproject.benchmark.results import (
    JOB_RESULT_SCHEMA_VERSION,
    Conditions,
    FrameSample,
    InputGeometry,
    JobIdentity,
    JobInputs,
    JobResult,
    Memory,
    Phases,
)
from unproject.benchmark.suite import (
    PROTOCOL_MEASURED_FRAMES,
    PROTOCOL_REPETITIONS,
    Job,
    Pair,
    SuiteConfig,
    load_suite_config,
    plan_jobs,
    results_dir,
    run_suite,
    select_frames,
)
from unproject.benchmark.summary import (
    PAIRS_CSV_FILENAME,
    SUMMARY_CSV_FILENAME,
    SUMMARY_FILENAME,
    pairs_to_csv,
    read_results,
    summarise,
    to_csv,
    write_summary,
)
from unproject.bodies.multihmr import FrameEstimate, FrameInput
from unproject.config import BodiesConfig, ConfigError, load_config

_STARTED = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)

_EXPERIMENT = """
name = "fixture"
[ingest]
source_path = "clip.avi"
"""

_SUITE = """
name = "suite"
warmup_frames = 2
measured_frames = 4
repetitions = 2
order_seed = 7

[[cells]]
name = "a-skeleton"
experiment = "fixture.toml"
output = "skeleton"

[[cells]]
name = "a-mesh"
experiment = "fixture.toml"
output = "mesh"

[[pairs]]
a = "a-skeleton"
b = "a-mesh"
question = "output mode"
"""


def _suite(**overrides: object) -> SuiteConfig:
    defaults: dict[str, object] = {
        "name": "suite",
        "warmup_frames": 2,
        "measured_frames": 4,
        "repetitions": 2,
        "order_seed": 7,
        "cells": [
            {"name": "a-skeleton", "experiment": "fixture.toml", "output": "skeleton"},
            {"name": "a-mesh", "experiment": "fixture.toml", "output": "mesh"},
        ],
        "pairs": [{"a": "a-skeleton", "b": "a-mesh", "question": "output mode"}],
    }
    return SuiteConfig.model_validate(defaults | overrides)


class TestTheSuiteConfig:
    def test_it_reads_a_whole_suite(self, write_config: WriteConfig) -> None:
        suite = load_suite_config(write_config(_SUITE, name="suite.toml"))

        assert suite.name == "suite"
        assert [cell.name for cell in suite.cells] == ["a-skeleton", "a-mesh"]
        assert suite.pairs[0].question == "output mode"
        assert suite.protocol == "rq1rq2-v2"

    def test_a_typo_is_refused_rather_than_defaulted(self, write_config: WriteConfig) -> None:
        with pytest.raises(ConfigError, match="measured_frame"):
            load_suite_config(
                write_config(_SUITE.replace("measured_frames", "measured_frame"), name="s.toml")
            )

    def test_a_cell_named_twice_is_refused(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            _suite(
                cells=[
                    {"name": "x", "experiment": "e.toml"},
                    {"name": "x", "experiment": "f.toml"},
                ],
                pairs=[],
            )

    def test_a_pair_must_name_cells_of_the_suite(self) -> None:
        with pytest.raises(ValueError, match="not in the suite"):
            _suite(pairs=[{"a": "a-skeleton", "b": "nope", "question": "q"}])

    def test_a_pair_cannot_compare_a_cell_with_itself(self) -> None:
        with pytest.raises(ValueError, match="itself"):
            _suite(pairs=[{"a": "a-mesh", "b": "a-mesh", "question": "q"}])

    def test_an_unsupported_schema_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            _suite(schema_version=9)

    def test_the_shipped_suite_meets_the_protocol_minimums(self) -> None:
        suite = load_suite_config(Path("experiments/benchmark/v2/matrix-suite.toml"))

        assert suite.measured_frames >= PROTOCOL_MEASURED_FRAMES
        assert suite.repetitions >= PROTOCOL_REPETITIONS
        assert len(suite.cells) == 8
        # Every cell measures footage with recorded provenance and pinned
        # bytes: the whole point of replacing the crowd clip. The four
        # geometries are the ingest's, on the one source, so every cell
        # pins the same source digest.
        digests = set()
        for cell in suite.cells:
            experiment = load_config(Path("experiments/benchmark/v2") / cell.experiment)
            assert experiment.provenance is not None, cell.name
            assert experiment.provenance.expected_sha256 is not None, cell.name
            digests.add(experiment.provenance.expected_sha256)
        assert len(digests) == 1


class TestPlanningJobs:
    def test_every_cell_and_repetition_appears_exactly_once(self) -> None:
        jobs = plan_jobs(_suite())

        assert sorted((job.cell, job.repetition) for job in jobs) == [
            ("a-mesh", 0),
            ("a-mesh", 1),
            ("a-skeleton", 0),
            ("a-skeleton", 1),
        ]
        assert [job.index for job in jobs] == [0, 1, 2, 3]

    def test_the_order_is_the_seeds_and_repeats(self) -> None:
        """The seed is recorded in every result so the order can be reproduced."""
        assert plan_jobs(_suite()) == plan_jobs(_suite())
        assert plan_jobs(_suite(order_seed=8)) != plan_jobs(_suite(order_seed=7))

    def test_the_order_is_not_the_files(self) -> None:
        """Cells in file order, repetitions in sequence, would put every
        repetition of a cell together; a drift of the machine would then land
        on one cell. Some seed keeps the file order, most do not."""
        differing = [
            seed
            for seed in range(20)
            if [(job.cell, job.repetition) for job in plan_jobs(_suite(order_seed=seed))]
            != [("a-skeleton", 0), ("a-mesh", 0), ("a-skeleton", 1), ("a-mesh", 1)]
        ]
        assert len(differing) > 15


class TestSelectingFrames:
    def test_warm_up_is_the_start_and_the_measured_frames_are_spread(self) -> None:
        warmup, measured = select_frames(375, warmup=5, measured=60)

        assert warmup == [0, 1, 2, 3, 4]
        assert measured[0] == 5
        assert measured[-1] == 374
        assert len(measured) == 60
        # Monotone, for the tracker, and never repeating a frame.
        assert measured == sorted(set(measured))

    def test_exactly_enough_frames_gives_consecutive_ones(self) -> None:
        warmup, measured = select_frames(8, warmup=2, measured=6)

        assert warmup == [0, 1]
        assert measured == [2, 3, 4, 5, 6, 7]

    def test_too_short_a_clip_is_refused_rather_than_trimmed(self) -> None:
        with pytest.raises(ConfigError, match="30 frame"):
            select_frames(30, warmup=5, measured=60)

    def test_one_measured_frame_is_the_first_after_warm_up(self) -> None:
        assert select_frames(10, warmup=3, measured=1) == ([0, 1, 2], [3])


class TestMeasuringAJob:
    def _prepare(self, make_video: MakeVideo, write_config: WriteConfig, frames: int = 12) -> Path:
        make_video(name="clip.avi", frame_count=frames)
        write_config(_EXPERIMENT, name="fixture.toml")
        return write_config(_SUITE, name="suite.toml")

    def test_it_times_the_planned_frames_in_order_with_phases_apart(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        suite_path = self._prepare(make_video, write_config)
        suite = load_suite_config(suite_path)

        result, path = measure_job(
            suite,
            Job(index=1, cell="a-mesh", repetition=0),
            suite_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
            job_count=4,
            estimator=DryRunEstimator(BodiesConfig()),
            dry_run=True,
        )

        assert result.status == "ok"
        assert path.is_file()
        assert result.identity.job_index == 1
        assert result.identity.job_count == 4
        assert result.identity.order_seed == 7
        assert [sample.phase for sample in result.frames] == ["warmup"] * 2 + ["steady"] * 4
        assert [sample.source_index for sample in result.frames] == [0, 1, 2, 5, 8, 11]
        assert result.inputs.warmup_indices == [0, 1]
        assert result.inputs.measured_indices == [2, 5, 8, 11]
        assert len(result.steady_state()) == 4
        assert result.phases.steady_wall_seconds > 0.0
        assert result.inputs.checkpoint_sha256 == DRY_RUN_CHECKPOINT
        assert result.dry_run is True

    def test_mesh_output_pays_for_a_rest_body_once_per_track(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        suite_path = self._prepare(make_video, write_config)
        suite = load_suite_config(suite_path)

        result, _ = measure_job(
            suite,
            Job(index=0, cell="a-mesh", repetition=0),
            suite_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
            job_count=4,
            estimator=DryRunEstimator(BodiesConfig()),
            dry_run=True,
        )

        # The stand-in returns track 0 every frame: one rest body, on the
        # first frame, and none after.
        assert [sample.new_tracks for sample in result.frames] == [1, 0, 0, 0, 0, 0]
        assert result.frames[0].rest_body_bytes > 0
        assert all(sample.rest_body_bytes == 0 for sample in result.frames[1:])
        # Bones travel in mesh mode, so the message is bigger than a skeleton's.
        skeleton, _ = measure_job(
            suite,
            Job(index=2, cell="a-skeleton", repetition=0),
            suite_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
            job_count=4,
            estimator=DryRunEstimator(BodiesConfig()),
            dry_run=True,
        )
        assert result.frames[2].message_bytes > skeleton.frames[2].message_bytes

    def test_the_geometry_records_what_the_model_actually_sees(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        suite_path = self._prepare(make_video, write_config)
        suite = load_suite_config(suite_path)

        result, _ = measure_job(
            suite,
            Job(index=0, cell="a-skeleton", repetition=0),
            suite_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
            job_count=4,
            estimator=DryRunEstimator(BodiesConfig()),
            dry_run=True,
        )

        # The fixture is 64x48; the model's long edge is 768, padded to patches.
        assert (result.geometry.source_width, result.geometry.source_height) == (64, 48)
        assert result.geometry.model_input_width == 768
        assert result.geometry.model_input_height == 576
        assert result.geometry.scale == pytest.approx(12.0)

    def test_an_estimator_that_dies_leaves_a_failed_result_with_a_prefix_of_the_plan(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        suite_path = self._prepare(make_video, write_config)
        suite = load_suite_config(suite_path)

        class _DiesOnThird(DryRunSession):
            def __init__(self) -> None:
                super().__init__(bone_count=4)
                self.calls = 0

            def frame(
                self, image: FrameInput, config: BodiesConfig, *, width: int, height: int
            ) -> FrameEstimate:
                self.calls += 1
                if self.calls == 3:
                    raise OSError("the model went away")
                return super().frame(image, config, width=width, height=height)

        class _Estimator(DryRunEstimator):
            def open(self) -> DryRunSession:
                return _DiesOnThird()

        result, path = measure_job(
            suite,
            Job(index=0, cell="a-skeleton", repetition=0),
            suite_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
            job_count=4,
            estimator=_Estimator(BodiesConfig()),
            dry_run=True,
        )

        assert result.status == "failed"
        assert "the model went away" in (result.error or "")
        assert [sample.source_index for sample in result.frames] == [0, 1]
        assert path.is_file()

    def test_too_short_a_clip_is_refused_before_anything_is_timed(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        suite_path = self._prepare(make_video, write_config, frames=4)
        suite = load_suite_config(suite_path)

        with pytest.raises(ConfigError, match="4 frame"):
            measure_job(
                suite,
                Job(index=0, cell="a-skeleton", repetition=0),
                suite_dir=tmp_path,
                artifacts_root=tmp_path / "artifacts",
                job_count=4,
                estimator=DryRunEstimator(BodiesConfig()),
                dry_run=True,
            )


class TestRunningASuite:
    """The driver, end to end, with a real interpreter per job."""

    def test_every_job_runs_in_its_own_process_and_is_recorded(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        make_video(name="clip.avi", frame_count=12)
        write_config(_EXPERIMENT, name="fixture.toml")
        suite_path = write_config(_SUITE, name="suite.toml")
        suite = load_suite_config(suite_path)

        record = run_suite(
            suite, suite_path=suite_path, artifacts_root=tmp_path / "artifacts", dry_run=True
        )

        assert [outcome.exit_code for outcome in record.jobs] == [0, 0, 0, 0]
        assert [(o.cell, o.repetition) for o in record.jobs] == [
            (job.cell, job.repetition) for job in plan_jobs(suite)
        ]
        assert all(outcome.result is not None for outcome in record.jobs)
        # The preflight report lives beside the results (ADR-0006) and is not
        # a job: found listed as "unreadable" on every suite of 2026-09-15.
        (
            results_dir(tmp_path / "artifacts", "suite") / "preflight-20260915T160000.json"
        ).write_text('{"schema_version": 1, "suite": "suite", "ok": true}', encoding="utf-8")
        results, unreadable = read_results(results_dir(tmp_path / "artifacts", "suite"))
        assert unreadable == []
        assert len(results) == 4
        # Each result knows its place in the order and the seed that made it.
        assert sorted(r.identity.job_index for r in results) == [0, 1, 2, 3]
        assert {r.identity.order_seed for r in results} == {7}
        record_files = list(results_dir(tmp_path / "artifacts", "suite").glob("*suite-run.json"))
        assert len(record_files) == 1
        assert record.finished_at is not None

    def test_a_job_that_cannot_start_is_recorded_and_the_suite_goes_on(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        make_video(name="clip.avi", frame_count=12)
        write_config(_EXPERIMENT, name="fixture.toml")
        broken = _SUITE.replace(
            'experiment = "fixture.toml"\noutput = "mesh"',
            'experiment = "missing.toml"\noutput = "mesh"',
        )
        suite_path = write_config(broken, name="suite.toml")
        suite = load_suite_config(suite_path)

        record = run_suite(
            suite, suite_path=suite_path, artifacts_root=tmp_path / "artifacts", dry_run=True
        )

        by_cell = {(o.cell, o.repetition): o for o in record.jobs}
        assert by_cell[("a-mesh", 0)].exit_code != 0
        assert by_cell[("a-mesh", 0)].result is None
        assert by_cell[("a-skeleton", 0)].exit_code == 0
        assert len(record.jobs) == 4

    def test_only_selects_cells_but_keeps_their_planned_indices(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        make_video(name="clip.avi", frame_count=12)
        write_config(_EXPERIMENT, name="fixture.toml")
        suite_path = write_config(_SUITE, name="suite.toml")
        suite = load_suite_config(suite_path)

        record = run_suite(
            suite,
            suite_path=suite_path,
            artifacts_root=tmp_path / "artifacts",
            dry_run=True,
            only=["a-mesh"],
        )

        assert {o.cell for o in record.jobs} == {"a-mesh"}
        planned = {job.index for job in plan_jobs(suite) if job.cell == "a-mesh"}
        assert {o.index for o in record.jobs} == planned

    def test_an_unknown_cell_is_refused(self, write_config: WriteConfig, tmp_path: Path) -> None:
        suite_path = write_config(_SUITE, name="suite.toml")
        with pytest.raises(ConfigError, match="no cell"):
            run_suite(
                load_suite_config(suite_path),
                suite_path=suite_path,
                artifacts_root=tmp_path,
                dry_run=True,
                only=["nope"],
            )


def _environment() -> Environment:
    return Environment(
        machine="Apple M3 Pro",
        platform="Darwin 25.6.0 arm64",
        python_version="3.12.14",
        device="mps",
        package_versions={"torch": "2.13.0"},
        git_commit="0" * 40,
    )


def _result(
    inference: list[float],
    *,
    cell: str = "a-skeleton",
    repetition: int = 0,
    job_index: int = 0,
    output: OutputMode = "skeleton",
    warmup: int = 0,
    steady_wall: float | None = None,
    measured: list[int] | None = None,
    started: datetime = _STARTED,
    **overrides: object,
) -> JobResult:
    """A job whose per-frame timings are exactly what the test says."""
    measured_indices = measured if measured is not None else list(range(warmup, len(inference)))
    warmup_indices = list(range(warmup))
    plan = [*warmup_indices, *measured_indices]
    frames = [
        FrameSample(
            phase="warmup" if position < warmup else "steady",
            position=position,
            source_index=plan[position],
            inference_seconds=value,
            rest_body_seconds=0.0,
            serialize_seconds=0.001,
            people=2,
            new_tracks=0,
            message_bytes=500,
            rest_body_bytes=0,
        )
        for position, value in enumerate(inference)
    ]
    defaults: dict[str, object] = {
        "identity": JobIdentity(
            suite="suite",
            suite_hash="a" * 64,
            cell=cell,
            output=output,
            repetition=repetition,
            job_index=job_index,
            job_count=4,
            order_seed=7,
        ),
        "started_at": started,
        "finished_at": started + timedelta(seconds=1),
        "inputs": JobInputs(
            experiment="fixture",
            source_sha256="a" * 64,
            ingest_hash="0123456789ab",
            bodies_config_hash="cafecafecafe",
            checkpoint_sha256="f" * 64,
            backbone_ref="0" * 40,
            warmup_indices=warmup_indices,
            measured_indices=measured_indices,
        ),
        "environment": _environment(),
        "conditions": Conditions(power_source="mains", cpu_speed_limit_before=100),
        "geometry": InputGeometry(
            source_width=1920,
            source_height=1080,
            model_input_width=768,
            model_input_height=432,
            scale=0.4,
        ),
        "phases": Phases(
            model_load_seconds=9.5,
            session_open_seconds=0.01,
            steady_wall_seconds=(
                steady_wall
                if steady_wall is not None
                else sum(inference[warmup:]) + 0.001 * (len(inference) - warmup)
            ),
        ),
        "memory": Memory(
            rss_baseline_bytes=100,
            rss_after_load_bytes=2_100,
            rss_end_bytes=2_200,
            peak_rss_bytes=2_300,
            metal_driver_peak_bytes=5_000,
        ),
        "frames": frames,
    }
    return JobResult.model_validate(defaults | overrides)


class TestTheJobResultContract:
    def test_warm_up_is_not_part_of_the_steady_state(self) -> None:
        result = _result([0.9, 0.5, 0.5, 0.5], warmup=1)

        assert [sample.inference_seconds for sample in result.steady_state()] == [0.5, 0.5, 0.5]

    def test_frames_that_disagree_with_the_plan_are_refused(self) -> None:
        """A job that timed other frames than it says is not the job it says."""
        data = _result([0.5, 0.5]).model_dump()
        data["frames"][1]["source_index"] = 7

        with pytest.raises(ValueError, match="plan"):
            JobResult.model_validate(data)

    def test_a_failed_job_may_stop_partway_but_not_elsewhere(self) -> None:
        data = _result([0.5, 0.5, 0.5]).model_dump()
        data.update(status="failed", error="OSError: gone")
        data["frames"] = data["frames"][:1]

        partial = JobResult.model_validate(data)
        assert partial.status == "failed"

        data["frames"][0]["source_index"] = 2
        with pytest.raises(ValueError, match="prefix"):
            JobResult.model_validate(data)

    def test_an_unreadable_schema_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            _result([0.5], schema_version=JOB_RESULT_SCHEMA_VERSION + 1)

    def test_it_round_trips_through_json(self) -> None:
        result = _result([0.5, 0.6])

        assert JobResult.model_validate_json(result.model_dump_json()) == result


class TestTheSummary:
    def test_the_same_inputs_produce_the_same_bytes(self) -> None:
        results = [_result([0.4, 0.5, 0.6]), _result([0.7], cell="a-mesh", output="mesh")]
        pairs = [Pair(a="a-skeleton", b="a-mesh", question="q")]

        first = summarise("suite", results, pairs=pairs).model_dump_json(indent=2)
        second = summarise("suite", list(reversed(results)), pairs=pairs).model_dump_json(indent=2)

        assert first == second
        assert to_csv(summarise("suite", results)) == to_csv(summarise("suite", results))

    def test_warm_up_never_reaches_the_median(self) -> None:
        summary = summarise("suite", [_result([9.0, 0.5, 0.5, 0.5], warmup=1)])

        cell = summary.cells[0]
        assert cell.inference.samples == 3
        assert cell.inference.median == 0.5
        assert cell.warmup_inference.samples == 1

    def test_the_spread_is_nearest_rank(self) -> None:
        summary = summarise("suite", [_result([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 5.0])])

        cell = summary.cells[0].inference
        assert cell.p25 == 0.3
        assert cell.p75 == 0.8
        assert cell.p90 == 0.9
        assert cell.p95 == 5.0

    def test_throughput_is_frames_over_wall_clock_not_a_reciprocal(self) -> None:
        """Three frames at 0.5 s each but a 3 s phase: something stalled, and
        the throughput says 1 fps, not 2."""
        summary = summarise("suite", [_result([0.5, 0.5, 0.5], steady_wall=3.0)])

        assert summary.cells[0].throughput_fps == [1.0]
        assert summary.cells[0].throughput_fps_median == 1.0

    def test_each_job_contributes_its_own_model_load(self) -> None:
        summary = summarise(
            "suite",
            [
                _result([0.5], job_index=3, repetition=1),
                _result([0.5], job_index=0, repetition=0),
            ],
        )

        # In job order, not file order.
        assert summary.cells[0].job_indices == [0, 3]
        assert summary.cells[0].model_load_seconds == [9.5, 9.5]

    def test_a_dry_run_is_not_averaged_with_a_real_one(self) -> None:
        summary = summarise("suite", [_result([0.5, 0.5]), _result([0.000002], dry_run=True)])

        assert summary.dry_runs_excluded == 1
        assert summary.cells[0].inference.median == 0.5

    def test_a_failed_job_stays_visible_and_out_of_the_aggregate(self) -> None:
        summary = summarise(
            "suite",
            [
                _result([0.5, 0.5]),
                _result(
                    [9.0], status="failed", error="OSError: gone", measured=[0, 1], job_index=1
                ),
            ],
        )

        cell = summary.cells[0]
        assert cell.jobs == 2
        assert cell.failed_jobs == 1
        assert cell.inference.median == 0.5
        assert "gone" in cell.errors[0]

    def test_the_protocol_minimums_are_reported_not_enforced(self) -> None:
        short = summarise("suite", [_result([0.5] * 4)]).cells[0]
        assert short.meets_protocol is False

        enough = summarise(
            "suite",
            [
                _result([0.5] * PROTOCOL_MEASURED_FRAMES, repetition=index, job_index=index)
                for index in range(PROTOCOL_REPETITIONS)
            ],
        ).cells[0]
        assert enough.meets_protocol is True

    def test_jobs_that_timed_different_frames_are_flagged(self) -> None:
        summary = summarise(
            "suite",
            [
                _result([0.5, 0.5], measured=[0, 1]),
                _result([0.5, 0.5], measured=[0, 2], job_index=1),
            ],
        )

        assert summary.cells[0].consistent_frames is False
        assert summary.cells[0].meets_protocol is False

    def test_a_pair_differences_the_same_frame_of_the_same_repetition(self) -> None:
        a = _result([0.50, 0.60, 0.70], cell="a-skeleton")
        b = _result([0.52, 0.66, 0.69], cell="a-mesh", output="mesh", job_index=1)

        summary = summarise("suite", [a, b], pairs=[Pair(a="a-skeleton", b="a-mesh", question="q")])

        pair = summary.pairs[0]
        assert pair.pairs == 3
        assert pair.same_frames is True
        # b - a per frame: +0.02, +0.06, -0.01 -> median +0.02.
        assert pair.difference.median == pytest.approx(0.02)
        assert pair.fraction_b_slower == pytest.approx(2 / 3)
        # Totals carry the 1 ms serialisation each: 0.661 over 0.601.
        assert pair.median_ratio == pytest.approx(0.661 / 0.601, rel=1e-3)

    def test_a_pair_over_different_frames_says_so(self) -> None:
        a = _result([0.5, 0.5], cell="a-skeleton", measured=[0, 1])
        b = _result([0.5, 0.5], cell="a-mesh", output="mesh", measured=[0, 3], job_index=1)

        summary = summarise("suite", [a, b], pairs=[Pair(a="a-skeleton", b="a-mesh", question="q")])

        assert summary.pairs[0].same_frames is False

    def test_the_model_footprint_is_after_load_minus_baseline(self) -> None:
        summary = summarise("suite", [_result([0.5])])

        assert summary.cells[0].model_footprint_bytes == 2_000
        assert summary.cells[0].peak_rss_bytes == 2_300


class TestTheCsvs:
    def test_a_missing_measurement_is_empty_rather_than_zero(self) -> None:
        rows = to_csv(summarise("suite", [_result([0.9], warmup=1)])).splitlines()

        header, row = rows[0].split(","), rows[1].split(",")
        assert row[header.index("inference_median_s")] == ""
        assert row[header.index("meets_protocol")] == "false"

    def test_pairs_carry_their_question(self) -> None:
        a = _result([0.5, 0.6], cell="a-skeleton")
        b = _result([0.6, 0.7], cell="a-mesh", output="mesh", job_index=1)
        summary = summarise(
            "suite", [a, b], pairs=[Pair(a="a-skeleton", b="a-mesh", question="output mode")]
        )

        rows = pairs_to_csv(summary).splitlines()
        header, row = rows[0].split(","), rows[1].split(",")
        assert row[header.index("question")] == "output mode"
        assert row[header.index("difference_median_s")] == "0.1"

    def test_all_three_forms_land_beside_the_results(self, tmp_path: Path) -> None:
        as_json, as_csv, as_pairs = write_summary(summarise("suite", [_result([0.5])]), tmp_path)

        assert as_json == tmp_path / SUMMARY_FILENAME
        assert as_csv == tmp_path / SUMMARY_CSV_FILENAME
        assert as_pairs == tmp_path / PAIRS_CSV_FILENAME
        # The summary is not read back as a result.
        results, unreadable = read_results(tmp_path)
        assert results == [] and unreadable == []


class TestReadingResults:
    def test_a_corrupt_file_is_reported_by_name(self, tmp_path: Path) -> None:
        (tmp_path / "cell").mkdir()
        (tmp_path / "cell" / "broken.json").write_text("{ no", "utf-8")
        (tmp_path / "cell" / "good.json").write_text(_result([0.5]).model_dump_json(), "utf-8")

        results, unreadable = read_results(tmp_path)

        assert len(results) == 1
        assert unreadable and "broken.json" in unreadable[0]

    def test_the_run_record_is_not_read_as_a_result(self, tmp_path: Path) -> None:
        (tmp_path / "20260915T000000000000-suite-run.json").write_text(
            json.dumps({"suite": "x"}), "utf-8"
        )

        assert read_results(tmp_path) == ([], [])


class TestConditions:
    def test_mains_low_power_and_a_thermal_limit_are_read_off_pmset(self) -> None:
        conditions = parse_conditions(
            {
                "batt": "Now drawing from 'AC Power'\n",
                "settings": "System-wide power settings:\n lowpowermode         1\n",
                "therm": "CPU_Speed_Limit \t= 78\n",
            }
        )

        assert conditions.power_source == "mains"
        assert conditions.low_power_mode is True
        assert conditions.cpu_speed_limit_before == 78

    def test_battery_is_read_and_absent_fields_stay_unknown(self) -> None:
        conditions = parse_conditions({"batt": "Now drawing from 'Battery Power'\n"})

        assert conditions.power_source == "battery"
        assert conditions.low_power_mode is None
        assert conditions.cpu_speed_limit_before is None

    def test_nothing_readable_is_unknown_not_mains(self) -> None:
        assert parse_conditions({}).power_source == "unknown"
