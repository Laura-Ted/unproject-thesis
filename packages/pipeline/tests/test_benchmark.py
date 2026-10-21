"""Tests for measurement orchestration and for the arithmetic on top of it.

The model never runs here. What is under test is which samples are allowed to
reach a number, whether the same inputs always produce the same number, and
whether a run that measured nothing can be mistaken for one that did.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fixture_types import MakeVideo, WriteConfig
from unproject.benchmark.aggregate import (
    SUMMARY_CSV_FILENAME,
    SUMMARY_FILENAME,
    read_results,
    summarise,
    to_csv,
    write_summary,
)
from unproject.benchmark.artifacts import (
    BENCHMARK_RESULT_SCHEMA_VERSION,
    BenchmarkResult,
    Environment,
    FrameSample,
    Inputs,
    Repetition,
)
from unproject.benchmark.config import BenchmarkConfig, load_benchmark_config
from unproject.benchmark.dryrun import DRY_RUN_CHECKPOINT, DryRunEstimator, DryRunSession
from unproject.benchmark.run import BenchmarkError, results_dir, run_benchmark, write_result
from unproject.config import BodiesConfig, ConfigError

_STARTED = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)


def _environment() -> Environment:
    return Environment(
        machine="Apple M3 Pro",
        platform="Darwin 25.6.0 arm64",
        python_version="3.12.14",
        device="mps",
        package_versions={"torch": "2.9.0"},
        git_commit="0" * 40,
    )


def _inputs() -> Inputs:
    return Inputs(
        experiment="fixture",
        source_sha256="a" * 64,
        ingest_hash="0123456789ab",
        benchmark_config_hash="ba9876543210",
        bodies_config_hash="cafecafecafe",
        checkpoint_sha256="f" * 64,
        backbone_ref="0" * 40,
    )


def _config(**overrides: object) -> BenchmarkConfig:
    defaults: dict[str, object] = {
        "name": "fixture-skeleton",
        "experiment": Path("../fixture.toml"),
        "warmup_frames": 1,
        "measured_frames": 3,
        "repetitions": 1,
    }
    return BenchmarkConfig.model_validate(defaults | overrides)


def _result(
    seconds: list[float],
    *,
    name: str = "fixture-skeleton",
    warmup: int = 0,
    started: datetime = _STARTED,
    **overrides: object,
) -> BenchmarkResult:
    """A result whose per-frame timings are exactly what the test says."""
    defaults: dict[str, object] = {
        "name": name,
        "started_at": started,
        "finished_at": started + timedelta(seconds=1),
        "config": _config(name=name),
        "environment": _environment(),
        "inputs": _inputs(),
        "repetitions": [Repetition(index=0, cold_start_seconds=10.0, peak_memory_bytes=1_000)],
        "frames": [
            FrameSample(
                repetition=0,
                frame_index=index,
                warmup=index < warmup,
                seconds=value,
                people=2,
                width=1280,
                height=720,
            )
            for index, value in enumerate(seconds)
        ],
    }
    return BenchmarkResult.model_validate(defaults | overrides)


class TestTheConfig:
    def test_a_typo_is_refused_rather_than_defaulted(self, write_config: WriteConfig) -> None:
        path = write_config(
            'name = "x"\nexperiment = "e.toml"\nmeasured_frame = 4\n', name="bench.toml"
        )
        with pytest.raises(ConfigError, match="measured_frame"):
            load_benchmark_config(path)

    def test_an_unsupported_schema_version_is_refused(self, write_config: WriteConfig) -> None:
        path = write_config(
            'schema_version = 99\nname = "x"\nexperiment = "e.toml"\n', name="bench.toml"
        )
        with pytest.raises(ConfigError, match="schema_version"):
            load_benchmark_config(path)

    def test_it_reads_a_whole_config(self, write_config: WriteConfig) -> None:
        path = write_config(
            'name = "crowd-mesh"\nexperiment = "../crowd.toml"\n'
            "warmup_frames = 5\nmeasured_frames = 60\nrepetitions = 3\n"
            'output = "mesh"\n',
            name="bench.toml",
        )

        config = load_benchmark_config(path)

        assert config.name == "crowd-mesh"
        assert config.output == "mesh"
        assert config.measured_frames == 60


class TestTheResultContract:
    def test_warm_up_is_not_part_of_the_steady_state(self) -> None:
        result = _result([0.9, 0.5, 0.5, 0.5], warmup=1)

        assert [sample.seconds for sample in result.steady_state()] == [0.5, 0.5, 0.5]

    def test_a_sample_cannot_belong_to_a_repetition_that_was_not_run(self) -> None:
        # Otherwise a stray sample joins an aggregate with nothing describing
        # the pass it came from.
        with pytest.raises(ValueError, match="repetition \\[7\\]"):
            _result(
                [0.5],
                frames=[
                    FrameSample(
                        repetition=7,
                        frame_index=0,
                        warmup=False,
                        seconds=0.5,
                        people=1,
                        width=64,
                        height=48,
                    )
                ],
            )

    def test_a_result_round_trips_through_disk(self, tmp_path: Path) -> None:
        result = _result([0.5, 0.6])

        path = write_result(result, tmp_path)

        assert BenchmarkResult.model_validate_json(path.read_text("utf-8")) == result
        assert path.parent == results_dir(tmp_path, "fixture-skeleton")

    def test_a_second_run_adds_a_file_rather_than_replacing_one(self, tmp_path: Path) -> None:
        """Raw observations are append-only; a repeat is more evidence, not new evidence."""
        first = write_result(_result([0.5]), tmp_path)
        second = write_result(_result([0.6], started=_STARTED + timedelta(seconds=30)), tmp_path)

        assert first != second
        assert first.is_file()
        assert len(list(results_dir(tmp_path, "fixture-skeleton").glob("*.json"))) == 2

    def test_an_unreadable_schema_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            _result([0.5], schema_version=BENCHMARK_RESULT_SCHEMA_VERSION + 1)


class TestAggregation:
    def test_the_same_inputs_produce_the_same_bytes(self, tmp_path: Path) -> None:
        """A figure is regenerated rather than kept, so this has to hold."""
        results = [_result([0.4, 0.5, 0.6]), _result([0.7], name="other")]

        first = summarise(results).model_dump_json(indent=2)
        second = summarise(results).model_dump_json(indent=2)

        assert first == second
        assert to_csv(summarise(results)) == to_csv(summarise(results))

    def test_the_order_results_were_read_in_does_not_matter(self) -> None:
        results = [_result([0.4]), _result([0.5], name="other"), _result([0.6])]

        forward = summarise(results)
        backward = summarise(list(reversed(results)))

        assert forward == backward

    def test_warm_up_never_reaches_the_median(self) -> None:
        # One nine-second warm-up frame against three half-second ones: if it
        # got in, the median would say so.
        summary = summarise([_result([9.0, 0.5, 0.5, 0.5], warmup=1)])

        assert summary.benchmarks[0].latency.samples == 3
        assert summary.benchmarks[0].latency.median_seconds == 0.5

    def test_the_tail_percentile_is_a_value_that_was_measured(self) -> None:
        summary = summarise([_result([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 5.0])])

        # Nearest-rank: the 90th of ten samples is the ninth, not an average of
        # two of them and not a number nothing recorded.
        assert summary.benchmarks[0].latency.p90_seconds == 0.9
        assert summary.benchmarks[0].latency.max_seconds == 5.0

    def test_cold_and_warm_starts_are_kept_apart(self) -> None:
        result = _result(
            [0.5],
            repetitions=[
                Repetition(index=0, cold_start_seconds=20.0, peak_memory_bytes=1),
                Repetition(index=1, cold_start_seconds=4.0, peak_memory_bytes=1),
            ],
        )

        summary = summarise([result]).benchmarks[0]

        assert summary.cold_start_seconds == [20.0]
        assert summary.warm_start_seconds == [4.0]

    def test_a_dry_run_is_not_averaged_with_a_real_one(self) -> None:
        """Microseconds from a stand-in would not show up as missing, but as wrong."""
        real = _result([0.5, 0.5])
        fake = _result([0.000002, 0.000002], dry_run=True)

        summary = summarise([real, fake])

        assert summary.dry_runs_excluded == 1
        assert summary.benchmarks[0].latency.median_seconds == 0.5

    def test_a_directory_of_only_dry_runs_summarises_nothing(self) -> None:
        summary = summarise([_result([0.1], dry_run=True)])

        assert summary.benchmarks == []
        assert summary.dry_runs_excluded == 1

    def test_a_failure_stays_visible_in_the_summary(self) -> None:
        summary = summarise([_result([0.5], status="failed", error="OSError: the model went away")])

        assert summary.benchmarks[0].failed_runs == 1
        assert "the model went away" in summary.benchmarks[0].errors[0]

    def test_a_result_that_stopped_parsing_is_named_rather_than_skipped(self) -> None:
        summary = summarise([_result([0.5])], ["bad.json: not valid JSON"])

        assert "bad.json" in summary.benchmarks[0].errors[0]

    def test_a_benchmark_with_no_steady_state_reports_no_median(self) -> None:
        summary = summarise([_result([0.9], warmup=1)])

        assert summary.benchmarks[0].latency.samples == 0
        assert summary.benchmarks[0].latency.median_seconds is None


class TestTheCsv:
    def test_a_missing_measurement_is_empty_rather_than_zero(self) -> None:
        """Zero is a measurement. Nothing is not."""
        rows = to_csv(summarise([_result([0.9], warmup=1)])).splitlines()

        header, row = rows[0].split(","), rows[1].split(",")
        assert row[header.index("median_seconds")] == ""
        assert row[header.index("metal_driver_bytes")] == ""

    def test_it_carries_the_numbers_the_summary_holds(self) -> None:
        rows = to_csv(summarise([_result([0.4, 0.5, 0.6])])).splitlines()

        header, row = rows[0].split(","), rows[1].split(",")
        assert row[header.index("name")] == "fixture-skeleton"
        assert row[header.index("median_seconds")] == "0.5"
        assert row[header.index("samples")] == "3"


class TestReadingAndWriting:
    def test_it_reads_back_what_was_written(self, tmp_path: Path) -> None:
        write_result(_result([0.5]), tmp_path)
        write_result(_result([0.6], name="other"), tmp_path)

        results, unreadable = read_results(tmp_path)

        assert sorted(result.name for result in results) == ["fixture-skeleton", "other"]
        assert unreadable == []

    def test_a_corrupt_file_is_reported_by_name(self, tmp_path: Path) -> None:
        write_result(_result([0.5]), tmp_path)
        (results_dir(tmp_path, "fixture-skeleton") / "broken.json").write_text("{ no", "utf-8")

        results, unreadable = read_results(tmp_path)

        assert len(results) == 1
        assert unreadable and "broken.json" in unreadable[0]

    def test_the_summary_it_wrote_is_not_read_back_as_a_result(self, tmp_path: Path) -> None:
        write_result(_result([0.5]), tmp_path)
        write_summary(summarise(read_results(tmp_path)[0]), tmp_path)

        results, unreadable = read_results(tmp_path)

        assert len(results) == 1
        assert unreadable == []

    def test_both_forms_land_beside_the_results(self, tmp_path: Path) -> None:
        as_json, as_csv = write_summary(summarise([_result([0.5])]), tmp_path)

        assert as_json == tmp_path / SUMMARY_FILENAME
        assert as_csv == tmp_path / SUMMARY_CSV_FILENAME
        assert as_csv.read_text("utf-8").startswith("name,output,")


_EXPERIMENT = """
name = "fixture"
[ingest]
source_path = "clip.avi"
"""


class TestOrchestration:
    """End to end on the stand-in estimator: no weights, no network, no model."""

    def _configs(self, write_config: WriteConfig, body: str) -> tuple[Path, Path]:
        experiment = write_config(_EXPERIMENT, name="fixture.toml")
        benchmark = write_config(body, name="bench.toml")
        return benchmark, experiment

    def test_it_measures_every_frame_of_every_repetition(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        make_video(name="clip.avi", frame_count=6)
        benchmark, _ = self._configs(
            write_config,
            'name = "fixture-skeleton"\nexperiment = "fixture.toml"\n'
            "warmup_frames = 1\nmeasured_frames = 3\nrepetitions = 2\n",
        )

        result, path = run_benchmark(
            load_benchmark_config(benchmark),
            config_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
            estimator=DryRunEstimator(BodiesConfig()),
            dry_run=True,
        )

        assert result.status == "ok"
        assert len(result.repetitions) == 2
        assert len(result.frames) == 8
        assert len(result.steady_state()) == 6
        assert path.is_file()

    def test_the_first_frames_of_each_repetition_are_marked_warm_up(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        make_video(name="clip.avi", frame_count=6)
        benchmark, _ = self._configs(
            write_config,
            'name = "fixture-skeleton"\nexperiment = "fixture.toml"\n'
            "warmup_frames = 2\nmeasured_frames = 2\nrepetitions = 2\n",
        )

        result, _ = run_benchmark(
            load_benchmark_config(benchmark),
            config_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
            estimator=DryRunEstimator(BodiesConfig()),
            dry_run=True,
        )

        for repetition in (0, 1):
            marked = [
                sample.frame_index
                for sample in result.frames
                if sample.repetition == repetition and sample.warmup
            ]
            assert marked == [0, 1]

    def test_it_records_what_the_run_was_made_of(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        make_video(name="clip.avi", frame_count=6)
        benchmark, _ = self._configs(
            write_config,
            'name = "fixture-skeleton"\nexperiment = "fixture.toml"\n'
            "warmup_frames = 0\nmeasured_frames = 2\nrepetitions = 1\n",
        )

        result, _ = run_benchmark(
            load_benchmark_config(benchmark),
            config_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
            estimator=DryRunEstimator(BodiesConfig()),
            dry_run=True,
        )

        # Enough to find the footage and the code again, per ADR-0002 §7.5.
        assert len(result.inputs.source_sha256) == 64
        assert result.inputs.experiment == "fixture"
        assert result.inputs.checkpoint_sha256 == DRY_RUN_CHECKPOINT
        assert result.environment.python_version
        assert result.dry_run is True

    def test_asking_for_more_frames_than_exist_is_refused(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        """Quietly measuring four frames when sixty were asked for is the worse failure."""
        make_video(name="clip.avi", frame_count=4)
        benchmark, _ = self._configs(
            write_config,
            'name = "fixture-skeleton"\nexperiment = "fixture.toml"\n'
            "warmup_frames = 5\nmeasured_frames = 60\nrepetitions = 1\n",
        )

        with pytest.raises(BenchmarkError, match="4 frame"):
            run_benchmark(
                load_benchmark_config(benchmark),
                config_dir=tmp_path,
                artifacts_root=tmp_path / "artifacts",
                estimator=DryRunEstimator(BodiesConfig()),
                dry_run=True,
            )

    def test_an_estimator_that_dies_still_leaves_a_result(
        self, make_video: MakeVideo, write_config: WriteConfig, tmp_path: Path
    ) -> None:
        make_video(name="clip.avi", frame_count=6)
        benchmark, _ = self._configs(
            write_config,
            'name = "fixture-skeleton"\nexperiment = "fixture.toml"\n'
            "warmup_frames = 0\nmeasured_frames = 3\nrepetitions = 1\n",
        )

        class _Dies(DryRunEstimator):
            def open(self) -> DryRunSession:
                raise OSError("the model went away")

        result, path = run_benchmark(
            load_benchmark_config(benchmark),
            config_dir=tmp_path,
            artifacts_root=tmp_path / "artifacts",
            estimator=_Dies(BodiesConfig()),
            dry_run=True,
        )

        assert result.status == "failed"
        assert "the model went away" in (result.error or "")
        assert path.is_file()
